#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, Transform, TransformedDistribution
import os
import cv2
from lerobot.policies.normalize import NormalizeBuffer
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.tqc.configuration_tqc import TQCConfig, is_image_feature
from lerobot.policies.utils import get_device_from_parameters

DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


class TQCPolicy(
    PreTrainedPolicy,
):
    config_class = TQCConfig
    name = "tqc"

    def __init__(
        self,
        config: TQCConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        dataset_stats=self.config.dataset_stats

        # Debug / analysis counters (avoid relying on external trainer global_step).
        self._critic_quantiles_dump_idx: int = 0

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features["action"].shape[0]
        self.continuous_action_dim = continuous_action_dim
        self._init_normalization(dataset_stats)
        # Initialize the observation encoder (shared between actor and critic, or separate)
        self._init_encoders()  
        # Initialize the critic networks (continuous action + optional discrete action)
        self._init_critics(continuous_action_dim)
        # Initialize the actor network (outputs a continuous action distribution)
        self._init_actor(continuous_action_dim)
        self._init_temperature()

    def get_optim_params(self) -> dict:
        """Collect the trainable parameters of each module, used to build the optimizers."""
        optim_params = {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                # With a shared encoder the actor does not optimize the encoder parameters (avoids gradient conflicts)
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic": self.critic_ensemble.parameters(),
            "temperature": self.log_alpha,
        }
        # With discrete actions, add the discrete critic parameters
        if self.config.num_discrete_actions is not None:
            optim_params["discrete_critic"] = self.discrete_critic.parameters()
        return optim_params

    def get_optimizer_and_scheduler(self):
        optim_dict = {
            "actor": torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr),
            "critic": torch.optim.Adam(list(self.critic_ensemble.parameters()), lr=self.config.critic_lr),
            "temperature": torch.optim.Adam([self.log_alpha], lr=self.config.temperature_lr),
        }
        if self.config.num_discrete_actions is not None:
            optim_dict["discrete_critic"] = torch.optim.Adam(self.discrete_critic.parameters(), lr=self.config.critic_lr)
        return optim_dict, None

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("SACPolicy does not support action chunking. It returns single actions!")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""
        """
        Select an action during inference / evaluation.
        Args:
            batch: observation dict (images (left, wrist) and state)
        Returns:
            The final action tensor (continuous action concatenated with the optional discrete action)
        """
        observations_features = None
        
        # With a shared encoder and image inputs, cache the image features (avoids re-encoding, faster)
        if self.shared_encoder and self.actor.encoder.has_images:
            # Cache and normalize image features

            observations_features = self.actor.encoder.get_cached_image_features(batch, normalize=True)
        actions, _, _ = self.actor(batch, observations_features)


        # With discrete actions, the discrete critic scores each action and the argmax is taken
        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(batch, observations_features)
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)

            actions = torch.cat([actions, discrete_action], dim=-1)
        return actions, {}


    @torch.no_grad()
    def eval_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""
        """
        Select an action during evaluation.
        Args:
            batch: observation dict (images (left, wrist) and state)
        Returns:
            The final action tensor (continuous action concatenated with the optional discrete action)
        """
        observations_features = None
        
        # With a shared encoder and image inputs, cache the image features (avoids re-encoding, faster)
        if self.shared_encoder and self.actor.encoder.has_images:
            # Cache and normalize image features

            observations_features = self.actor.encoder.get_cached_image_features(batch, normalize=True)
        _, _, actions = self.actor(batch, observations_features)


        # With discrete actions, the discrete critic scores each action and the argmax is taken
        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(batch, observations_features)
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)

            actions = torch.cat([actions, discrete_action], dim=-1)
        return actions, {}
        
    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of shape (num_critics, batch, n_quantiles) with predicted quantiles per critic.
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values

    def discrete_critic_forward(
        self, observations, use_target=False, observation_features=None
    ) -> torch.Tensor:
        """Forward pass through a discrete critic network

        Args:
            observations: Dictionary of observations
            use_target: If True, use target critics, otherwise use ensemble critics
            observation_features: Optional pre-computed observation features to avoid recomputing encoder output

        Returns:
            Tensor of Q-values from the discrete critic network
        """
        discrete_critic = self.discrete_critic_target if use_target else self.discrete_critic
        q_values = discrete_critic(observations, observation_features)
        return q_values

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "discrete_critic"] = "critic",
    ) -> dict[str, Tensor]:
        """Compute the loss for the given model

        Args:
            batch: Dictionary containing:
                - action: Action tensor
                - reward: Reward tensor
                - state: Observations tensor dict
                - next_state: Next observations tensor dict
                - done: Done mask tensor
                - observation_feature: Optional pre-computed observation features
                - next_observation_feature: Optional pre-computed next observation features
            model: Which model to compute the loss for ("actor", "critic", "discrete_critic", or "temperature")

        Returns:
            The computed loss tensor
        """
        # Extract common components from batch
        actions: Tensor = batch["action"]
        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor = batch.get("observation_feature")

        if model == "critic":
            # Extract critic-specific components
            next_observations: dict[str, Tensor] = batch["next_state"]
            rewards1: Tensor = batch["reward"]
            rewards2: Tensor = batch["complementary_info"]["first_intervene_reward"]
            # rewards = rewards2
            rewards = rewards1 + rewards2
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")

            loss_critic = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )

            return {"loss_critic": loss_critic}

        if model == "discrete_critic" and self.config.num_discrete_actions is not None:
            # Extract critic-specific components
            rewards1: Tensor = batch["reward"]
            rewards2: Tensor = batch["complementary_info"]["first_intervene_reward"]
            # rewards = rewards2
            rewards = rewards1 + rewards2
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")
            complementary_info = batch.get("complementary_info")
            loss_discrete_critic = self.compute_loss_discrete_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
            )
            return {"loss_discrete_critic": loss_discrete_critic}
        if model == "actor":
            loss_actor_dict = self.compute_loss_actor(
                    observations=observations,
                    observation_features=observation_features,
                )
            return loss_actor_dict

        if model == "temperature":
            return {
                "loss_temperature": self.compute_loss_temperature(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

        raise ValueError(f"Unknown model type: {model}")

    def draw_critic_qc(self, action: np.ndarray, device: torch.device, policy_obs: dict[str, Tensor]) -> None:
        """Visualize critic_qc quantile distribution as an on-screen bar chart.
        Bar colors map to Q values via a colormap (blue=low, red=high).

        Args:
            critic_qc: numpy array of shape (batch, n_quantiles) or (n_quantiles,).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import cv2

        with torch.no_grad():
            action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(device)
            action_tensor = action_tensor[:, :self.continuous_action_dim]
            critic_qc = self.critic_forward(policy_obs, action_tensor, use_target=False, observation_features=None)
            critic_qc = critic_qc.min(dim=0)[0].cpu().numpy()

        VMIN, VMAX = -1.0, 5.0

        qc = critic_qc.flatten()
        qc_clipped = np.clip(qc, VMIN, VMAX)
        n_q = len(qc_clipped)
        tau = np.linspace(0.5 / n_q, 1.0 - 0.5 / n_q, n_q)

        norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)
        cmap = cm.get_cmap("coolwarm")
        colors = [cmap(norm(v)) for v in qc_clipped]

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#2d2d2d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

        ax.bar(tau, qc_clipped, width=0.9 / n_q, color=colors, alpha=0.9)
        ax.set_ylim(VMIN, VMAX)
        mean_val = float(np.mean(qc_clipped))
        ax.axhline(mean_val, color="yellow", linestyle="--", linewidth=1.2,
                   label=f"mean = {mean_val:.4f}  (raw mean = {float(np.mean(qc)):.4f})")
        ax.set_xlabel("Quantile level τ")
        ax.set_ylabel("Q value")
        ax.set_title("critic_q — current state")
        ax.legend(facecolor="#3d3d3d", labelcolor="white", framealpha=0.8)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        plt.tight_layout(pad=1.5)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        cv2.imshow("critic_q", img[:, :, ::-1])
        cv2.waitKey(1)

    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )
        if self.config.num_discrete_actions is not None:
            for target_param, param in zip(
                self.discrete_critic_target.parameters(),
                self.discrete_critic.parameters(),
                strict=True,
            ):
                target_param.data.copy_(
                    param.data * self.config.critic_target_update_weight
                    + target_param.data * (1.0 - self.config.critic_target_update_weight)
                )

    def update_temperature(self):
        self.temperature = self.log_alpha.exp().item()

    def _quantile_huber_loss_tqc(self, quantiles: Tensor, target_atoms: Tensor) -> Tensor:
        """Quantile Huber loss matching the reference TQC implementation (BayesGroup).

        Args:
            quantiles: (batch, num_critics, n_quantiles) online predictions.
            target_atoms: (batch, num_target_atoms) Bellman target samples (truncated mixture).

        Returns:
            Scalar loss (mean over batch, critics, quantiles, and target atoms).
        """
        # pairwise_delta: batch x nets x quantiles x samples
        pairwise_delta = target_atoms[:, None, None, :] - quantiles[:, :, :, None]
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(
            abs_pairwise_delta > 1.0,
            abs_pairwise_delta - 0.5,
            pairwise_delta**2 * 0.5,
        )
        n_quantiles = quantiles.shape[2]
        tau = (
            torch.arange(n_quantiles, device=quantiles.device, dtype=quantiles.dtype) / n_quantiles
            + 0.5 / n_quantiles
        )
        loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0.0).float()) * huber_loss).mean()
        return loss

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.actor(next_observations, next_observation_features)

            q_targets = self.critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
                observation_features=next_observation_features,
            )

            num_critics_used, batch_size, n_quantiles = q_targets.shape
            next_z = q_targets.permute(1, 0, 2).reshape(batch_size, num_critics_used * n_quantiles)
            sorted_z, _ = torch.sort(next_z, dim=-1)
            top_drop = self.config.tqc_top_quantiles_to_drop_per_net * num_critics_used
            n_atoms = sorted_z.shape[-1] - top_drop
            sorted_z_part = sorted_z[:, :n_atoms]

            # if self.config.use_backup_entropy:
            #     sorted_z_part = sorted_z_part - (self.temperature * next_log_probs).unsqueeze(-1)

            rewards_b = rewards.reshape(batch_size, -1)
            done_b = done.to(dtype=sorted_z_part.dtype).reshape(batch_size, -1)
            # done_b = done.reshape(batch_size, -1)
            td_atoms = rewards_b + (1.0 - done_b) * self.config.discount * sorted_z_part

        actions = actions[:, : self.continuous_action_dim]
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        # Dump per-critic quantiles to .npy for analysis.
        # q_preds shape: (num_critics, batch, n_quantiles)
        if self.config.dump_critic_quantiles_npy:
            dump_idx = int(self._critic_quantiles_dump_idx)
            if dump_idx % int(self.config.dump_critic_quantiles_every) == 0:
                try:
                    dump_dir = self.config.dump_critic_quantiles_dir
                    os.makedirs(dump_dir, exist_ok=True)

                    # q_cpu = q_preds.detach().to("cpu")
                    # q_mean_cpu = q_cpu.mean(dim=1)  # (num_critics, n_quantiles)

                    # np.save(os.path.join(dump_dir, f"q_preds_quantiles_{dump_idx:012d}.npy"), q_cpu.numpy())
                    # np.save(
                    #     os.path.join(dump_dir, f"q_preds_quantiles_meanB_{dump_idx:012d}.npy"),
                    #     q_mean_cpu.numpy(),
                    # )

                    # Also dump sorted target atoms after TQC truncation (and optional entropy backup).
                    # sorted_z_part shape: (B, n_atoms)
                    sorted_cpu = sorted_z_part.detach().to("cpu")
                    np.save(
                        os.path.join(dump_dir, f"sorted_z_part_{dump_idx:012d}.npy"),
                        sorted_cpu.numpy(),
                    )
                    if self.config.dump_critic_quantiles_max_files is not None:
                        max_files = int(self.config.dump_critic_quantiles_max_files)
                        # Each dump writes 2 files; keep at most max_files dumps (best effort).
                        if max_files > 0 and dump_idx // int(self.config.dump_critic_quantiles_every) >= max_files:
                            self.config.dump_critic_quantiles_npy = False
                except Exception:
                    # Best-effort debug utility; never break training due to I/O issues.
                    pass
            self._critic_quantiles_dump_idx += 1

        # (num_critics, batch, n_quantiles) -> (batch, num_critics, n_quantiles)
        q_perm = q_preds.permute(1, 0, 2).contiguous()
        critic_loss = self._quantile_huber_loss_tqc(q_perm, td_atoms)
        return critic_loss

    def compute_loss_discrete_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features=None,
        next_observation_features=None,
        complementary_info=None,
    ):
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        actions_discrete: Tensor = actions[:, DISCRETE_DIMENSION_INDEX:].clone()
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()
        discrete_penalties: Tensor | None = None

        if complementary_info is not None:
            discrete_penalties: Tensor | None = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            # For DQN, select actions using online network, evaluate with target network
            next_discrete_qs = self.discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )

            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)
    

            # Get target Q-values from target network
            target_next_discrete_qs = self.discrete_critic_forward(
                observations=next_observations,
                use_target=True,
                observation_features=next_observation_features,
            )
         

            # Use gather to select Q-values for best actions
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=1, index=best_next_discrete_action
            ).squeeze(-1)


            # Compute target Q-value with Bellman equation
            rewards_discrete = rewards
            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties

            
            target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q
            # print("target_discrete_q:", target_discrete_q)
        # Get predicted Q-values for current observations
        predicted_discrete_qs = self.discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)
        
        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)

        return discrete_critic_loss

    def compute_loss_temperature(self, observations, observation_features: Tensor | None = None) -> Tensor:
        """Compute the temperature loss"""
        # calculate temperature loss
        with torch.no_grad():
            _, log_probs, _ = self.actor(observations, observation_features)
        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
        return temperature_loss

    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        actions_pi, log_probs, _ = self.actor(observations, observation_features)

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        # # qc_preds: [n_qc_critics, B, n_quantiles] -> [B, n_quantiles]
        # q_preds_bcq = q_preds.mean(0)
        # N = q_preds_bcq.shape[-1]

        # q_for_actor = q_preds.permute(1, 0, 2).mean(dim=-1).mean(dim=-1)
        # k = max(1, int((1 - self.config.quantile_level) * N))
        # q_for_actor = q_preds_bcq[..., -k:].mean(dim=-1)

        # actor_loss = ((self.temperature * log_probs) - q_for_actor).mean()
        
        q_for_actor = q_preds.min(dim=0)[0].mean()
        actor_loss = (- q_for_actor).mean()
        # actor_loss = ((self.temperature * log_probs) - q_for_actor).mean()

        loss_from_q = (- q_for_actor)
        # loss_from_q = (- q_for_actor).mean()
        loss_from_entropy = (self.temperature * log_probs).mean()

        return {
            "loss_actor":    actor_loss,    # tensor, left for the caller to backward
            "loss_from_q":   loss_from_q,   # tensor, left for the caller to compute gradients
            "loss_from_entropy":  loss_from_entropy,  # tensor, left for the caller to compute gradients
            "min_q_preds":   q_for_actor.mean().detach().item(),
        }

    def _init_normalization(self, dataset_stats):
        """Initialize input/output normalization modules."""
        self.normalize_inputs = nn.Identity()
        self.normalize_targets = nn.Identity()
        if self.config.dataset_stats is not None:
            params = _convert_normalization_params_to_tensor(self.config.dataset_stats)
            self.normalize_inputs = NormalizeBuffer(
                self.config.input_features, self.config.normalization_mapping, params
            )
            stats = dataset_stats or params
            self.normalize_targets = NormalizeBuffer(
                self.config.output_features, self.config.normalization_mapping, stats
            )

    def _init_encoders(self):
        """Initialize encoders in the same style as RiseSingleDonePolicy.

        We keep the `shared_encoder` flag (used for actor-side detach / caching),
        but we instantiate separate encoder modules for:
        - continuous critic (online + target)
        - actor
        - optional discrete critic (online + target)
        """
        self.shared_encoder = self.config.shared_encoder

        # Continuous-action critic encoders (online + target)
        self.encoder_critic = SACObservationEncoder(self.config, self.normalize_inputs)
        self.encoder_critic_target = SACObservationEncoder(self.config, self.normalize_inputs)

        # Actor encoder
        self.encoder_actor = SACObservationEncoder(self.config, self.normalize_inputs)

        # Optional discrete-action critic encoders (online + target)
        if self.config.num_discrete_actions is not None:
            self.discrete_encoder = SACObservationEncoder(self.config, self.normalize_inputs)
            self.discrete_encoder_target = SACObservationEncoder(self.config, self.normalize_inputs)

    def _init_critics(self, continuous_action_dim):
        """Build TQC distributional critic ensemble (quantile outputs per head), targets, optional discrete critic."""
        n_q = self.config.tqc_n_quantiles
        heads = [
            QuantileCriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                n_quantiles=n_q,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = QuantileCriticEnsemble(
            encoder=self.encoder_critic, ensemble=heads, output_normalization=self.normalize_targets
        )
        target_heads = [
            QuantileCriticHead(
                input_dim=self.encoder_critic_target.output_dim + continuous_action_dim,
                n_quantiles=n_q,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = QuantileCriticEnsemble(
            encoder=self.encoder_critic_target, ensemble=target_heads, output_normalization=self.normalize_targets
        )
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            torch._functorch.config.donated_buffer = False
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()

    def _init_discrete_critics(self):
        """Build discrete discrete critic ensemble and target networks."""
        self.discrete_critic = DiscreteCritic(
            encoder=self.discrete_encoder,
            input_dim=self.discrete_encoder.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        self.discrete_critic_target = DiscreteCritic(
            encoder=self.discrete_encoder_target,
            input_dim=self.discrete_encoder_target.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )

        # TODO: (maractingi, azouitine) Compile the discrete critic
        self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())

    def _init_actor(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = continuous_action_dim + (1 if self.config.num_discrete_actions is not None else 0)
            self.target_entropy = -np.prod(dim) / 2

    def _init_temperature(self):
        """Set up temperature parameter and initial log_alpha."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))
        self.temperature = self.log_alpha.exp().item()


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: TQCConfig, input_normalizer: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.input_normalization = input_normalizer
        self._init_image_layers()
        self._init_state_layers()
        self._compute_output_dim()

    def _init_image_layers(self) -> None:
        self.image_keys = [k for k in self.config.input_features if is_image_feature(k)]
        self.has_images = bool(self.image_keys)
        if not self.has_images:
            return

        if self.config.vision_encoder_name is not None:
            self.image_encoder = PretrainedImageEncoder(self.config)
        else:
            self.image_encoder = DefaultImageEncoder(self.config)

        if self.config.freeze_vision_encoder:
            freeze_image_encoder(self.image_encoder)

        dummy = torch.zeros(1, *self.config.input_features[self.image_keys[0]].shape)
        with torch.no_grad():
            _, channels, height, width = self.image_encoder(dummy).shape

        self.spatial_embeddings = nn.ModuleDict()
        self.post_encoders = nn.ModuleDict()

        for key in self.image_keys:
            name = key.replace(".", "_")
            self.spatial_embeddings[name] = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channels,
                num_features=self.config.image_embedding_pooling_dim,
            )
            self.post_encoders[name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(
                    in_features=channels * self.config.image_embedding_pooling_dim,
                    out_features=self.config.latent_dim,
                ),
                nn.LayerNorm(normalized_shape=self.config.latent_dim),
                nn.Tanh(),
            )

    def _init_state_layers(self) -> None:
        self.has_env = "observation.environment_state" in self.config.input_features
        self.has_state = "observation.state" in self.config.input_features
        if self.has_env:
            dim = self.config.input_features["observation.environment_state"].shape[0]
            self.env_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )
        if self.has_state:
            dim = self.config.input_features["observation.state"].shape[0]
            self.state_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )

    def _compute_output_dim(self) -> None:
        out = 0
        if self.has_images:
            out += len(self.image_keys) * self.config.latent_dim
        if self.has_env:
            out += self.config.latent_dim
        if self.has_state:
            out += self.config.latent_dim
        self._out_dim = out

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ) -> Tensor:
        obs = self.input_normalization(obs)
        parts = []
        if self.has_images:
            if cache is None:
                cache = self.get_cached_image_features(obs, normalize=False)
            parts.append(self._encode_images(cache, detach))
        if self.has_env:
            parts.append(self.env_encoder(obs["observation.environment_state"]))
        if self.has_state:
            parts.append(self.state_encoder(obs["observation.state"]))
        if parts:
            return torch.cat(parts, dim=-1)

        raise ValueError(
            "No parts to concatenate, you should have at least one image or environment state or state"
        )

    def get_cached_image_features(self, obs: dict[str, Tensor], normalize: bool = False) -> dict[str, Tensor]:
        """Extract and optionally cache image features from observations.

        This function processes image observations through the vision encoder once and returns
        the resulting features.
        When the image encoder is shared between actor and critics AND frozen, these features can be safely cached and
        reused across policy components (actor, critic, discrete_critic), avoiding redundant forward passes.

        Performance impact:
        - The vision encoder forward pass is typically the main computational bottleneck during training and inference
        - Caching these features can provide 2-4x speedup in training and inference

        Normalization behavior:
        - When called from inside forward(): set normalize=False since inputs are already normalized
        - When called from outside forward(): set normalize=True to ensure proper input normalization

        Usage patterns:
        - Called in select_action() with normalize=True
        - Called in learner.py's get_observation_features() to pre-compute features for all policy components
        - Called internally by forward() with normalize=False

        Args:
            obs: Dictionary of observation tensors containing image keys
            normalize: Whether to normalize observations before encoding
                      Set to True when calling directly from outside the encoder's forward method
                      Set to False when calling from within forward() where inputs are already normalized

        Returns:
            Dictionary mapping image keys to their corresponding encoded features
        """
        """
        Compute and cache the image features (key optimization: fewer encoder calls).
        This fixes the low-FPS issue by encoding all images in one batch and caching the result.
        Args:
            normalize: whether to normalize (inputs are already normalized inside forward; enable this for external calls)
        Returns:
            A dict of image features: {image key: feature tensor}
        """
        
        if normalize:
            obs = self.input_normalization(obs)


        # Concatenate all images along the batch dim (dim=0) instead of the channel dim: two (1,3,128,128) images become (2,3,128,128)
        batched = torch.cat([obs[k]  for k in self.image_keys], dim=0)
        
        # Encode the whole batch in one pass
        out = self.image_encoder(batched)
        # Split back into per-image features
        chunks = torch.chunk(out, len(self.image_keys), dim=0)
        # Return the image feature dict: {image key: feature tensor}
        return dict(zip(self.image_keys, chunks, strict=False))

    def _encode_images(self, cache: dict[str, Tensor], detach: bool) -> Tensor:
        """Encode image features from cached observations.

        This function takes pre-encoded image features from the cache and applies spatial embeddings and post-encoders.
        It also supports detaching the encoded features if specified.

        Args:
            cache (dict[str, Tensor]): The cached image features.
            detach (bool): Usually when the encoder is shared between actor and critics,
            we want to detach the encoded features on the policy side to avoid backprop through the encoder.
            More detail here `https://cdn.aaai.org/ojs/17276/17276-13-20770-1-2-20210518.pdf`

        Returns:
            Tensor: The encoded image features.
        """
        feats = []
        for k, feat in cache.items():
            safe_key = k.replace(".", "_")
            x = self.spatial_embeddings[safe_key](feat)
            x = self.post_encoders[safe_key](x)
            if detach:
                x = x.detach()
            feats.append(x)
        return torch.cat(feats, dim=-1)

    @property
    def output_dim(self) -> int:
        return self._out_dim


class MLP(nn.Module):
    """Multi-layer perceptron builder.

    Dynamically constructs a sequence of layers based on `hidden_dims`:
      1) Linear (in_dim -> out_dim)
      2) Optional Dropout if `dropout_rate` > 0 and (not final layer or `activate_final`)
      3) LayerNorm on the output features
      4) Activation (standard for intermediate layers, `final_activation` for last layer if `activate_final`)

    Arguments:
        input_dim (int): Size of input feature dimension.
        hidden_dims (list[int]): Sizes for each hidden layer.
        activations (Callable or str): Activation to apply between layers.
        activate_final (bool): Whether to apply activation at the final layer.
        dropout_rate (Optional[float]): Dropout probability applied before normalization and activation.
        final_activation (Optional[Callable or str]): Activation for the final layer when `activate_final` is True.

    For each layer, `in_dim` is updated to the previous `out_dim`. All constructed modules are
    stored in `self.net` as an `nn.Sequential` container.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        total = len(hidden_dims)

        for idx, out_dim in enumerate(hidden_dims):
            # 1) linear transform
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = idx == total - 1
            # 2-4) optionally add dropout, normalization, and activation
            if not is_last or activate_final:
                if dropout_rate and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(out_dim))
                act_cls = final_activation if is_last and final_activation else activations
                act = act_cls if isinstance(act_cls, nn.Module) else getattr(nn, act_cls)()
                layers.append(act)

            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantileCriticHead(nn.Module):
    """Critic head outputting N quantiles (TQC / QR-DQN style)."""

    def __init__(
        self,
        input_dim: int,
        n_quantiles: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.n_quantiles = int(n_quantiles)
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=self.n_quantiles)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))




class CriticEnsemble(nn.Module):
    """
    CriticEnsemble wraps multiple QuantileCriticHead modules into an ensemble.

    Args:
        encoder (SACObservationEncoder): encoder for observations.
        ensemble (List[QuantileCriticHead]): list of critic heads.
        output_normalization (nn.Module): normalization layer for actions.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size, n_quantiles).
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[QuantileCriticHead],
        output_normalization: nn.Module,
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.output_normalization = output_normalization
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items()}
        # NOTE: We normalize actions it helps for sample efficiency
        actions: dict[str, torch.tensor] = {"action": actions}
        # NOTE: Normalization layer took dict in input and outputs a dict that why

        # print("actions before normalization:", actions)
        actions = self.output_normalization(actions)["action"]
        # print("actions after normalization:", actions)
        
        actions = actions.to(device)

        obs_enc = self.encoder(observations, cache=observation_features)

        inputs = torch.cat([obs_enc, actions], dim=-1)

        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        return torch.stack(q_values, dim=0)

class QuantileCriticEnsemble(nn.Module):
    """Ensemble of quantile critics. Forward returns (num_critics, batch, n_quantiles)."""

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[QuantileCriticHead],
        output_normalization: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_normalization = output_normalization
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}

        actions_dict: dict[str, torch.Tensor] = {"action": actions}
        actions_dict = self.output_normalization(actions_dict)
        actions_normed = actions_dict["action"].to(device)

        obs_enc = self.encoder(observations, cache=observation_features)
        inputs = torch.cat([obs_enc, actions_normed], dim=-1)

        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))
        return torch.stack(q_values, dim=0)

class DiscreteCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 3,
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )

        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=self.output_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self, observations: torch.Tensor, observation_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}
        obs_enc = self.encoder(observations, cache=observation_features)
        return self.output_layer(self.net(obs_enc))


class Policy(nn.Module):
    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        std_min: float = -5,
        std_max: float = 2,
        fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        self.fixed_std = fixed_std
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        # Mean layer
        self.mean_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

        # Standard deviation layer or parameter
        if fixed_std is None:
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                orthogonal_init()(self.std_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            log_std = torch.clamp(log_std, self.std_min, self.std_max)  # Match JAX default clip
            std = torch.exp(log_std)  # Match JAX "exp"
        else:
            std = self.fixed_std.expand_as(means)

        # Build transformed distribution
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        # Sample actions (reparameterized)
        actions = dist.rsample()

        # Compute log_probs
        log_probs = dist.log_prob(actions)

        return actions, log_probs, means

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations


class DefaultImageEncoder(nn.Module):
    def __init__(self, config: TQCConfig):
        super().__init__()
        image_key = next(key for key in config.input_features if is_image_feature(key))
        self.image_enc_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=config.input_features[image_key].shape[0],
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=7,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=5,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.image_enc_layers(x)
        return x


def freeze_image_encoder(image_encoder: nn.Module):
    """Freeze all parameters in the encoder"""
    for param in image_encoder.parameters():
        param.requires_grad = False


class PretrainedImageEncoder(nn.Module):
    def __init__(self, config: TQCConfig):
        super().__init__()

        self.image_enc_layers, self.image_enc_out_shape = self._load_pretrained_vision_encoder(config)

    def _load_pretrained_vision_encoder(self, config: TQCConfig):
        """Set up CNN encoder"""
        from transformers import AutoModel

        self.image_enc_layers = AutoModel.from_pretrained(
            config.vision_encoder_name,
            trust_remote_code=True,
            revision="c587b31f2f79e653249ed4af78f0ba7dff72122c",
        )

        if hasattr(self.image_enc_layers.config, "hidden_sizes"):
            self.image_enc_out_shape = self.image_enc_layers.config.hidden_sizes[-1]  # Last channel dimension
        elif hasattr(self.image_enc_layers, "fc"):
            self.image_enc_out_shape = self.image_enc_layers.fc.in_features
        else:
            raise ValueError("Unsupported vision encoder architecture, make sure you are using a CNN")
        return self.image_enc_layers, self.image_enc_out_shape

    def forward(self, x):
        enc_feat = self.image_enc_layers(x).last_hidden_state
        return enc_feat


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings

        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding

        Args:
            features: Input tensor of shape [B, C, H, W] where B is batch size,
                     C is number of channels, H is height, and W is width
        Returns:
            Output tensor of shape [B, C*F] where F is the number of features
        """

        features_expanded = features.unsqueeze(-1)  # [B, C, H, W, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, C, H, W, F]

        # Element-wise multiplication and spatial reduction
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # Sum over H,W dimensions

        # Reshape to combine channel and feature dimensions
        output = output.view(output.size(0), -1)  # [B, C*F]

        return output


class RescaleFromTanh(Transform):
    def __init__(self, low: float = -1, high: float = 1):
        super().__init__()

        self.low = low

        self.high = high

    def _call(self, x):
        # Rescale from (-1, 1) to (low, high)

        return 0.5 * (x + 1.0) * (self.high - self.low) + self.low

    def _inverse(self, y):
        # Rescale from (low, high) back to (-1, 1)

        return 2.0 * (y - self.low) / (self.high - self.low) - 1.0

    def log_abs_det_jacobian(self, x, y):
        # log|d(rescale)/dx| = sum(log(0.5 * (high - low)))

        scale = 0.5 * (self.high - self.low)

        return torch.sum(torch.log(scale), dim=-1)


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(self, loc, scale_diag, low=None, high=None):
        base_dist = MultivariateNormal(loc, torch.diag_embed(scale_diag))

        transforms = [TanhTransform(cache_size=1)]

        if low is not None and high is not None:
            low = torch.as_tensor(low)

            high = torch.as_tensor(high)

            transforms.insert(0, RescaleFromTanh(low, high))

        super().__init__(base_dist, transforms)

    def mode(self):
        # Mode is mean of base distribution, passed through transforms

        x = self.base_dist.mean

        for transform in self.transforms:
            x = transform(x)

        return x

    def stddev(self):
        std = self.base_dist.stddev

        x = std

        for transform in self.transforms:
            x = transform(x)

        return x


def _convert_normalization_params_to_tensor(normalization_params: dict) -> dict:
    converted_params = {}
    for outer_key, inner_dict in normalization_params.items():
        converted_params[outer_key] = {}
        for key, value in inner_dict.items():
            converted_params[outer_key][key] = torch.tensor(value)
            if "image" in outer_key:
                converted_params[outer_key][key] = converted_params[outer_key][key].view(3, 1, 1)

    return converted_params
