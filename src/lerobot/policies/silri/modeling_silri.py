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

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, Transform, TransformedDistribution
import os
import cv2
import copy
from lerobot.policies.normalize import NormalizeBuffer
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.policies.sac.modeling_sac import SACObservationEncoder, CriticHead, CriticEnsemble, DiscreteCritic, MLP
from lerobot.policies.silri.configuration_silri import SiLRIConfig


class SiLRIPolicy(
    PreTrainedPolicy,
):
    config_class = SiLRIConfig
    name = "silri"

    def __init__(
        self,
        config: SiLRIConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        dataset_stats=self.config.dataset_stats

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features["action"].shape[0]
        self.continuous_action_dim = continuous_action_dim
        
        self._init_normalization(dataset_stats)
        # 初始化观测编码器（Actor与Critic可共享或独立）
        self._init_encoders()  
        # 初始化Critic网络（连续动作+可选离散动作）
        self._init_critics(continuous_action_dim)

        self._init_expert_network(continuous_action_dim)
        self._init_lagrange_network()

        # 初始化Actor网络（输出连续动作分布）
        self._init_actor(continuous_action_dim)

    def get_optim_params(self) -> dict:
        """获取各模块的可优化参数，用于构建优化器"""
        optim_params = {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                # 若共享编码器，Actor不优化编码器参数（避免梯度冲突）
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic": self.critic_ensemble.parameters(),
            "expert": self.expert_network.parameters(),
            "lagrange": self.lagrange_net.parameters(),
        }
        return optim_params

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
        Select action for inference/evaluation
        Args:
            batch: Dictionary of observations (containing images (right, wrist) and state)
        Returns:
            Final action tensor (continuous action + optional discrete action concatenation)
        """
        observations_features = None
        
        # 若共享编码器且含图像，缓存图像特征（避免重复编码，提升速度）
        if self.shared_encoder and self.actor.encoder.has_images:
            # Cache and normalize image features

            observations_features = self.actor.encoder.get_cached_image_features(batch, normalize=True)
        # actor网络生成当前观测对应的基础动作
        actions, *_ = self.actor(batch, observations_features)

        epsilon = 1e-6
        actions = torch.clamp(actions, -1+epsilon, 1-epsilon)

        # 若有离散动作，离散Critic输出各动作价值，选价值最大的动作
        # todo11
        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_actor(batch, observations_features)
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
            Tensor of Q-values from all critics
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values
    

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "discrete_critic", "discrete_actor", "expert", "lagrange", "actor_bc"] = "critic",
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
        # weights: Tensor = batch["weights"]

        if model == "critic":
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
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

            return {
                "loss_critic": loss_critic,
            }
        
        if model == "expert":
            is_intervention = batch.get("is_intervention")
            loss_expert, allow_distance = self.compute_loss_expert(
                observations=observations,
                actions=actions,
                observation_features=observation_features,
                is_intervention=is_intervention,
            )
            return {"loss_expert": loss_expert, "allow_d": allow_distance}
        
        if model == "actor_bc":
            is_intervention = batch.get("is_intervention")
            loss_actor_dict = self.compute_loss_actor_bc(
                observations=observations,
                actions=actions,
                observation_features=observation_features,
                is_intervention=is_intervention,
            )

            if self.config.num_discrete_actions is not None:
                loss_discrete_actor = self.compute_loss_discrete_actor(
                    observations=observations,
                    observation_features=observation_features,
                    is_intervention=is_intervention,
                    old_actions=actions,
                )
                loss_actor_dict["loss_actor_bc"] = loss_actor_dict["loss_actor_bc"] + loss_discrete_actor["loss_actor"]
            return loss_actor_dict
            
        if model == "lagrange":
            is_intervention = batch.get("is_intervention")
            loss_lagrange, mean_d, allow_d, cost_dev = self.compute_loss_lagrange(
                observations=observations,
                observation_features=observation_features,
            )
            return {
                "loss_lagrange": loss_lagrange,
                "mean_d": mean_d,
                "allow_d": allow_d, 
                "cost_dev": cost_dev
            }

        if model == "actor":
            is_intervention = batch.get("is_intervention")
            loss_actor_dict = self.compute_loss_actor(
                    observations=observations,
                    observation_features=observation_features,
                )

            if self.config.num_discrete_actions is not None:
                loss_discrete_actor = self.compute_loss_discrete_actor(
                    observations=observations,
                    observation_features=observation_features,
                    is_intervention=is_intervention,
                    old_actions=actions,
                )
                loss_actor_dict["loss_continuous_actor"] = loss_actor_dict["loss_actor"].item()
                loss_actor_dict["loss_actor"] = loss_actor_dict["loss_actor"] + loss_discrete_actor["loss_actor"]
                loss_actor_dict["loss_discrete_actor"] = loss_discrete_actor["loss_actor"].item()
            else:
                loss_actor_dict["loss_actor"] = loss_actor_dict["loss_actor"]
            return loss_actor_dict

        raise ValueError(f"Unknown model type: {model}")


    def update_target_networks(self):

        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )

        for target_param, param in zip(
            self.actor_target.parameters(),
            self.actor.parameters(),
        ):
            target_param.data.copy_(
                param.data * self.config.actor_target_update_weight
                + target_param.data * (1.0 - self.config.actor_target_update_weight)
            )

    def compute_loss_expert(self, observations, actions, observation_features: Tensor | None = None, is_intervention: Tensor | None = None) -> Tensor:
        log_probs = self.expert_network.get_log_probs(observations, actions[:, 0:self.continuous_action_dim], observation_features)
        
        loss_expert = - log_probs * is_intervention
        loss_expert = loss_expert.sum() / is_intervention.sum().item()

        _, expert_means, expert_std = self.expert_network.get_dist(observations, observation_features)
        allow_distance = expert_std.sum(dim=-1).mean().item()
        return loss_expert, allow_distance  

    def compute_loss_actor_bc(self, observations, actions, observation_features: Tensor | None = None, is_intervention: Tensor | None = None) -> Tensor:
        log_probs = self.actor.get_log_probs(observations, actions[:, 0:self.continuous_action_dim], observation_features)
        loss_actor_bc = - log_probs * is_intervention
        loss_actor_bc = loss_actor_bc.sum() / is_intervention.sum().item()
        return {
            "loss_actor_bc":loss_actor_bc
            }


    def compute_loss_lagrange(self, observations, observation_features: Tensor | None = None) -> Tensor:
        with torch.no_grad():
          
            _, actions_expert, expert_std = self.expert_network.get_dist(observations, observation_features)
            allow_distance = expert_std.sum(dim=-1)
            _, _, actions_model = self.actor(observations, observation_features)
            
            mean_distance = (actions_expert - actions_model).pow(2).sum(-1).sqrt()
            cost_dev = mean_distance - allow_distance


            cost_dev = cost_dev - 0.2

        lagrange_multiplier = self.lagrange_net(observations, observation_features=observation_features)
        lagrange_multiplier = lagrange_multiplier.squeeze(-1)

        loss_lagrange = - lagrange_multiplier * cost_dev
        loss_lagrange = loss_lagrange.mean()

        return loss_lagrange, mean_distance.mean().item(), allow_distance.mean().item(), cost_dev.mean().item()

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
            next_action_preds, *_ = self.actor_target(next_observations, next_observation_features)
			
            q_targets = self.critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
                observation_features=next_observation_features,
            )

            # subsample critics to prevent overfitting if use high UTD (update to date)
            # TODO: Get indices before forward pass to avoid unnecessary computation
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            # critics subsample size
            min_q, _ = q_targets.min(dim=0)  # Get values from min operation
            # Compute the final target Q value (TD Target): r + γ*(1-done)*min_q
            td_target = rewards + (1 - done) * self.config.discount * min_q


        actions = actions[:, :self.continuous_action_dim]
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()

        return critics_loss
    

    def compute_loss_discrete_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
        is_intervention: Tensor | None = None,
        old_actions: Tensor | None = None
    ):
        actions_discrete: Tensor = old_actions[:, self.continuous_action_dim:].clone()
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()
        actions_discrete = actions_discrete.squeeze(-1)
        actions_pi = self.discrete_actor(observations, observation_features)

        mask = is_intervention == 1
        actions_pi = actions_pi[mask]
        actions_discrete = actions_discrete[mask]
        discrete_loss = F.cross_entropy(actions_pi, actions_discrete, reduction="none")

        
        discrete_loss = discrete_loss.mean()
        return {
            "loss_actor": discrete_loss
        }
         
 
    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None
    ) -> Tensor:    

        with torch.no_grad():
            lagrange_multiplier = self.lagrange_net(observations, observation_features=observation_features)
            lagrange_multiplier = lagrange_multiplier.squeeze(-1) 
            _, _, expert_actions = self.expert_network(observations, observation_features) 

        _, _, model_actions = self.actor(observations, observation_features)
        
        combine_BC = (expert_actions - model_actions).pow(2).sum(dim=-1).sqrt()
    

        q_preds = self.critic_forward(
            observations=observations,
            actions=model_actions,
            use_target=False,
            observation_features=observation_features,
        )

        min_q_preds = - q_preds.min(dim=0)[0]

        actor_loss  = (min_q_preds + combine_BC * lagrange_multiplier) / (1 + lagrange_multiplier)
        actor_loss = actor_loss.mean()

        min_q_preds = min_q_preds.mean().detach()
        bc_loss = combine_BC.mean().detach()
        
        lagrange_multiplier_value = lagrange_multiplier.mean().item()
        
        return {
            "loss_actor": actor_loss,
            "bc_loss": bc_loss,
            "min_q_preds": min_q_preds,
            'lagrange_multiplier_value': lagrange_multiplier_value,
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
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config, self.normalize_inputs)
        self.encoder_actor = (
            self.encoder_critic
            if self.shared_encoder
            else SACObservationEncoder(self.config, self.normalize_inputs)
        )

    def _init_critics(self, continuous_action_dim):
        """Build critic ensemble, targets, and optional discrete critic."""
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=heads, output_normalization=self.normalize_targets
        )
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=target_heads, output_normalization=self.normalize_targets
        )
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)



    def _init_lagrange_network(self, ):
        """Build lagrange ensemble, targets, and optional discrete critic."""
        lagrange_network_kwargs = copy.deepcopy(self.config.critic_network_kwargs)
        lagrange_network_kwargs.init_final = 0.0
        lagrange_network_kwargs.final_activation = nn.Identity()
        heads = CriticHead(
                input_dim=self.encoder_critic.output_dim,
                **asdict(lagrange_network_kwargs),
            )
        lagrange_encoder = SACObservationEncoder(self.config, self.normalize_inputs)
   
        self.lagrange_net = ValueEnsemble(
            encoder=lagrange_encoder, ensemble=heads, output_normalization=nn.Softplus()
        )

    
    # todo3: initialize discrete_actor
    def _init_discrete_actor(self):
        """Build discrete discrete critic ensemble and target networks."""
        self.discrete_actor = DiscreteCritic(
            encoder=self.encoder_actor,
            input_dim=self.encoder_actor.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_actor_network_kwargs),
        )

    def _init_actor(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            fixed_std=torch.tensor([5e-2]).to("cuda:0"),
            **asdict(self.config.policy_kwargs),
        )
        if self.config.num_discrete_actions is not None:
            self._init_discrete_actor()
        
        self.actor_target = Policy(
            encoder=self.encoder_actor,
            network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            fixed_std=torch.tensor([5e-2]).to("cuda:0"),
            **asdict(self.config.policy_kwargs),
        )
        self.actor_target.load_state_dict(self.actor.state_dict())




    def _init_expert_network(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        self.encoder_expert = SACObservationEncoder(self.config, self.normalize_inputs)
        self.expert_network = Policy(
            encoder=self.encoder_expert,
            network=MLP(input_dim=self.encoder_expert.output_dim, **asdict(self.config.actor_network_kwargs)),
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            fixed_std=None,
            **asdict(self.config.policy_kwargs),
        )

        

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
        model_name: str = "MultivariateNormalDiag",
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
        
        self.model_name = model_name
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
        n=1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Better do not detach the encoder to ensure enough parameters to fit data
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        # Outputs and mean calculation
        outputs = self.network(obs_enc)

        means = self.mean_layer(outputs)

        """
        means through tanh squashing
        """
        means = torch.tanh(means)

        # Compute standard deviations
        # Standard deviation calculation
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
    
            std = torch.exp(log_std)  # Match JAX "exp"
        
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
     
        else:
            std = self.fixed_std.expand_as(means)
        

        # Build transformed distribution
        # Build a multivariate normal distribution with a diagonal covariance matrix
        dist = MultivariateNormalDiag(loc=means, scale_diag=std)

        # Sample actions (reparameterized)
        # Action sampling
        if n == 1:
            actions = dist.rsample()
            """
            Clip actions
            """
            log_probs = dist.log_prob(actions) # torch.Size([batch_size, action_dim])
        else:
            """
            Sample multiple samples
            """
            actions = dist.rsample(sample_shape=(n,)) # torch.Size([n, batch_size, action_dim])
            log_probs = torch.stack([dist.log_prob(actions[i]) for i in range(n)])
            log_probs = dist.log_prob(actions)
            
        return actions, log_probs, means

    def get_dist(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ):
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        # Process through the encoder (e.g., neural network) to extract useful features
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
      
        # Get network outputs
        # Outputs and mean calculation
        outputs = self.network(obs_enc)
      
        means = self.mean_layer(outputs)

        """
        means through tanh squashing
        """
        means = torch.tanh(means)

        # Compute standard deviations
        # Standard deviation calculation
    
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)  # Match JAX "exp"
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
        else:
            std = self.fixed_std.expand_as(means)

        """
        Use multivariate normal distribution
        """
        dist = MultivariateNormalDiag(loc=means, scale_diag=std)

        return dist, means, std

    """
    Compute log probabilities for given states and actions
    """
    def get_log_probs(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        # Process through the encoder (e.g., neural network) to extract useful features
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
      
        # Get network outputs
        # Outputs and mean calculation
        outputs = self.network(obs_enc)
      
        means = self.mean_layer(outputs)

        """
        means through tanh squashing
        """
        means = torch.tanh(means)

        # Compute standard deviations
        # Standard deviation calculation
    
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)  # Match JAX "exp"
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
        else:
            std = self.fixed_std.expand_as(means)

        # Build transformed distribution
        # Build a multivariate normal distribution with a diagonal covariance matrix
        # dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
        """
        Use multivariate normal distribution
        """
        dist = MultivariateNormalDiag(loc=means, scale_diag=std)


        if actions.dim() == 2:  # Single action: [batch_size, action_dim]
            log_probs = dist.log_prob(actions)
        elif actions.dim() == 3:  # Multiple actions: [n, batch_size, action_dim]
            # Compute log probabilities for each sample
            n = actions.shape[0]
            log_probs = torch.stack([dist.log_prob(actions[i]) for i in range(n)])
        else:
            raise ValueError(f"Unexpected actions dimension: {actions.dim()}")
       

        return log_probs
    
    def print_params(self):
        print('---------------- print params ----------------')
        for n, p in self.named_parameters():
            print(n, p.mean()) 

    def entropy(self, observations: torch.Tensor, observation_features: torch.Tensor | None = None):
        

        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
      
        # Get network outputs
        # Outputs and mean calculation
        outputs = self.network(obs_enc)
      
        means = self.mean_layer(outputs)

        """
        means through tanh squashing
        """
        means = torch.tanh(means)

        # Compute standard deviations
        # Standard deviation calculation    
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)  # Match JAX "exp"
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
        else:
            std = self.fixed_std.expand_as(means)
        

        # Build transformed distribution
        # Build a multivariate normal distribution with a diagonal covariance matrix
        # dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        """
        Use multivariate normal distribution
        """

        dist = MultivariateNormalDiag(loc=means, scale_diag=std)

        entropy = dist.entropy()


        return entropy

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations




def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)




def _convert_normalization_params_to_tensor(normalization_params: dict) -> dict:
    converted_params = {}
    for outer_key, inner_dict in normalization_params.items():
        converted_params[outer_key] = {}
        for key, value in inner_dict.items():
            converted_params[outer_key][key] = torch.tensor(value)
            if "image" in outer_key:
                converted_params[outer_key][key] = converted_params[outer_key][key].view(3, 1, 1)

    return converted_params



class MultivariateNormalDiag(MultivariateNormal):
    def __init__(self, loc, scale_diag):
        # Create diagonal covariance matrix from scale_diag
        covariance_matrix = torch.diag_embed(scale_diag)
        # Initialize MultivariateNormal with loc and covariance_matrix
        super().__init__(loc, covariance_matrix)
        

    def mode(self):
        return self.mean

    @property
    def stddev(self):
        # Access parent class stddev property via MultivariateNormal
        # stddev is the square root of the diagonal of the covariance matrix
        return torch.sqrt(torch.diagonal(self.covariance_matrix, dim1=-2, dim2=-1))

    def entropy(self):
        # Use parent class entropy method
        return super().entropy()




class ValueEnsemble(nn.Module):
    """
    ValueEnsemble wraps multiple CriticHead modules into an ensemble.

    Args:
        encoder (SACObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        output_normalization (nn.Module): normalization layer for actions.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing V-values.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: CriticHead,
        output_normalization: nn.Module,
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.output_normalization = output_normalization
        self.critics = ensemble


    def forward(
        self,
        observations: dict[str, torch.Tensor],
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items()}

        obs_enc = self.encoder(observations, cache=observation_features)

        inputs = obs_enc

        q_values = self.critics(inputs)
        # Loop through critics and collect outputs
        # q_values = []
        # for critic in self.critics:
        #     q_values.append(critic(inputs))

        # # Stack outputs to match expected shape [num_critics, batch_size]
        # q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        q_values = self.output_normalization(q_values)
        return q_values


