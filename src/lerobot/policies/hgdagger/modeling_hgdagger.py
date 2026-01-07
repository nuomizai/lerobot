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
from lerobot.policies.normalize import NormalizeBuffer
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.policies.sac.modeling_sac import SACObservationEncoder, CriticHead, CriticEnsemble, DiscreteCritic, MLP
from lerobot.policies.awac.modeling_awac import Policy as TanhPolicy
from lerobot.policies.hgdagger.configuration_hgdagger import HGDaggerConfig
DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


class HGDaggerPolicy(
    PreTrainedPolicy,
):
    config_class = HGDaggerConfig
    name = "hgdagger"

    def __init__(
        self,
        config: HGDaggerConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        dataset_stats=self.config.dataset_stats

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features["action"].shape[0]
        
        self._init_normalization(dataset_stats)
        # 初始化观测编码器（Actor与Critic可共享或独立）
        self._init_encoders()  
        # 初始化Critic网络（连续动作+可选离散动作）
        # self._init_critics(continuous_action_dim)
        # # 初始化Critic网络（连续动作+可选离散动作）
        # self._init_prob_networks()
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
            #  "critic": self.critic_ensemble.parameters(),
            #  "prob": self.prob_ensemble.parameters(),
        }
        # 若有离散动作，添加离散Critic参数
        # if self.config.num_discrete_actions is not None:
        #     optim_params["discrete_critic"] = self.discrete_critic.parameters()
            # todo9:优化discrete_actor参数
            # optim_params["discrete_actor"] = self.discrete_actor.parameters()
        return optim_params

    def reset(self):
        """Reset the policy"""
        pass
    

    # @torch.no_grad()
    # def get_prob(self, observation: dict[str, Tensor], future=False) -> float:
    #     """Get the probability of the action"""
    #     observations_features = None
    #     if self.shared_encoder and self.actor.encoder.has_images:
    #         observations_features = self.actor.encoder.get_cached_image_features(observation, normalize=True)
    #     probs = self.prob_ensemble(observation, observation_features=observations_features)
    #     return probs.squeeze(-1).item()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("SACPolicy does not support action chunking. It returns single actions!")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], policy_noise=None) -> Tensor:
        """Select action for inference/evaluation"""
        """
        推理/评估阶段选择动作
        Args:
            batch: 观测字典（含图像（left、wrist）、状态）
        Returns:
            最终动作张量（连续动作 + 可选离散动作拼接）
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
            # discrete_action_value = self.discrete_critic(batch, observations_features)
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
        model: Literal["actor", "critic", "temperature", "discrete_critic", "discrete_actor"] = "critic",
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


        if model == "discrete_critic" and self.config.num_discrete_actions is not None:
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            is_intervention: Tensor = batch["is_intervention"]
            next_observation_features: Tensor = batch.get("next_observation_feature")
            complementary_info = batch.get("complementary_info")
            loss_discrete_critic_dict = self.compute_loss_discrete_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                is_intervention=is_intervention,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
            )
            return loss_discrete_critic_dict
        if model == "actor":
            is_intervention = batch.get("is_intervention")
            loss_actor_dict = self.compute_loss_actor(
                    observations=observations,
                    observation_features=observation_features,
                    is_intervention=is_intervention,
                    old_actions=actions,
                )

            if self.config.num_discrete_actions is not None:
                loss_discrete_actor = self.compute_loss_discrete_actor(
                    observations=observations,
                    observation_features=observation_features,
                    is_intervention=is_intervention,
                    old_actions=actions,
                )
                loss_actor_dict["loss_actor"] = loss_actor_dict["loss_actor"] + loss_discrete_actor["loss_actor"]
            return loss_actor_dict

        raise ValueError(f"Unknown model type: {model}")


    def update_target_networks(self):
        pass

        # """Update target networks with exponential moving average"""
        # # for target_param, param in zip(
        # #     self.critic_target.parameters(),
        # #     self.critic_ensemble.parameters(),
        # # ):
        #     # target_param.data.copy_(
        #     #     param.data * self.config.critic_target_update_weight
        #     #     + target_param.data * (1.0 - self.config.critic_target_update_weight)
        #     # )

        # if self.config.num_discrete_actions is not None:
        #     for target_param, param in zip(
        #         self.discrete_critic_target.parameters(),  # 离散目标Critic参数
        #         self.discrete_critic.parameters(),  # 当前离散Critic参数
        #     ):
        #         target_param.data.copy_(
        #             # 指数移动平均公式：新目标参数 = 当前参数×α + 旧目标参数×(1-α)
        #             param.data * self.config.critic_target_update_weight
        #             + target_param.data * (1.0 - self.config.critic_target_update_weight)
        #         )

        # for target_param, param in zip(
        #     self.actor_target.parameters(),
        #     self.actor_ensemble.parameters(),
        # ):
        #     target_param.data.copy_(
        #         param.data * self.config.actor_target_update_weight
        #         + target_param.data * (1.0 - self.config.actor_target_update_weight)
        #     )

        
    
    def compute_loss_prob(self,
        target_prob,
        observations,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        # Get predicted Q-values for current observations
        probs = self.prob_ensemble(observations, observation_features=observation_features)
        probs = probs.squeeze(-1)
        prob_loss = F.mse_loss(
                input=probs,
                target=target_prob,
                reduction="none",
            )
        prob_loss = prob_loss.mean()
        return prob_loss

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
            
            # next_action_preds = self.actor(next_observations, next_observation_features)
            next_action_preds = self.actor_target(next_observations, next_observation_features)
            noise = (
				torch.randn_like(next_action_preds) * self.config.policy_noise
			).clamp(-self.config.noise_clip, self.config.noise_clip)
            next_action_preds = next_action_preds + noise
			
            # 目标Critic计算 “下一状态-动作对” 的 Q 值
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
            # 计算最终目标Q值（TD Target）：r + γ*(1-done)*min_q
            td_target = rewards + (1 - done) * self.config.discount * min_q

        # 当前Critic预测 “当前状态 - 动作对” 的 Q 值
        if self.config.num_discrete_actions is not None:
            # NOTE: We only want to keep the continuous action part
            # In the buffer we have the full action space (continuous + discrete)
            # We need to split them before concatenating them in the critic forward
            actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()

        return critics_loss
    
    def compute_loss_discrete_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        is_intervention,
        observation_features=None,
        next_observation_features=None,
        complementary_info=None,
    ):
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        # ============= todo: add bc loss to discrete critic =============
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

        # Get predicted Q-values for current observations
        predicted_discrete_qs = self.discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )


        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)

        # Compute MSE loss between predicted and target Q-values
    
        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)

        expert_q_preds = predicted_discrete_q
        bc_loss = - expert_q_preds * is_intervention
        bc_loss = bc_loss.mean()      
        discrete_critic_loss_total = discrete_critic_loss + bc_loss
        return {
            "loss_discrete_critic": discrete_critic_loss_total,
            "loss_bc": bc_loss,
            "loss_q": discrete_critic_loss,
        }


    # todo1:compute_loss_discrete_actor nll
    def compute_loss_discrete_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
        is_intervention: Tensor | None = None,
        old_actions: Tensor | None = None
    ):
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        # ============= todo: add bc loss to discrete critic =============
        # 提取真实的离散动作部分并转换为整数标签
        actions_discrete: Tensor = old_actions[:, DISCRETE_DIMENSION_INDEX:].clone()
        # 四舍五入为整数离散值
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()
        actions_discrete = actions_discrete.squeeze(-1)  # batch_size,) 1维张量
        # print('actions_discrete:', actions_discrete.shape) (256)
        # 当前预测的夹爪动作
        actions_pi = self.discrete_actor(observations, observation_features)
        # print('actions_pi:', actions_pi.shape) (256, 2)
        discrete_loss = F.nll_loss(actions_pi, actions_discrete, reduction="none")
        discrete_loss = discrete_loss.mean()
        return {
            "loss_actor": discrete_loss
        }
         
 
    """
    hg_dagger属于模仿学习，actor loss即bc loss
    随机性策略不再采用mse loss
    """
    # todo2:去掉夹爪action loss的计算
    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
        is_intervention: Tensor | None = None,
        old_actions: Tensor | None = None,
    ) -> Tensor:
        
        # actions_pi = self.actor(observations, observation_features)
        # bc_loss = F.mse_loss(actions_pi, old_actions[:, 0:6], reduction="none").sum(dim=-1)
        
        log_probs = self.actor.get_log_probs(observations, old_actions[:, 0:6], observation_features)
        # print('log_probs:', log_probs)
        bc_loss = - log_probs

        bc_loss = bc_loss.mean()

        _, _, expert_std = self.actor.get_dist(observations, observation_features)
        allow_distance = expert_std.sum(dim=-1)


     

        # ============ todo: add to_goal_probs to min_q_preds ============
        # actor_loss = - min_q_preds + bc_loss
        actor_loss = bc_loss
        return {
            "loss_actor": actor_loss,
            "bc_loss": bc_loss,
            "min_q_preds": actor_loss.clone().detach(),
            "allow_d": allow_distance.mean().item()
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

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()
            # self._init_discrete_actor()
    
    def _init_prob_networks(self, ):
        """Build critic ensemble, targets, and optional discrete critic."""
        heads = CriticHead(
                input_dim=self.encoder_critic.output_dim,
                **asdict(self.config.critic_network_kwargs),
            )
        self.prob_ensemble = ValueEnsemble(
            encoder=self.encoder_critic, ensemble=heads, output_normalization=nn.Sigmoid()
        )

    # todo3:初始化discrete_actor
    def _init_discrete_actor(self):
        """Build discrete discrete critic ensemble and target networks."""
        self.discrete_actor = DiscreteCritic(
            encoder=self.encoder_actor,
            input_dim=self.encoder_actor.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_actor_network_kwargs),
        )

    def _init_discrete_critics(self):
        """Build discrete discrete critic ensemble and target networks."""
        self.discrete_critic = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        self.discrete_critic_target = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
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
            # fixed_std=torch.tensor([5e-3]).to("cuda:0"),
            **asdict(self.config.policy_kwargs),
        )
        if self.config.num_discrete_actions is not None:
            self._init_discrete_actor()
        
        # self.actor_target = Policy(
        #     encoder=self.encoder_actor,
        #     network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
        #     action_dim=continuous_action_dim,
        #     encoder_is_shared=self.shared_encoder,
        #     **asdict(self.config.policy_kwargs),
        # )
        # self.actor_target.load_state_dict(self.actor.state_dict())

        # # ensemble
        # policy_ensemble = [
        #     Policy(
        #         encoder=self.encoder_actor, 
        #         network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
        #         action_dim=continuous_action_dim, 
        #         encoder_is_shared=self.shared_encoder, 
        #         **asdict(self.config.policy_kwargs), 
        #     )
        #     for _ in range(self.config.num_actors)
        # ]
        # self.actor = ActorEnsemble(
        #     encoder=self.encoder_actor, 
        #     ensemble=policy_ensemble
        # )

        # 目标熵
        # self.target_entropy = self.config.target_entropy
        # if self.target_entropy is None:
        #     dim = continuous_action_dim + (1 if self.config.num_discrete_actions is not None else 0)
        #     self.target_entropy = -np.prod(dim) / 2

        

class Policy(nn.Module):
    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        std_min: float = 1e-5,
        std_max: float = 10,
        # std_min: float = -5,
        # std_max: float = 2,
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
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        # 通过编码器（如神经网络）处理，提取有用特征
        # print("observations.observation.images.right:", observations.observation.images.right.shape)
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        # 网络输出与均值计算
        outputs = self.network(obs_enc)

        means = self.mean_layer(outputs)

        """
        means经过tanh放缩
        """
        means = torch.tanh(means)

        # Compute standard deviations
        # 标准差计算

        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
    
            std = torch.exp(log_std)  # Match JAX "exp"
        
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
     
        else:
            std = self.fixed_std.expand_as(means)
        

        # Build transformed distribution
        # 构建一个对角线协方差的多元正态分布
        # dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        """
        采用多元正态分布
        """
        dist = MultivariateNormalDiag(loc=means, scale_diag=std)

        # Sample actions (reparameterized)
        # 动作采样
        if n == 1:
            actions = dist.rsample()
            """
            裁剪动作
            """
            log_probs = dist.log_prob(actions) # torch.Size([batch_size, action_dim])
        else:
            """
            采样多个样本
            """
            actions = dist.rsample(sample_shape=(n,)) # torch.Size([n, batch_size, action_dim])
            # reshape -> [B, action_dim]
            # actions=actions.reshape(-1, actions.shape[-1])
            # print("调整后actions尺寸: ", actions.shape)

            # 为每个样本计算对数概率
            log_probs = torch.stack([dist.log_prob(actions[i]) for i in range(n)])
            log_probs = dist.log_prob(actions) # torch.Size([n*batch_size, action_dim])
            
            # reshape -> [n, batch_size, action_dim]
            # log_probs = log_probs.reshape(n, -1, log_probs.shape[-1])
       
     
    
        return actions, log_probs, means

    def get_dist(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ):
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        # 通过编码器（如神经网络）处理，提取有用特征
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
      
        # Get network outputs
        # 网络输出与均值计算
        outputs = self.network(obs_enc)
      
        means = self.mean_layer(outputs)

        """
        means经过tanh放缩
        """
        means = torch.tanh(means)

        # Compute standard deviations
        # 标准差计算
    
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)  # Match JAX "exp"
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
        else:
            std = self.fixed_std.expand_as(means)

        # Build transformed distribution
        # 构建一个对角线协方差的多元正态分布
        # dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
        """
        采用多元正态分布
        """
        dist = MultivariateNormalDiag(loc=means, scale_diag=std)

        return dist, means, std

    """
    add 2:计算给定状态和动作下的对数概率
    """
    def get_log_probs(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        # 通过编码器（如神经网络）处理，提取有用特征
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
      
        # Get network outputs
        # 网络输出与均值计算
        outputs = self.network(obs_enc)
      
        means = self.mean_layer(outputs)

        """
        means经过tanh放缩
        """
        means = torch.tanh(means)

        # Compute standard deviations
        # 标准差计算
    
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)  # Match JAX "exp"
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
        else:
            std = self.fixed_std.expand_as(means)

        # Build transformed distribution
        # 构建一个对角线协方差的多元正态分布
        # dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
        """
        采用多元正态分布
        """
        dist = MultivariateNormalDiag(loc=means, scale_diag=std)

        # 裁剪动作以保持与forward函数一致
        # epsilon = 1e-6
        # actions = torch.clamp(actions, -1+epsilon, 1-epsilon)
        
        # Compute log_probs
        # 计算采样动作的对数概率
        if actions.dim() == 2:  # 单个动作: [batch_size, action_dim]
            log_probs = dist.log_prob(actions)
        elif actions.dim() == 3:  # 多个动作: [n, batch_size, action_dim]
            # 为每个样本计算对数概率
            n = actions.shape[0]
            log_probs = torch.stack([dist.log_prob(actions[i]) for i in range(n)])
           
            # actions=actions.reshape(-1, actions.shape[-1])
            # print("调整后actions尺寸: ", actions.shape)
            # log_probs = dist.log_prob(actions) # torch.Size([n*batch_size, action_dim])
            # log_probs = log_probs.reshape(n, -1, log_probs.shape[-1])
        
        else:
            raise ValueError(f"Unexpected actions dimension: {actions.dim()}")
       
        if log_probs.abs().mean() > 100:
            print('actions:', actions)
            print('means:', means)
            print('log_probs:', log_probs)
        return log_probs
    
    def print_params(self):
        print('---------------- print params ----------------')
        for n, p in self.named_parameters():
            print(n, p.mean()) 

    def entropy(self, observations: torch.Tensor, observation_features: torch.Tensor | None = None):
        

        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
      
        # Get network outputs
        # 网络输出与均值计算
        outputs = self.network(obs_enc)
      
        means = self.mean_layer(outputs)

        """
        means经过tanh放缩
        """
        means = torch.tanh(means)

        # Compute standard deviations
        # 标准差计算    
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)  # Match JAX "exp"
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
        else:
            std = self.fixed_std.expand_as(means)
        

        # Build transformed distribution
        # 构建一个对角线协方差的多元正态分布
        # dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        """
        采用多元正态分布
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


# class ActorEnsemble(nn.Module):
#     """
#     ActorEnsemble wraps multiple ActorHead modules into an ensemble.

#     Args:
#         encoder (SACObservationEncoder): encoder for observations.
#         ensemble (List[ActorHead]): list of critic heads.
#         output_normalization (nn.Module): normalization layer for actions.
#         init_final (float | None): optional initializer scale for final layers.

#     Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
#     """

#     def __init__(
#         self,
#         encoder: SACObservationEncoder,
#         ensemble: list[Policy],
#         init_final: float | None = None,
#     ):
#         super().__init__()
#         self.encoder = encoder
#         self.init_final = init_final
#         self.policies = nn.ModuleList(ensemble)

#     def forward(
#         self,
#         observations: dict[str, torch.Tensor],
#         observation_features: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         """
#         前向传播：输入观测，输出所有策略头的动作。
        
#         Args:
#             observations: 原始观测字典（如图像、传感器数据）。
#             observation_features: 预计算的观测特征（可选，用于复用特征）。
        
#         Returns:
#             所有策略头的动作，形状为 (num_actors, batch_size, action_dim)
#         """
#         device = get_device_from_parameters(self)
#         # Move each tensor in observations to device
#         observations = {k: v.to(device) for k, v in observations.items()}

#         # 每个策略头独立计算动作
#         actions = []
#         for policy in self.policies:
#             action = policy(observations, observation_features) # action 形状 (batch_size, action_dim)
#             actions.append(action)

#         # 堆叠所有策略的动作，形状为 (num_actors, batch_size, action_dim)
#         actions = torch.stack(actions, dim=0)
#         # print("before mean actions.shape:", actions.shape)
#         actions = torch.mean(actions, dim=0)
#         # print("after mean actions.shape:", actions.shape)
#         return actions

        