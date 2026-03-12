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
        self.continuous_action_dim = continuous_action_dim

        self._init_normalization(dataset_stats)
        # 初始化观测编码器（Actor与Critic可共享或独立）
        self._init_encoders()  
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
        model: Literal["actor"] = "actor",
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
        actions_discrete: Tensor = old_actions[:, self.continuous_action_dim:].clone()
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

        log_probs = self.actor.get_log_probs(observations, old_actions[:, 0:self.continuous_action_dim], observation_features)
        bc_loss = - log_probs

        bc_loss = bc_loss.mean()

        actor_loss = bc_loss
        return {
            "loss_actor": actor_loss,
            "bc_loss": bc_loss,
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
            **asdict(self.config.policy_kwargs),
        )
        if self.config.num_discrete_actions is not None:
            self._init_discrete_actor()


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

