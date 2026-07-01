from __future__ import annotations

from functools import partial
from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


class SquashedMlpLstmPolicy(RecurrentActorCriticPolicy):
    """
    Recurrent PPO policy with a tanh-squashed diagonal Gaussian action distribution.

    This replaces the default DiagGaussianDistribution with
    SquashedDiagGaussianDistribution so actions are bounded by tanh and
    log-probabilities are corrected properly.

    Designed for Box action spaces such as [-1, 1]^n.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,  # intentionally ignored
        features_extractor_class: type[nn.Module] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: dict[str, Any] | None = None,
    ):
        if not isinstance(action_space, spaces.Box):
            raise ValueError("SquashedMlpLstmPolicy only supports Box action spaces.")

        if use_sde:
            raise ValueError(
                "SquashedMlpLstmPolicy is intended for standard PPO Gaussian actions, not gSDE."
            )

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        # Keep squash_output=False here.
        # We are NOT using SB3's separate squash/unscale output path.
        # The action distribution itself will do the tanh squashing.
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=False,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=False,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
            lstm_kwargs=lstm_kwargs,
        )

        # Replace the default Gaussian with the squashed Gaussian
        action_dim = get_action_dim(self.action_space)
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

        # Rebuild the action head to match the new distribution
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi,
            log_std_init=self.log_std_init,
        )

        # Reinitialize only the new action head
        if self.ortho_init:
            self.action_net.apply(partial(self.init_weights, gain=0.01))

        # Rebuild optimizer so it includes the new action head/log_std params
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )