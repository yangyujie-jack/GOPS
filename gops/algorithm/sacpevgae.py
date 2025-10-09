#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Soft Actor-Critic (SAC) algorithm
#  Reference: Haarnoja T, Zhou A, Abbeel P et al (2018) 
#             Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor. 
#             ICML, Stockholm, Sweden.
#  Update: 2021-03-05, Yujie Yang: create SAC algorithm

__all__ = ["ApproxContainer", "SACPEVGAE"]

import time
import math
from copy import deepcopy
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict


class ApproxContainer(ApprBase):
    """Approximate function container for SAC.

    Contains one policy and two action values.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create q network
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q: nn.Module = create_apprfunc(**q_args)

        # create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)

        # create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # create optimizers
        self.q_optimizer = Adam(self.q.parameters(), lr=kwargs["q_learning_rate"])
        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs["policy_learning_rate"])
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class SACPEVGAE(AlgorithmBase):
    """Soft Actor-Critic (SAC) algorithm

    Paper: https://arxiv.org/abs/1801.01290

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param float alpha: initial temperature.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param Optional[float] target_entropy: target entropy for automatic
        temperature adjustment.
    """

    def __init__(
        self,
        index: int = 0,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = math.e,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        n_iter: int = 1,
        **kwargs: Any,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.networks.log_alpha.data.fill_(math.log(alpha))
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -kwargs["action_dim"]
        self.target_entropy = target_entropy
        self.n_iter = n_iter

    @property
    def adjustable_parameters(self):
        return ("gamma", "tau", "alpha", "auto_alpha", "target_entropy")

    def local_update(self, data: DataDict, iteration: int) -> dict:
        for _ in range(self.n_iter):
            tb_info = self._compute_gradient(data, iteration)
            self._update(iteration)
        return tb_info

    def _get_alpha(self, requires_grad: bool = False):
        alpha = self.networks.log_alpha.exp()
        if requires_grad:
            return alpha
        else:
            return alpha.item()

    def _compute_gradient(self, data: DataDict, iteration: int):
        start_time = time.time()

        obs = data["obs"]
        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_logp = act_dist.rsample()
        data.update({"new_act": new_act, "new_logp": new_logp})

        self.networks.q_optimizer.zero_grad()
        loss_q, q = self._compute_loss_q(data)
        loss_q.backward()

        self.networks.q.requires_grad_(False)

        self.networks.policy_optimizer.zero_grad()
        loss_policy, entropy = self._compute_loss_policy(data)
        loss_policy.backward()

        self.networks.q.requires_grad_(True)

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            loss_alpha = self._compute_loss_alpha(data)
            loss_alpha.backward()

        tb_info = {
            tb_tags["loss_critic"]: loss_q.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            "SAC/critic_avg_q-RL iter": q.item(),
            "SAC/entropy-RL iter": entropy.item(),
            "SAC/alpha-RL iter": self._get_alpha(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    def _compute_loss_q(self, data: DataDict):
        obs, act, ret = data["obs"], data["act"], data["ret"]
        q = self.networks.q(obs, act)
        loss = (q - ret).pow(2).mean()
        return loss, q.detach().mean()

    def _compute_loss_policy(self, data: DataDict):
        obs, new_act, new_logp = data["obs"], data["new_act"], data["new_logp"]
        q = self.networks.q(obs, new_act)
        loss_policy = (self._get_alpha() * new_logp - q).mean()
        entropy = -new_logp.detach().mean()
        return loss_policy, entropy

    def _compute_loss_alpha(self, data: DataDict):
        new_logp = data["new_logp"]
        loss_alpha = (
            -self.networks.log_alpha * (new_logp.detach() + self.target_entropy).mean()
        )
        return loss_alpha

    def _update(self, iteration: int):
        self.networks.q_optimizer.step()
        self.networks.policy_optimizer.step()
        if self.auto_alpha:
            self.networks.alpha_optimizer.step()
