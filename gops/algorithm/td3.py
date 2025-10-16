#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Twin Delayed Deep Deterministic policy gradient (TD3) algorithm
#  Reference: Fujimoto S, Hoof H, Meger D (2018) 
#             Addressing function approximation error in actor-critic methods. 
#             ICML, Stockholm, Sweden.
#  Update: 2021-03-05, Wenxuan Wang: create TD3 algorithm

__all__ = ["ApproxContainer", "TD3"]

import time
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.gops_typing import DataDict
from gops.utils.math_utils import incremental_update
from gops.utils.tensorboard_setup import tb_tags


class ApproxContainer(ApprBase):
    def __init__(
        self,
        value_learning_rate: float,
        policy_learning_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # create value network
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args)
        self.q2: nn.Module = create_apprfunc(**q_args)

        # create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        self.act_low_lim = torch.from_numpy(policy_args["act_low_lim"]).float()
        self.act_high_lim = torch.from_numpy(policy_args["act_high_lim"]).float()

        #  create target networks
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self.policy_target = deepcopy(self.policy)

        # set target network gradients
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

        # set optimizers
        self.q1_optimizer = Adam(self.q1.parameters(), lr=value_learning_rate)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=value_learning_rate)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_learning_rate)

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class TD3(AlgorithmBase):
    """
    Twin Delayed Deep Deterministic policy gradient (TD3) algorithm

    Paper: https://arxiv.org/pdf/1802.09477.pdf

    Args:
        float   target_noise        : action noise for target pi network. Default to 0.2
        float   noise_clip          : range [-noise_clip, noise_clip] for target_noise. Default to 0.5
        int     index               : for calculating offset of random seed for subprocess. Default to 0.
    """

    def __init__(
        self,
        index: int = 0,
        gamma: float = 0.99,
        tau: float = 0.005,
        delay_update: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        **kwargs,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = gamma
        self.tau = tau
        self.delay_update = delay_update
        self.target_noise = target_noise
        self.noise_clip = noise_clip

    @property
    def adjustable_parameters(self):
        return ("gamma", "tau", "delay_update", "target_noise", "noise_clip")

    @torch.compile
    def local_update(self, data: DataDict, iteration: int) -> dict:
        tb_info = self._compute_gradient(data, iteration)
        self._update(iteration)
        return tb_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        extra_info = self._compute_gradient(data, iteration)

        update_info = {
            "q1_grad": [p._grad for p in self.networks.q1.parameters()],
            "q2_grad": [p._grad for p in self.networks.q2.parameters()],
            "policy_grad": [p._grad for p in self.networks.policy.parameters()],
            "iteration": iteration,
        }

        return extra_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q1_grad = update_info["q1_grad"]
        q2_grad = update_info["q2_grad"]
        policy_grad = update_info["policy_grad"]

        for p, grad in zip(self.networks.q1.parameters(), q1_grad):
            p._grad = grad
        for p, grad in zip(self.networks.q2.parameters(), q2_grad):
            p._grad = grad
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad

        self._update(iteration)

    def _compute_gradient(self, data: dict, iteration):
        start_time = time.time()

        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        loss_q, q1, q2 = self._compute_loss_q(o, a, r, o2, d)
        loss_q.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy = self._compute_loss_pi(o)
        loss_policy.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True

        tb_info = {
            tb_tags["loss_critic"]: loss_q.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            "TD3/critic_avg_q1-RL iter": q1.item(),
            "TD3/critic_avg_q2-RL iter": q2.item(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    def _compute_loss_q(self, o, a, r, o2, d):
        q1 = self.networks.q1(o, a)
        q2 = self.networks.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            a2 = self.networks.policy_target(o2)
            # Target policy smoothing
            epsilon = torch.clamp(
                torch.randn_like(a2) * self.target_noise,
                -self.noise_clip, self.noise_clip,
            )
            a2 = torch.clamp(
                a2 + epsilon,
                self.networks.act_low_lim.to(a2.device),
                self.networks.act_high_lim.to(a2.device),
            )
            # Target Q-values
            q1_pi_targ = self.networks.q1_target(o2, a2)
            q2_pi_targ = self.networks.q2_target(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q, q1.detach().mean(), q2.detach().mean()

    def _compute_loss_pi(self, o):
        return -self.networks.q1(o, self.networks.policy(o)).mean()

    def _update(self, iteration):
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()
        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()

        incremental_update(self.networks.q1, self.networks.q1_target, self.tau)
        incremental_update(self.networks.q2, self.networks.q2_target, self.tau)
        incremental_update(self.networks.policy, self.networks.policy_target, self.tau)
