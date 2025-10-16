__all__ = ["ApproxContainer", "FPITD3"]

import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam
from gops.algorithm.td3 import ApproxContainer as TD3ApproxContainer, TD3
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.gops_typing import DataDict
from gops.utils.math_utils import incremental_update
from gops.utils.tensorboard_setup import tb_tags


class ApproxContainer(TD3ApproxContainer):
    def __init__(
        self,
        value_learning_rate: float,
        feasibility_learning_rate: float,
        policy_learning_rate: float,
        **kwargs,
    ):
        super().__init__(
            value_learning_rate=value_learning_rate,
            policy_learning_rate=policy_learning_rate,
            **kwargs,
        )
        # create feasibility networks
        g_args = get_apprfunc_dict("feasibility", **kwargs)
        self.g1: nn.Module = create_apprfunc(**g_args)
        self.g2: nn.Module = create_apprfunc(**g_args)
        self.g1_target: nn.Module = deepcopy(self.g1)
        self.g2_target: nn.Module = deepcopy(self.g2)

        for p in self.g1_target.parameters():
            p.requires_grad = False
        for p in self.g2_target.parameters():
            p.requires_grad = False

        self.g1_optimizer = Adam(self.g1.parameters(), lr=feasibility_learning_rate)
        self.g2_optimizer = Adam(self.g2.parameters(), lr=feasibility_learning_rate)


class FPITD3(TD3):
    def __init__(
        self,
        index: int = 0,
        gamma: float = 0.99,
        tau: float = 0.005,
        delay_update: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        gamma_g: float = 0.99,
        epsilon: float = 0.1,
        penalty: float = 1.,
        **kwargs,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = gamma
        self.tau = tau
        self.delay_update = delay_update
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.gamma_g = gamma_g
        self.epsilon = epsilon
        self.penalty = penalty

    @property
    def adjustable_parameters(self):
        return super().adjustable_parameters + ("gamma_g", "epsilon", "penalty")

    def _compute_gradient(self, data: DataDict, iteration: int):
        start_time = time.time()

        o, a, r, c, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["next_constraint"],
            data["obs2"],
            data["done"],
        )

        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        loss_q, q1, q2 = self._compute_loss_q(o, a, r, o2, d)
        loss_q.backward()

        self.networks.g1_optimizer.zero_grad()
        self.networks.g2_optimizer.zero_grad()
        loss_g, g1, g2 = self._compute_loss_g(o, a, c, o2, d)
        loss_g.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False
        for p in self.networks.g1.parameters():
            p.requires_grad = False
        for p in self.networks.g2.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, fea = self._compute_loss_pi(o)
        loss_policy.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True
        for p in self.networks.g1.parameters():
            p.requires_grad = True
        for p in self.networks.g2.parameters():
            p.requires_grad = True

        tb_info = {
            tb_tags["loss_critic"]: loss_q.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            "FPITD3/critic_avg_q1-RL iter": q1.item(),
            "FPITD3/critic_avg_q2-RL iter": q2.item(),
            "FPITD3/scenery_avg_g1-RL iter": g1.item(),
            "FPITD3/scenery_avg_g2-RL iter": g2.item(),
            "FPITD3/violation-RL iter": data["next_constraint"].mean().item(),
            "FPITD3/feasible-RL iter": fea.item(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    def _compute_loss_g(self, o, a, c, o2, d):
        g1 = self.networks.g1(o, a)
        g2 = self.networks.g2(o, a)
        with torch.no_grad():
            a2 = self.networks.policy_target(o2)
            epsilon = torch.clamp(
                torch.randn_like(a2) * self.target_noise,
                -self.noise_clip, self.noise_clip,
            )
            a2 = torch.clamp(
                a2 + epsilon,
                self.networks.act_low_lim.to(a2.device),
                self.networks.act_high_lim.to(a2.device),
            )
            g1_next = self.networks.g1_target(o2, a2)
            g2_next = self.networks.g2_target(o2, a2)
            g_next = torch.clamp(torch.max(g1_next, g2_next), 0, 1)
            backup = c + (1 - d) * (1 - c) * self.gamma * g_next
        loss_g1 = ((g1 - backup) ** 2).mean()
        loss_g2 = ((g2 - backup) ** 2).mean()
        return loss_g1 + loss_g2, g1.detach().mean(), g2.detach().mean()

    def _compute_loss_pi(self, o):
        a = self.networks.policy(o)
        q = torch.min(self.networks.q1(o, a), self.networks.q2(o, a))
        g = torch.max(self.networks.g1(o, a), self.networks.g2(o, a))
        fea = g <= self.epsilon
        loss = torch.where(fea, -q, self.penalty * g).mean()
        return loss, fea.float().mean()

    def _update(self, iteration):
        super()._update(iteration)

        self.networks.g1_optimizer.step()
        self.networks.g2_optimizer.step()

        incremental_update(self.networks.g1, self.networks.g1_target, self.tau)
        incremental_update(self.networks.g2, self.networks.g2_target, self.tau)
