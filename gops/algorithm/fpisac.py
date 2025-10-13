__all__ = ["ApproxContainer", "FPISAC"]

import time
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict


class ApproxContainer(ApprBase):
    def __init__(
        self,
        value_learning_rate: float,
        scenery_learning_rate: float,
        policy_learning_rate: float,
        alpha_learning_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # create q networks
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args)
        self.q2: nn.Module = create_apprfunc(**q_args)
        self.q1_target: nn.Module = deepcopy(self.q1)
        self.q2_target: nn.Module = deepcopy(self.q2)

        # create scenery networks
        g_args = get_apprfunc_dict("scenery", **kwargs)
        self.g1: nn.Module = create_apprfunc(**g_args)
        self.g2: nn.Module = create_apprfunc(**g_args)
        self.g1_target: nn.Module = deepcopy(self.g1)
        self.g2_target: nn.Module = deepcopy(self.g2)

        # create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)

        self.q1_target.requires_grad_(False)
        self.q2_target.requires_grad_(False)
        self.g1_target.requires_grad_(False)
        self.g2_target.requires_grad_(False)

        self.log_alpha = nn.Parameter(torch.tensor(0, dtype=torch.float32))

        self.q1_optimizer = Adam(self.q1.parameters(), lr=value_learning_rate)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=value_learning_rate)
        self.g1_optimizer = Adam(self.g1.parameters(), lr=scenery_learning_rate)
        self.g2_optimizer = Adam(self.g2.parameters(), lr=scenery_learning_rate)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_learning_rate)
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_learning_rate)

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class FPISAC(AlgorithmBase):
    def __init__(
        self,
        gamma: float,
        gamma_g: float,
        tau: float,
        target_entropy: Optional[float] = None,
        pf: float = 0.1,
        index: int = 0,
        **kwargs,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = gamma
        self.gamma_g = gamma_g
        self.tau = tau
        if target_entropy is None:
            self.target_entropy = -kwargs["action_dim"]
        else:
            self.target_entropy = target_entropy
        self.pf = pf

    @property
    def adjustable_parameters(self):
        return ("gamma", "gamma_g", "tau", "target_entropy", "pf")

    @torch.compile
    def local_update(self, data: DataDict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    @property
    def alpha(self):
        return self.networks.log_alpha.exp().detach()

    def __compute_gradient(self, data: DataDict, iteration: int):
        start_time = time.time()
        tb_info = {}

        obs, obs2 = data["obs"], data["obs2"]

        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_logp = act_dist.rsample()
        data.update({"new_act": new_act, "new_logp": new_logp})

        with torch.no_grad():
            next_logits = self.networks.policy(obs2)
            next_act, next_logp = self.networks.create_action_distributions(next_logits).sample()
        data.update({"next_act": next_act, "next_logp": next_logp})

        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        self.networks.g1_optimizer.zero_grad()
        self.networks.g2_optimizer.zero_grad()
        self.networks.policy_optimizer.zero_grad()
        self.networks.alpha_optimizer.zero_grad()

        loss_q, q1, q2 = self.__compute_loss_q(data)
        loss_g, g1, g2 = self.__compute_loss_g(data)

        frozen_net = [
            self.networks.q1,
            self.networks.q2,
            self.networks.g1,
            self.networks.g2,
        ]
        for nn in frozen_net:
            nn.requires_grad_(False)

        loss_policy, (fea, entropy) = self.__compute_loss_policy(data)

        for nn in frozen_net:
            nn.requires_grad_(True)

        loss_alpha = -self.networks.log_alpha * (self.target_entropy - entropy)

        loss = loss_q + loss_g + loss_policy + loss_alpha

        loss.backward()

        tb_info.update({
            tb_tags["loss_critic"]: loss_q.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            "Loss/Scenery loss-RL iter": loss_g.item(),
            "SACFPI/critic_avg_q1-RL iter": q1.item(),
            "SACFPI/critic_avg_q2-RL iter": q2.item(),
            "SACFPI/scenery_avg_g1-RL iter": g1.item(),
            "SACFPI/scenery_avg_g2-RL iter": g2.item(),
            "SACFPI/entropy-RL iter": entropy.item(),
            "SACFPI/alpha-RL iter": self.alpha.item(),
            "SACFPI/violation-RL iter": data["next_constraint"].mean().item(),
            "SACFPI/feasible-RL iter": fea.float().mean().item(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        })
        return tb_info

    def __compute_loss_q(self, data: DataDict):
        obs, act, rew, obs2, done, next_act, next_logp = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
            data["next_act"],
            data["next_logp"],
        )

        with torch.no_grad():
            q1_next = self.networks.q1_target(obs2, next_act)
            q2_next = self.networks.q2_target(obs2, next_act)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_logp
            target_q = rew + (1 - done) * self.gamma * q_next

        q1 = self.networks.q1(obs, act)
        q2 = self.networks.q2(obs, act)
        q1_loss = ((q1 - target_q) ** 2).mean()
        q2_loss = ((q2 - target_q) ** 2).mean()

        return q1_loss + q2_loss, q1.mean().detach(), q2.mean().detach()

    def __compute_loss_g(self, data: DataDict):
        obs, act, constraint, obs2, done, next_act = (
            data["obs"],
            data["act"],
            data["next_constraint"],
            data["obs2"],
            data["done"],
            data["next_act"],
        )

        with torch.no_grad():
            g1_next = self.networks.g1_target(obs2, next_act)
            g2_next = self.networks.g2_target(obs2, next_act)
            g_next = torch.clamp(torch.max(g1_next, g2_next), 0, 1)
            target_g = constraint + (1 - done) * (1 - constraint) * self.gamma_g * g_next

        g1 = self.networks.g1(obs, act)
        g2 = self.networks.g2(obs, act)
        g1_loss = ((g1 - target_g) ** 2).mean()
        g2_loss = ((g2 - target_g) ** 2).mean()

        return g1_loss + g2_loss, g1.mean().detach(), g2.mean().detach()

    def __compute_loss_policy(self, data: DataDict):
        obs, new_act, new_logp = (
            data["obs"],
            data["new_act"],
            data["new_logp"],
        )

        g1 = self.networks.g1(obs, new_act)
        g2 = self.networks.g2(obs, new_act)
        g = torch.max(g1, g2)

        q1 = self.networks.q1(obs, new_act)
        q2 = self.networks.q2(obs, new_act)
        q = torch.min(q1, q2)

        fea = g < self.pf
        loss = ((fea * -q + ~fea * g + self.alpha * new_logp)).mean()
        return loss, (fea, -new_logp.mean().detach())

    def __update(self, iteration: int):
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()
        self.networks.g1_optimizer.step()
        self.networks.g2_optimizer.step()
        self.networks.policy_optimizer.step()
        self.networks.alpha_optimizer.step()

        incremental_update(self.networks.q1, self.networks.q1_target, self.tau)
        incremental_update(self.networks.q2, self.networks.q2_target, self.tau)
        incremental_update(self.networks.g1, self.networks.g1_target, self.tau)
        incremental_update(self.networks.g2, self.networks.g2_target, self.tau)


def incremental_update(net_from: nn.Module, net_to: nn.Module, tau: float):
    poltak = 1 - tau
    for (p, p_tar) in zip(net_from.parameters(), net_to.parameters()):
        p_tar.data.mul_(poltak)
        p_tar.data.add_(tau * p.data)
