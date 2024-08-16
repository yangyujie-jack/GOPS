#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

__all__ = ["FHADPDualValue"]

import time
from typing import Tuple

import torch
from torch.optim import Adam
from gops.apprfunc.mlp import FiniteHorizonPolicy, FiniteHorizonStateValue
from gops.algorithm.base import ApprBase
from gops.algorithm.fhadp import ApproxContainer, FHADP
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.gops_typing import DataDict, InfoDict
from gops.utils.tensorboard_setup import tb_tags


EPSILON = 1e-6


class ApproxContainer(ApprBase):
    def __init__(
        self,
        *,
        policy_learning_rate: float,
        value_learning_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        policy_args = get_apprfunc_dict("policy", **kwargs)
        value_args = get_apprfunc_dict("value", **kwargs)

        self.policy: FiniteHorizonPolicy = create_apprfunc(**policy_args)
        self.safe_policy: FiniteHorizonPolicy = create_apprfunc(**policy_args)
        self.value: FiniteHorizonStateValue = create_apprfunc(**value_args)
        self.safe_value: FiniteHorizonStateValue = create_apprfunc(**value_args)

        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=policy_learning_rate
        )
        self.safe_policy_optimizer = Adam(
            self.safe_policy.parameters(), lr=policy_learning_rate
        )
        self.value_optimizer = Adam(
            self.value.parameters(), lr=value_learning_rate
        )
        self.safe_value_optimizer = Adam(
            self.safe_value.parameters(), lr=value_learning_rate
        )
        self.optimizer_dict = {
            "policy": self.policy_optimizer,
            "safe_policy": self.safe_policy_optimizer,
            "value": self.safe_value_optimizer,
            "safe_value": self.safe_value_optimizer,
        }
        self.init_scheduler(**kwargs)

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class FHADPDualValue(FHADP):
    def __init__(
        self,
        *,
        pre_horizon: int,
        gamma: float = 1.0,
        interior_t: float = 1.0,
        index: int = 0,
        **kwargs,
    ):
        super().__init__(
            pre_horizon=pre_horizon,
            gamma=gamma,
            index=index,
            **kwargs,
        )
        self.networks = ApproxContainer(**kwargs)
        self.interior_t = interior_t

    def _local_update(self, data: DataDict, iteration: int) -> InfoDict:
        self._compute_gradient(data)
        self.networks.policy_optimizer.step()
        self.networks.safe_policy_optimizer.step()
        self.networks.value_optimizer.step()
        self.networks.safe_value_optimizer.step()
        return self.tb_info

    def get_remote_update_info(self, data: DataDict, iteration: int) -> Tuple[InfoDict, DataDict]:
        self._compute_gradient(data)
        update_info = {
            "policy_grad": [p._grad for p in self.networks.policy.parameters()],
            "safe_policy_grad": [p._grad for p in self.networks.safe_policy.parameters()],
            "value_grad": [p._grad for p in self.networks.value.parameters()],
            "safe_value_grad": [p._grad for p in self.networks.safe_value.parameters()],
        }
        return self.tb_info, update_info

    def _remote_update(self, update_info: DataDict):
        for p, grad in zip(self.networks.policy.parameters(), update_info["policy_grad"]):
            p.grad = grad
        for p, grad in zip(self.networks.safe_policy.parameters(), update_info["safe_policy_grad"]):
            p.grad = grad
        for p, grad in zip(self.networks.value.parameters(), update_info["value_grad"]):
            p.grad = grad
        for p, grad in zip(self.networks.safe_value.parameters(), update_info["safe_value_grad"]):
            p.grad = grad
        self.networks.policy_optimizer.step()
        self.networks.safe_policy_optimizer.step()
        self.networks.value_optimizer.step()
        self.networks.safe_value_optimizer.step()

    def _compute_gradient(self, data: DataDict):
        start_time = time.time()
        o, d, info = data["obs"], data["done"], data

        # update value
        observations = []
        dones = []
        infos = []
        rewards = []
        with torch.no_grad():
            for i in range(self.pre_horizon):
                observations.append(o)
                dones.append(d)
                infos.append(info)
                a = self.networks.policy(o, i + 1)
                o, r, d, info = self.envmodel.forward(o, a, d, info)
                rewards.append(r)
        observations = torch.stack(observations)
        ts = torch.arange(1, self.pre_horizon + 1).unsqueeze(1).expand(self.pre_horizon, observations.shape[1])
        values = self.networks.value(observations, ts)
        sum_rewards = torch.cumsum(torch.stack(rewards[::-1]), dim=0).flip([0])
        value_loss = ((values - sum_rewards) ** 2).mean()
        self.networks.value.zero_grad()
        value_loss.backward()

        # update safe policy and safe value
        safe_actions = self.networks.safe_policy(observations, ts)
        safe_next_observations = []
        safe_constraints = []
        for i in range(self.pre_horizon):
            o, _, _, info = self.envmodel.forward(observations[i], safe_actions[i], dones[i], infos[i])
            safe_next_observations.append(o)
            safe_constraints.append(info["constraint"].max(1).values)
        safe_next_observations = torch.stack(safe_next_observations[:-1])
        safe_next_values = self.networks.safe_value(safe_next_observations, ts[1:])
        safe_next_values = torch.cat((safe_next_values, safe_constraints[-1].unsqueeze(0)))
        safe_constraints = torch.stack(safe_constraints)
        safe_one_step_max_constraints = torch.maximum(safe_constraints, safe_next_values)

        safe_policy_loss = safe_one_step_max_constraints.mean()
        self.networks.safe_policy.zero_grad()
        safe_policy_loss.backward()

        safe_values = self.networks.safe_value(observations, ts)
        safe_value_loss = ((safe_values - safe_one_step_max_constraints.detach()) ** 2).mean()
        self.networks.safe_value.zero_grad()
        safe_value_loss.backward()

        # update policy
        for p in self.networks.value.parameters():
            p.requires_grad = False
        for p in self.networks.safe_value.parameters():
            p.requires_grad = False

        actions = self.networks.policy(observations, ts)
        next_observations = []
        rewards = []
        constraints = []
        for i in range(self.pre_horizon):
            o, r, _, info = self.envmodel.forward(observations[i], actions[i], dones[i], infos[i])
            next_observations.append(o)
            rewards.append(r)
            constraints.append(info["constraint"].max(1).values)
        next_observations = torch.stack(next_observations[:-1])
        next_values = self.networks.value(next_observations, ts[1:])
        next_values = torch.cat((next_values, torch.zeros_like(next_values[0:1])))
        rewards = torch.stack(rewards)
        one_step_sum_rewards = rewards + next_values
        next_safe_values = self.networks.safe_value(next_observations, ts[1:])
        next_safe_values = torch.cat((next_safe_values, constraints[-1].unsqueeze(0)))
        constraints = torch.stack(constraints)
        one_step_max_constraints = torch.maximum(constraints, next_safe_values)
        feasible = one_step_max_constraints < -EPSILON
        feasible_policy_loss = masked_mean(
            -one_step_sum_rewards - 1 / self.interior_t * 
            torch.log(-torch.clamp_max(one_step_max_constraints, -EPSILON)),
            feasible,
        )
        infeasible_policy_loss = masked_mean(one_step_max_constraints, ~feasible)
        policy_loss = feasible_policy_loss + infeasible_policy_loss
        self.networks.policy.zero_grad()
        policy_loss.backward()

        for p in self.networks.value.parameters():
            p.requires_grad = True
        for p in self.networks.safe_value.parameters():
            p.requires_grad = True

        end_time = time.time()
        self.tb_info.update({
            tb_tags["loss_actor"]: policy_loss.item(),
            tb_tags["loss_actor_reward"]: one_step_sum_rewards[0].mean().item(),
            tb_tags["loss_actor_constraint"]: one_step_max_constraints[0].mean().item(),
            tb_tags["loss_critic"]: value_loss.item(),
            "Loss/Safe actor loss-RL iter": safe_policy_loss.item(),
            "Loss/Safe critic loss-RL iter": safe_value_loss.item(),
            "Loss/Feasible ratio-RL iter": feasible.float().mean().item(),
        })
        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / torch.clamp_min(mask.sum(), 1)
