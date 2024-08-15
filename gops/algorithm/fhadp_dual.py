#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

__all__ = ["FHADPDual"]

import time
from copy import deepcopy
from typing import Tuple

import torch
from torch.optim import Adam
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        policy_args = get_apprfunc_dict("policy", **kwargs)

        self.policy = create_apprfunc(**policy_args)
        self.safe_policy = create_apprfunc(**policy_args)
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=policy_learning_rate
        )
        self.safe_policy_optimizer = Adam(
            self.safe_policy.parameters(), lr=policy_learning_rate
        )
        self.optimizer_dict = {
            "policy": self.policy_optimizer,
            "safe_policy": self.safe_policy_optimizer,
        }
        self.init_scheduler(**kwargs)

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class FHADPDual(FHADP):
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
        return self.tb_info

    def get_remote_update_info(self, data: DataDict, iteration: int) -> Tuple[InfoDict, DataDict]:
        self._compute_gradient(data)
        policy_grad = [p._grad for p in self.networks.policy.parameters()]
        safe_policy_grad = [p._grad for p in self.networks.safe_policy.parameters()]
        update_info = dict()
        update_info["grad"] = policy_grad
        update_info["safe_grad"] = safe_policy_grad
        return self.tb_info, update_info

    def _remote_update(self, update_info: DataDict):
        for p, grad in zip(self.networks.policy.parameters(), update_info["grad"]):
            p.grad = grad
        for p, grad in zip(self.networks.safe_policy.parameters(), update_info["safe_grad"]):
            p.grad = grad
        self.networks.policy_optimizer.step()
        self.networks.safe_policy_optimizer.step()

    def _compute_gradient(self, data: DataDict):
        start_time = time.time()

        self.networks.safe_policy.zero_grad()
        loss_safe_policy, loss_safe_info = self._compute_loss_safe_policy(deepcopy(data))
        loss_safe_policy.backward()

        self.networks.policy.zero_grad()
        loss_policy, loss_info = self._compute_loss_policy(deepcopy(data))
        loss_policy.backward()

        end_time = time.time()

        self.tb_info.update(loss_safe_info)
        self.tb_info.update(loss_info)
        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000

    def _compute_loss_safe_policy(self, data: DataDict) -> Tuple[torch.Tensor, InfoDict]:
        o, d = data["obs"], data["done"]
        info = data
        for step in range(self.pre_horizon):
            a = self.networks.safe_policy(o, step + 1)
            o, _, d, info = self.envmodel.forward(o, a, d, info)
            if step == 0:
                max_constraint = info["constraint"].max(1).values
            else:
                max_constraint = torch.maximum(max_constraint, info["constraint"].max(1).values)
        loss_safe_policy = max_constraint.mean()
        loss_safe_info = {
            "Loss/Safe actor loss-RL iter": loss_safe_policy.item()
        }
        return loss_safe_policy, loss_safe_info

    def _compute_loss_policy(self, data: DataDict) -> Tuple[torch.Tensor, InfoDict]:
        o, d, info = data["obs"], data["done"], data

        sum_rewards = []
        max_constraints = []
        safe_max_constraints = []

        for p in self.networks.safe_policy.parameters():
            p.requires_grad = False

        for step in range(self.pre_horizon):
            a = self.networks.policy(o, step + 1)
            o, r, d, info = self.envmodel.forward(o, a, d, info)

            sum_reward = r
            max_constraint = info["constraint"].max(1).values
            safe_max_constraint = info["constraint"].max(1).values

            orig_o = o.clone()
            orig_d = d.clone()
            orig_info = dict_clone(info)
            safe_o = o.clone()
            safe_d = d.clone()
            safe_info = dict_clone(info)

            for p in self.networks.policy.parameters():
                p.requires_grad = False

            for i in range(step + 1, self.pre_horizon):
                orig_a = self.networks.policy(orig_o, i + 1)
                orig_o, orig_r, orig_d, orig_info = self.envmodel.forward(orig_o, orig_a, orig_d, orig_info)
                safe_a = self.networks.safe_policy(safe_o, i + 1)
                safe_o, _, safe_d, safe_info = self.envmodel.forward(safe_o, safe_a, safe_d, safe_info)
                sum_reward += orig_r
                max_constraint = torch.maximum(max_constraint, orig_info["constraint"].max(1).values)
                safe_max_constraint = torch.maximum(safe_max_constraint, safe_info["constraint"].max(1).values)

            for p in self.networks.policy.parameters():
                p.requires_grad = True

            sum_rewards.append(sum_reward)
            max_constraints.append(max_constraint)
            safe_max_constraints.append(safe_max_constraint)

        for p in self.networks.safe_policy.parameters():
            p.requires_grad = True

        sum_rewards = torch.stack(sum_rewards)
        max_constraints = torch.stack(max_constraints)
        safe_max_constraints = torch.stack(safe_max_constraints)

        feasible = safe_max_constraints < -EPSILON
        feasible_loss = masked_mean(
            -sum_rewards - 1 / self.interior_t * 
            torch.log(-torch.clamp_max(safe_max_constraints, -EPSILON)),
            feasible,
        )
        infeasible_loss = masked_mean(max_constraints, ~feasible)
        loss_policy = feasible_loss + infeasible_loss
        loss_info = {
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_actor_reward"]: sum_rewards[0].mean().item(),
            tb_tags["loss_actor_constraint"]: max_constraints[0].mean().item(),
            "Loss/Feasible ratio-RL iter": feasible.float().mean().item(),
        }
        return loss_policy, loss_info


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / torch.clamp_min(mask.sum(), 1)


def dict_clone(d: dict) -> dict:
    d_clone = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d_clone[k] = v.clone()
        else:
            d_clone[k] = deepcopy(v)
    return d_clone
