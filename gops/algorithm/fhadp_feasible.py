#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

__all__ = ["FHADPFeasible"]

import time

import torch
from gops.algorithm.fhadp import FHADP
from gops.utils.gops_typing import DataDict
from gops.utils.tensorboard_setup import tb_tags


EPSILON = 1e-6


class FHADPFeasible(FHADP):
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
        self.interior_t = interior_t

    def _compute_gradient(self, data: DataDict):
        start_time = time.time()
        o, d, info = data["obs"], data["done"], data

        sum_rewards = []
        max_constraints = []

        for step in range(self.pre_horizon):
            o = o.detach()
            a = self.networks.policy(o, step + 1)
            o, r, d, info = self.envmodel.forward(o, a, d, info)

            sum_reward = r
            max_constraint = info["constraint"].max(1).values

            orig_o = o
            orig_d = d
            orig_info = info

            for i in range(step + 1, self.pre_horizon):
                orig_a = self.networks.policy(orig_o, i + 1)
                orig_o, orig_r, orig_d, orig_info = self.envmodel.forward(orig_o, orig_a, orig_d, orig_info)
                sum_reward += orig_r
                max_constraint = torch.maximum(max_constraint, orig_info["constraint"].max(1).values)

            sum_rewards.append(sum_reward)
            max_constraints.append(max_constraint)

        sum_rewards = torch.stack(sum_rewards)
        max_constraints = torch.stack(max_constraints)

        feasible = max_constraints < -EPSILON
        feasible_loss = masked_mean(
            -sum_rewards - 1 / self.interior_t * 
            torch.log(-torch.clamp_max(max_constraints, -EPSILON)),
            feasible,
        )
        infeasible_loss = masked_mean(max_constraints, ~feasible)
        loss_policy = feasible_loss + infeasible_loss
        self.networks.policy.zero_grad()
        loss_policy.backward()

        end_time = time.time()
        self.tb_info.update({
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_actor_reward"]: -sum_rewards[0].mean().item(),
            tb_tags["loss_actor_constraint"]: max_constraints[0].mean().item(),
            "Loss/Feasible ratio-RL iter": feasible.float().mean().item(),
            "Loss/Infeasible loss-RL iter": infeasible_loss.item(),
        })
        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / torch.clamp_min(mask.sum(), 1)
