#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com


__all__ = ["ApproxContainer", "DPPO"]


import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import huber_loss
from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags


class ApproxContainer(ApprBase):
    """Approximate function container for PPO.

    Contains one policy and one state value.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        value_args = get_apprfunc_dict("value", **kwargs)
        self.value: nn.Module = create_apprfunc(**value_args)

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class DPPO(AlgorithmBase):
    """PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347

    :param max_iteration: Maximum iterations for learning rate schedule.
    :param num_repeat: Number of repeats (to reuse sample batch).
    :param num_mini_batch: Number of minibatches to divide sample batch.
    :param mini_batch_size: Minibatch size.
    :param sample_batch_size: Sample batch size.
    """

    def __init__(
        self,
        *,
        max_iteration: int,
        num_repeat: int,
        num_mini_batch: int,
        mini_batch_size: int,
        sample_batch_size: int,
        index: int = 0,
        gamma: float = 0.99,
        clip: float = 0.2,
        loss_coefficient_kl: float = 0.2,
        loss_coefficient_value: float = 1.0,
        loss_coefficient_entropy: float = 0.0,
        schedule_adam: str = "None",
        schedule_clip: str = "None",
        bias: float = 0.1,
        sample_clip: float = 3.,
        huber_delta: float = 50.,
        ratio_min: float = 0.1,
        ratio_max: float = 10.,
        tau_b: float = 0.005,
        **kwargs
    ):
        super().__init__(index, **kwargs)
        self.max_iteration = max_iteration
        self.num_repeat = num_repeat
        self.num_mini_batch = num_mini_batch
        self.mini_batch_size = mini_batch_size
        self.sample_batch_size = sample_batch_size
        self.indices = np.arange(self.sample_batch_size)

        # Parameters for algorithm
        self.gamma = gamma
        self.clip = clip
        self.clip_now = self.clip
        self.loss_coefficient_kl = loss_coefficient_kl
        self.loss_coefficient_value = loss_coefficient_value
        self.loss_coefficient_entropy = loss_coefficient_entropy
        self.schedule_adam = schedule_adam
        self.schedule_clip = schedule_clip
        self.bias = bias
        self.sample_clip = sample_clip
        self.huber_delta = huber_delta
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.tau_b = tau_b

        self.networks = ApproxContainer(**kwargs)
        self.learning_rate = kwargs["learning_rate"]
        self.approximate_optimizer = Adam(
            self.networks.parameters(), lr=self.learning_rate
        )
        self.mean_std = None

    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "clip",
            "loss_coefficient_kl",
            "loss_coefficient_value",
            "loss_coefficient_entropy",
            "schedule_adam",
            "schedule_clip",
        )

    def local_update(self, data: DataDict, iteration: int) -> dict:
        start_time = time.perf_counter()
        data["adv"] = (data["adv"] - data["adv"].mean()) / (data["adv"].std() + 1e-8)
        with torch.no_grad():
            data["logits"] = self.networks.policy(data["obs"])

        for _ in range(self.num_repeat):
            np.random.shuffle(self.indices)

            for n in range(self.num_mini_batch):
                mb_start = self.mini_batch_size * n
                mb_end = self.mini_batch_size * (n + 1)
                mb_indices = self.indices[mb_start:mb_end]
                mb_sample = {k: v[mb_indices] for k, v in data.items()}
                (
                    loss_total,
                    loss_surrogate,
                    loss_value,
                    loss_entropy,
                    approximate_kl,
                    clip_fra,
                ) = self._compute_loss(mb_sample, iteration)
                self.approximate_optimizer.zero_grad()
                loss_total.backward()
                self.approximate_optimizer.step()
                if self.schedule_adam == "linear":
                    decay_rate = 1 - (iteration / self.max_iteration)
                    assert decay_rate >= 0, "the decay_rate is less than 0!"
                    lr_now = self.learning_rate * decay_rate
                    # set learning rate
                    for g in self.approximate_optimizer.param_groups:
                        g["lr"] = lr_now

        end_time = time.perf_counter()

        tb_info = dict()
        tb_info[tb_tags["loss_actor"]] = loss_surrogate.item()
        tb_info[tb_tags["loss_critic"]] = loss_value.item()
        tb_info["PPO/KL_divergence-RL iter"] = approximate_kl.item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000

        return tb_info

    def _compute_loss(self, data: DataDict, iteration: int):
        mb_old_act_dist = self.networks.create_action_distributions(data["logits"])
        mb_new_logits = self.networks.policy(data["obs"])
        mb_new_act_dist = self.networks.create_action_distributions(mb_new_logits)
        mb_new_log_pro = mb_new_act_dist.log_prob(data["act"])

        val_logits = self.networks.value(data["obs"])
        val_mean, val_std = torch.chunk(val_logits, chunks=2, dim=-1)
        val_mean = val_mean.squeeze(-1)
        val_std = val_std.squeeze(-1)
        val_std_detach = val_std.detach()
        ret_bound = torch.clamp(
            data["ret_sample"],
            val_mean - self.sample_clip * val_std,
            val_mean + self.sample_clip * val_std,
        ).detach()

        new_mean_std = val_std.mean().detach()
        if self.mean_std is None:
            self.mean_std = new_mean_std
        else:
            self.mean_std = (1 - self.tau_b) * self.mean_std + self.tau_b * new_mean_std

        # policy loss
        ratio = torch.exp(mb_new_log_pro - data["logp"])
        sur1 = ratio * data["adv"]
        sur2 = ratio.clamp(1 - self.clip_now, 1 + self.clip_now) * data["adv"]
        loss_surrogate = -torch.mean(torch.min(sur1, sur2))

        # value loss
        ratio = (
            self.mean_std.pow(2) / (val_std_detach.pow(2) + self.bias)
        ).clamp(self.ratio_min, self.ratio_max)
        loss_value = torch.mean(
            ratio * (
                huber_loss(val_mean, data["ret"], delta=self.huber_delta, reduction='none') +
                val_std * (
                    val_std_detach.pow(2) -
                    huber_loss(val_mean.detach(), ret_bound, delta=self.huber_delta, reduction='none')
                ) / (val_std_detach + self.bias)
            )
        )
        # loss_value = torch.mean((val_mean - data["ret"]).pow(2))

        # entropy loss
        loss_entropy = torch.mean(mb_new_act_dist.entropy())
        loss_kl = torch.mean(mb_old_act_dist.kl_divergence(mb_new_act_dist))
        clip_fraction = torch.mean(
            torch.gt(torch.abs(ratio - 1.0), self.clip_now).float()
        )

        # total loss
        loss_total = (
            loss_surrogate
            + self.loss_coefficient_kl * loss_kl
            + self.loss_coefficient_value * loss_value
            - self.loss_coefficient_entropy * loss_entropy
        )

        if self.schedule_clip == "linear":
            decay_rate = 1 - (iteration / self.max_iteration)
            assert decay_rate >= 0, "decay_rate is less than 0!"
            self.clip_now = self.clip * decay_rate

        return (
            loss_total,
            loss_surrogate,
            loss_value,
            loss_entropy,
            loss_kl,
            clip_fraction,
        )
