#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

__all__ = ["ApproxContainer", "SACPen"]

from typing import Any, Optional

from gops.algorithm.sac import ApproxContainer, SAC
from gops.utils.gops_typing import DataDict


class SACPen(SAC):
    def __init__(
        self,
        index: int = 0,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 1.,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        penalty: float = 1.,
        **kwargs: Any,
    ):
        super().__init__(
            index=index,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            auto_alpha=auto_alpha,
            target_entropy=target_entropy,
            **kwargs,
        )
        self.penalty = penalty

    @property
    def adjustable_parameters(self):
        return super().adjustable_parameters + ("penalty",)

    def _compute_loss_q(self, data: DataDict):
        data["rew"] = data["rew"] - self.penalty * data["next_constraint"]
        return super()._compute_loss_q(data)
