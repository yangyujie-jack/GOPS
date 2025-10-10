import gym
import numpy as np


class ConstraintInfo(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["constraint"] = self._get_constraint(info)
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["constraint"] = self._get_constraint(info)
        return obs, reward, done, info

    @property
    def additional_info(self):
        return {
            **getattr(self.env, "additional_info", {}),
            "constraint": {"shape": (1,), "dtype": np.float32},
        }

    def _get_constraint(self, info: dict) -> float:
        return info.get("constraint") or info.get("cost") or 0.
