#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluation of trained policy
#  Update Date: 2021-05-10, Yang Guan: renew environment parameters


import numpy as np
import torch

from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_alg import create_approx_contrainer
from gops.utils.common_utils import set_seed, seeding


class Evaluator:
    def __init__(self, index=0, constraint: bool = False, **kwargs):
        kwargs.update({
            "reward_scale": None,
            "repeat_num": None,
            "gym2gymnasium": False,
            "vector_env_num": None,
        })
        self.env = create_env(**kwargs)

        set_seed(kwargs["trainer"], kwargs["seed"], index + 400)
        self.rng, _ = seeding(kwargs["seed"] + index + 400)

        self.networks = create_approx_contrainer(**kwargs)
        self.render = kwargs["is_render"]

        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
        self.eval_save = kwargs.get("eval_save", True)
        self.constraint = constraint

        self.print_time = 0
        self.print_iteration = -1

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def run_an_episode(self, iteration, render=True):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        obs_list = []
        action_list = []
        reward_list = []
        if self.constraint:
            violation_list = []
        obs, info = self.env.reset(seed=int(self.rng.integers(0, 2 ** 32 - 1)))
        done = 0
        info["TimeLimit.truncated"] = False
        while not (done or info["TimeLimit.truncated"]):
            with torch.no_grad():
                logits = self.networks.policy(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0)
            action = self.networks.create_action_distributions(logits).mode().numpy()
            next_obs, reward, done, next_info = self.env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            obs = next_obs
            info = next_info
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                self.env.render()
            reward_list.append(reward)
            if self.constraint:
                violation_list.append(float(info["constraint"] > 0))
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "obs_list": obs_list,
        }
        if self.eval_save:
            np.save(
                self.save_folder
                + "/evaluator/iter{}_ep{}".format(iteration, self.print_time),
                eval_dict,
            )
        episode_return = sum(reward_list)
        if self.constraint:
            episode_violation = sum(violation_list)
            return episode_return, episode_violation
        else:
            return episode_return

    def run_n_episodes(self, n, iteration):
        episode_return_list = []
        if self.constraint:
            episode_violation_list = []
        for _ in range(n):
            episode_result = self.run_an_episode(iteration, self.render)
            if self.constraint:
                episode_return, episode_violation = episode_result
                episode_return_list.append(episode_return)
                episode_violation_list.append(episode_violation)
            else:
                episode_return_list.append(episode_result)
        if self.constraint:
            return np.mean(episode_return_list), np.mean(episode_violation_list)
        else:
            return np.mean(episode_return_list)

    def run_evaluation(self, iteration):
        return self.run_n_episodes(self.num_eval_episode, iteration)
