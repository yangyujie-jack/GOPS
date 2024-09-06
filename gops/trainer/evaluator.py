#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluation of trained policy
#  Update Date: 2021-05-10, Yang Guan: renew environment parameters


import numpy as np
import torch
import os
import pickle

from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_alg import create_approx_contrainer
from gops.utils.common_utils import set_seed
from gops.utils.gops_path import gops_path


class Evaluator:
    def __init__(self, index=0, **kwargs):
        kwargs.update({
            "reward_scale": None,
            "repeat_num": None,
            "gym2gymnasium": False,
            "vector_env_num": None,
        })
        self.env = create_env(**kwargs)

        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 400, self.env)

        self.networks = create_approx_contrainer(**kwargs)
        self.render = kwargs["is_render"]

        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
        self.eval_save = kwargs.get("eval_save", True)
        self.eval_accuracy = kwargs.get("eval_accuracy", True)

        self.print_time = 0
        self.print_iteration = -1

        # load init_state_list
        seed = 0
        self.env_id = kwargs["env_id"]
        self.pre_horizon = kwargs["pre_horizon"]
        print("env_id: ", self.env_id)
        print(self.env_id in ["veh3dof_tracking", "veh3dof_tracking_detour", "lq_control"])
        if self.env_id in ["veh3dof_tracking", "veh3dof_tracking_detour", "lq_control"]:
            self.dataset_root = os.path.join(gops_path, f"../mpc_datasets/{self.env_id}/seed_{seed}/horizon_{self.pre_horizon}")
            with open(self.dataset_root + "/init_state_list.pkl", "rb") as f:
                self.init_state_list = pickle.load(f)
            print("init_state_list: ", len(self.init_state_list))
            self.eval_accuracy = True

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def run_an_episode(self, iteration, render=True, ref=0, ind=0):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        obs_list = []
        action_list = []
        reward_list = []
        obs, info = self.env.reset()
        done = 0
        info["TimeLimit.truncated"] = False
        while not (done or info["TimeLimit.truncated"]):
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            action = action.detach().numpy()[0]
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
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "obs_list": obs_list,
        }

        # action accuracy
        action_accuracy = [0., 0., 0., 0.]
        # action_accuracy [max_error_1, mean_error_1, max_error_2, mean_error_2]  
        if self.eval_accuracy and self.env_id in ["veh3dof_tracking", "veh3dof_tracking_detour"]:
            with open(self.dataset_root + f"/ref{ref}_init_{ind}_optdata.pkl", "rb") as f:
                opt_data = pickle.load(f)
            opt_action_list = opt_data['eval_dict']['action_list']
            opt_action_max1 = max([i[0] for i in opt_action_list])
            opt_action_min1 = min([i[0] for i in opt_action_list])
            opt_action_max2 = max([i[1] for i in opt_action_list])
            opt_action_min2 = min([i[1] for i in opt_action_list])
            opt_action_max_list = [opt_action_max1, opt_action_max2]
            opt_action_min_list = [opt_action_min1, opt_action_min2]
        
            # calculate action accuracy
            # action error
            action_dim = 2
            action_accuracy = []
            for j in range(action_dim):
                error_list = []
                if len(action_list) != len(opt_action_list):
                    error_list.append(1.0)
                else:
                    for k, action in enumerate(action_list):
                        error = np.abs(
                            action[j] - opt_action_list[k][j]
                        ) / (
                            opt_action_max_list[j] - opt_action_min_list[j]
                        )
                        error_list.append(error)
                action_accuracy.extend(
                    [max(error_list), sum(error_list)/len(error_list)]
                )

        if self.eval_accuracy and self.env_id in ["lq_control"]:
            with open(self.dataset_root + f"/dataset.pkl", "rb") as f:
                opt_data = pickle.load(f)
            (obs_array, action_array) = opt_data
            opt_action_list = list(action_array[0, ind])
            opt_action_max = max(opt_action_list)
            opt_action_min = min(opt_action_list)
            print(
                'action', action_list[0], 
                'opt_action', opt_action_list[0],
                'obs', obs_list[0],
                'opt_obs', obs_array[0, ind, 0],
            )
            # calculate action accuracy
            # action error
            error_list = []
            if len(action_list) != len(opt_action_list):
                error_list.append(1.0)
            else:
                for k, action in enumerate(action_list):
                    error = np.abs(action - opt_action_list[k]) / (opt_action_max - opt_action_min)
                    error_list.append(float(error))
            action_accuracy = [max(error_list), sum(error_list)/len(error_list), 0., 0.]
        # print("action_accuracy", action_accuracy)

        if self.eval_save:
            np.save(
                self.save_folder
                + "/evaluator/iter{}_ep{}".format(iteration, self.print_time),
                eval_dict,
            )
        episode_return = sum(reward_list)
        return episode_return, action_accuracy

    def run_n_episodes(self, n, iteration):
        episode_return_list = []
        if self.eval_accuracy:
            if self.env_id == "veh3dof_tracking":
                action_accuracy_array = np.ones((4*n, 4))
                ref_range = [0, 1, 2, 3]
            elif self.env_id == "veh3dof_tracking_detour":
                action_accuracy_array = np.ones((n, 4))
                ref_range = [9]
            elif self.env_id == "lq_control":
                action_accuracy_array = np.ones((n, 4))
                ref_range = [0]
            for ref in ref_range:
                ref_ind = ref_range.index(ref)
                for ind in range(n):
                    episode_return, action_accuracy = self.run_an_episode(iteration, self.render, ref, ind)
                    episode_return_list.append(episode_return)
                    action_accuracy_array[ind+ref_ind*n] = action_accuracy
            action_accuracy_mean = np.mean(action_accuracy_array, axis=0)
        else:
            for ind in range(n):
                episode_return, action_accuracy = self.run_an_episode(iteration, self.render, 0, 0)
                episode_return_list.append(episode_return)
            action_accuracy_mean = np.zeros(4)
        episode_return_mean = np.mean(episode_return_list)
        return episode_return_mean, action_accuracy_mean

    def run_evaluation(self, iteration):
        return self.run_n_episodes(self.num_eval_episode, iteration)
