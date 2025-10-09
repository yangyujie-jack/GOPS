#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

from typing import List

import numpy as np
import torch

from gops.trainer.sampler.base import BaseSampler, Experience


class OnQSampler(BaseSampler):
    def __init__(
        self, 
        sample_batch_size,
        index=0, 
        noise_params=None,
        **kwargs
    ):
        super().__init__(
            sample_batch_size,
            index, 
            noise_params,
            **kwargs
        )
        
        alg_name = kwargs["algorithm"]
        self.gamma = 0.99  #? why hard-coded?
        if self._is_vector:
            self.obs_dim = self.env.single_observation_space.shape
            self.act_dim = self.env.single_action_space.shape
        else:
            self.obs_dim = self.env.observation_space.shape
            self.act_dim = self.env.action_space.shape

        self.mb_obs = np.zeros(
            (self.num_envs, self.horizon, *self.obs_dim), dtype=np.float32
        )
        self.mb_act = np.zeros(
            (self.num_envs, self.horizon, *self.act_dim), dtype=np.float32
        )
        self.mb_next_obs = np.zeros(
            (self.num_envs, self.horizon, *self.obs_dim), dtype=np.float32
        )
        self.mb_next_act = np.zeros(
            (self.num_envs, self.horizon, *self.act_dim), dtype=np.float32
        )
        self.mb_rew = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_done = np.zeros((self.num_envs, self.horizon), dtype=np.bool_)
        self.mb_tlim = np.zeros((self.num_envs, self.horizon), dtype=np.bool_)
        self.mb_logp = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_next_logp = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.need_value_flag = not (alg_name == "FHADP" or alg_name == "INFADP")
        if self.need_value_flag:
            self.gae_lambda = 0.95
            self.mb_val = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_adv = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_ret = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_info = {}
        self.info_keys = kwargs["additional_info"].keys()
        for k, v in kwargs["additional_info"].items():
            self.mb_info[k] = np.zeros(
                (self.num_envs, self.horizon, *v["shape"]), dtype=v["dtype"]
            )
            self.mb_info["next_" + k] = np.zeros(
                (self.num_envs, self.horizon, *v["shape"]), dtype=v["dtype"]
            )

    def _sample(self) -> dict:
        self.end_ptr = np.zeros(self.num_envs, dtype=np.int32)
        self.start_ptr = np.zeros(self.num_envs, dtype=np.int32)
        for t in range(self.horizon):
            # interact with environment
            experiences = self._step()
            self._process_experiences(experiences, t)

        # wrap collected data into replay format
        mb_data = {
            "obs": torch.from_numpy(self.mb_obs.reshape(-1, *self.obs_dim)),
            "act": torch.from_numpy(self.mb_act.reshape(-1, *self.act_dim)),
            "rew": torch.from_numpy(self.mb_rew.reshape(-1)),
            "done": torch.from_numpy(self.mb_done.reshape(-1)),
            "logp": torch.from_numpy(self.mb_logp.reshape(-1)),
            "time_limited": torch.from_numpy(self.mb_tlim.reshape(-1)),
            "next_obs": torch.from_numpy(self.mb_next_obs.reshape(-1, *self.obs_dim)),
            "next_act": torch.from_numpy(self.mb_next_act.reshape(-1, *self.act_dim)),
            "next_logp": torch.from_numpy(self.mb_next_logp.reshape(-1)),
        }
        if self.need_value_flag:
            mb_data.update({
                "ret": torch.from_numpy(self.mb_ret.reshape(-1)),   
                "adv": torch.from_numpy(self.mb_adv.reshape(-1)),
            })
        for k, v in self.mb_info.items():
            mb_data[k] = torch.from_numpy(v.reshape(-1, *v.shape[2:]))
        return mb_data

    def sample_with_replay_format(self):
        return self.sample()

    def _process_experiences(
        self, 
        experiences: List[Experience],
        t: int
    ):
        store_next = False
        for i in np.arange(self.num_envs):
            (
                obs, 
                action, 
                reward, 
                done, 
                info, 
                next_obs, 
                next_info, 
                logp,
            ) = experiences[i]

            (
                self.mb_obs[i, t],
                self.mb_act[i, t],
                self.mb_rew[i, t],
                self.mb_done[i, t],
                self.mb_tlim[i, t],
                self.mb_logp[i, t],
            ) = (
                obs,
                action,
                reward,
                done,
                next_info["TimeLimit.truncated"],
                logp,
            )

            if store_next:
                self.mb_next_obs[i, t - 1] = obs
                self.mb_next_act[i, t - 1] = action
                self.mb_next_logp[i, t - 1] = logp

            for key in self.info_keys:
                self.mb_info[key][i, t] = info[key]
                self.mb_info["next_" + key][i, t] = next_info[key]

            obs_tensor = torch.from_numpy(obs).float()
            act_tensor = torch.from_numpy(action).float()
            with torch.no_grad():
                # act_tensor = self.networks.create_action_distributions(
                #     self.networks.policy(obs_tensor)).mode()
                # act_tensor = self.networks.create_action_distributions(
                #     self.networks.policy(obs_tensor)).sample()[0]
                val = self.networks.q(obs_tensor, act_tensor)
            self.mb_val[i, t] = val.item()

            # calculate value target (mb_ret) & gae (mb_adv)
            if (
                done
                or next_info["TimeLimit.truncated"]
                or t == self.horizon - 1
            ) and self.need_value_flag:
                last_obs = torch.from_numpy(next_obs).float()
                with torch.no_grad():
                    logits = self.networks.policy(last_obs)
                    action_distribution = self.networks.create_action_distributions(logits)
                    last_act, last_logp = action_distribution.sample()
                    # last_act = action_distribution.mode()
                    last_val = self.networks.q(last_obs, last_act)
                est_last_value = last_val.item() * (1 - done)
                self.end_ptr[i] = t
                self._finish_trajs(i, est_last_value)
                self.start_ptr[i] = t + 1
                self.mb_next_obs[i, t] = next_obs
                self.mb_next_act[i, t] = last_act.numpy()
                # self.mb_next_logp[i, t] = last_logp.numpy()
                store_next = False
            else:
                store_next = True

    def _finish_trajs(self, env_index: int, est_last_val: float):
        # calculate value target (mb_ret) & gae (mb_adv) whenever episode is finished
        path_slice = slice(self.start_ptr[env_index], self.end_ptr[env_index] + 1)
        value_preds_slice = np.append(self.mb_val[env_index, path_slice], est_last_val)
        obs_slice = self.mb_obs[env_index, path_slice]
        obs_slice_tensor = torch.from_numpy(obs_slice)
        with torch.no_grad():
            # logits = self.networks.policy(obs_slice_tensor)
            # mean_act_slice = self.networks.create_action_distributions(logits).mode()
            # mean_act_slice = self.networks.create_action_distributions(logits).sample()[0]
            mean_act_slice = torch.from_numpy(self.mb_act[env_index, path_slice])
            value_preds_mean_act_slice = self.networks.q(obs_slice_tensor, mean_act_slice).numpy()
        rews_slice = self.mb_rew[env_index, path_slice]
        length = len(rews_slice)
        ret = np.zeros(length)
        adv = np.zeros(length)
        gae = 0.0
        for i in reversed(range(length)):
            delta = (
                rews_slice[i]
                + self.gamma * value_preds_slice[i + 1]
                - value_preds_mean_act_slice[i]
            )
            gae = delta + self.gamma * self.gae_lambda * gae
            ret[i] = gae + value_preds_slice[i]
            adv[i] = gae
        self.mb_ret[env_index, path_slice] = ret
        self.mb_adv[env_index, path_slice] = adv
