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


class OnDSampler(BaseSampler):
    def __init__(
        self, 
        sample_batch_size,
        index=0, 
        noise_params=None,
        sample_clip: float = 3.,
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
        self.mb_rew = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_done = np.zeros((self.num_envs, self.horizon), dtype=np.bool_)
        self.mb_tlim = np.zeros((self.num_envs, self.horizon), dtype=np.bool_)
        self.mb_logp = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.need_value_flag = not (alg_name == "FHADP" or alg_name == "INFADP")
        if self.need_value_flag:
            self.gae_lambda = 0.95
            self.mb_val = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_val_std = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_adv = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_adv_sample = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_ret = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_ret_sample = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_info = {}
        self.info_keys = kwargs["additional_info"].keys()
        for k, v in kwargs["additional_info"].items():
            self.mb_info[k] = np.zeros(
                (self.num_envs, self.horizon, *v["shape"]), dtype=v["dtype"]
            )
            self.mb_info["next_" + k] = np.zeros(
                (self.num_envs, self.horizon, *v["shape"]), dtype=v["dtype"]
            )
        self.sample_clip = sample_clip

    def _sample(self) -> dict:
        self.end_ptr = np.zeros(self.num_envs, dtype=np.int32)
        self.start_ptr = np.zeros(self.num_envs, dtype=np.int32)
        for t in range(self.horizon):
            # batch_obs has shape (num_envs, obs_dim)
            if not self._is_vector:
                batch_obs = torch.from_numpy(
                    np.expand_dims(self.obs, axis=0).astype("float32")
                )
            else:
                batch_obs = torch.from_numpy(self.obs.astype("float32"))
            # interact with environment
            experiences = self._step()
            self._process_experiences(experiences, batch_obs, t)

        # wrap collected data into replay format
        mb_data = {
            "obs": torch.from_numpy(self.mb_obs.reshape(-1, *self.obs_dim)),
            "act": torch.from_numpy(self.mb_act.reshape(-1, *self.act_dim)),
            "rew": torch.from_numpy(self.mb_rew.reshape(-1)),
            "done": torch.from_numpy(self.mb_done.reshape(-1)),
            "logp": torch.from_numpy(self.mb_logp.reshape(-1)),
            "time_limited": torch.from_numpy(self.mb_tlim.reshape(-1)),
        }
        if self.need_value_flag:
            mb_data.update({
                "ret": torch.from_numpy(self.mb_ret.reshape(-1)),   
                "ret_sample": torch.from_numpy(self.mb_ret_sample.reshape(-1)),   
                "adv": torch.from_numpy(self.mb_adv.reshape(-1)),
                "adv_sample": torch.from_numpy(self.mb_adv_sample.reshape(-1)),
            })
        for k, v in self.mb_info.items():
            mb_data[k] = torch.from_numpy(v.reshape(-1, *v.shape[2:]))
        return mb_data

    def sample_with_replay_format(self):
        return self.sample()

    def _process_experiences(
        self, 
        experiences: List[Experience],
        batch_obs: torch.Tensor, 
        t: int,
    ):
        if self.need_value_flag:
            with torch.no_grad():
                val_logits = self.networks.value(batch_obs)
            val_mean, val_std = torch.chunk(val_logits, chunks=2, dim=-1)
            self.mb_val[:, t] = val_mean.squeeze(-1).numpy()
            self.mb_val_std[:, t] = val_std.squeeze(-1).numpy()

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
                self.mb_obs[i, t, ...],
                self.mb_act[i, t, ...],
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

            for key in self.info_keys:
                self.mb_info[key][i, t] = info[key]
                self.mb_info["next_" + key][i, t] = next_info[key]

            # calculate value target (mb_ret) & gae (mb_adv)
            if (
                done
                or next_info["TimeLimit.truncated"]
                or t == self.horizon - 1
            ) and self.need_value_flag:
                if not done:
                    last_obs = torch.from_numpy(next_obs).float()
                    with torch.no_grad():
                        last_val_logits = self.networks.value(last_obs)
                    last_val_mean, last_val_std = torch.chunk(last_val_logits, chunks=2, dim=-1)
                    last_val_mean.item()
                    last_val_std.item()
                else:
                    last_val_mean = 0.
                    last_val_std = 0.
                self.end_ptr[i] = t
                self._finish_trajs(i, last_val_mean, last_val_std)
                self.start_ptr[i] = t + 1

    def _finish_trajs(self, env_index: int, last_val: float, last_val_std: float):
        # calculate value target (mb_ret) & gae (mb_adv) whenever episode is finished
        path_slice = slice(self.start_ptr[env_index], self.end_ptr[env_index] + 1)
        val = np.append(self.mb_val[env_index, path_slice], last_val)
        val_std = np.append(self.mb_val_std[env_index, path_slice], last_val_std)
        val_sample = np.clip(
            np.random.normal(val, val_std),
            -self.sample_clip * val_std,
            self.sample_clip * val_std,
        )
        rew = self.mb_rew[env_index, path_slice]
        length = len(rew)
        ret = np.zeros(length)
        ret_sample = np.zeros(length)
        adv = np.zeros(length)
        adv_sample = np.zeros(length)
        gae = 0.
        gae_sample = 0.
        for i in reversed(range(length)):
            delta = rew[i] + self.gamma * val[i + 1] - val[i]
            delta_sample = rew[i] + self.gamma * val_sample[i + 1] - val_sample[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            gae_sample = delta_sample + self.gamma * self.gae_lambda * gae_sample
            ret[i] = gae + val[i]
            ret_sample[i] = gae_sample + val_sample[i]
            adv[i] = gae
            adv_sample[i] = gae_sample
        self.mb_ret[env_index, path_slice] = ret
        self.mb_ret_sample[env_index, path_slice] = ret_sample
        self.mb_adv[env_index, path_slice] = adv
        self.mb_adv_sample[env_index, path_slice] = adv_sample
