#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Utils Function
#  Update Date: 2021-03-10, Yuhang Zhang: Create codes


import time
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional

from gops.utils.tensorboard_setup import tb_tags
from gops.utils.act_distribution import *
import random
import importlib

logger = logging.getLogger(__name__)

def get_activation_func(key: str):
    assert isinstance(key, str)

    activation_func = None
    if key == "relu":
        activation_func = nn.ReLU

    elif key == "elu":
        activation_func = nn.ELU

    elif key == "gelu":
        activation_func = nn.GELU

    elif key == "tanh":
        activation_func = nn.Tanh

    elif key == "linear":
        activation_func = nn.Identity

    if activation_func is None:
        print("input activation name:" + key)
        raise RuntimeError

    return activation_func


def get_apprfunc_dict(key: str, type=None, **kwargs):
    var = dict()
    var["apprfunc"] = kwargs[key + "_func_type"]
    var["name"] = kwargs[key + "_func_name"]
    var["obs_dim"] = kwargs["obsv_dim"]
    var["min_log_std"] = kwargs.get(key + "_min_log_std", float("-inf"))
    var["max_log_std"] = kwargs.get(key + "_max_log_std", float("inf"))
    var["std_sype"] = kwargs.get(key + "_std_sype", "mlp_shared")

    apprfunc_type = kwargs[key + "_func_type"]
    if apprfunc_type == "MLP" or apprfunc_type == "RNN":
        var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
    elif apprfunc_type == "GAUSS":
        var["num_kernel"] = kwargs[key + "_num_kernel"]
    elif apprfunc_type == "CNN":
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
        var["conv_type"] = kwargs[key + "_conv_type"]
    elif apprfunc_type == "CNN_SHARED":
        if key == "feature":
            var["conv_type"] = kwargs["conv_type"]
        else:
            var["feature_net"] = kwargs["feature_net"]
            var["hidden_activation"] = kwargs[key + "_hidden_activation"]
            var["output_activation"] = kwargs[key + "_output_activation"]
    elif apprfunc_type == "POLY":
        var["degree"] = kwargs[key + "_degree"]
    elif apprfunc_type == "GAUSS":
        var["num_kernel"] = kwargs[key + "_num_kernel"]
    else:
        raise NotImplementedError

    if kwargs["action_type"] == "continu":
        var["act_high_lim"] = kwargs["action_high_limit"]
        var["act_low_lim"] = kwargs["action_low_limit"]
        var["act_dim"] = kwargs["action_dim"]

    else:
        var["act_num"] = kwargs["action_num"]

    if kwargs["policy_act_distribution"] == "default":
        if kwargs["action_type"] == "continu":
            if kwargs["policy_func_name"] == "StochaPolicy":  # todo: add TanhGauss
                var["action_distirbution_cls"] = GaussDistribution
            elif kwargs["policy_func_name"] == "DetermPolicy":
                var["action_distirbution_cls"] = DiracDistribution
        else:
            if kwargs["policy_func_name"] == "StochaPolicyDis":
                var["action_distirbution_cls"] = CategoricalDistribution
            elif kwargs["policy_func_name"] == "DetermPolicyDis":
                var["action_distirbution_cls"] = ValueDiracDistribution
    else:

        var["action_distirbution_cls"] = getattr(
            sys.modules[__name__], kwargs["policy_act_distribution"]
        )

    return var


def change_type(obj):
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):  # add this line
        return obj.tolist()  # add this line
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = change_type(v)
        return obj
    elif isinstance(obj, list):
        for i, o in enumerate(obj):
            obj[i] = change_type(o)
        return obj
    else:
        return obj


def random_choice_with_index(obj_list):
    obj_len = len(obj_list)
    random_index = random.choice(list(range(obj_len)))
    random_value = obj_list[random_index]
    return random_value, random_index


def array_to_scalar(arrayLike):
    """Convert size-1 array to scalar"""
    return arrayLike if isinstance(arrayLike, (int, float)) else arrayLike.item()


def seed_everything(seed: Optional[int] = None) -> int:
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        seed = random.randint(min_seed_value, max_seed_value)

    elif not isinstance(seed, int):
        seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def set_seed(trainer_name, seed, offset, env=None):
    """
    When trainer_name is `**_async_**` or `**_sync_**`, set random seed for the subprocess and gym env, 
    else only set the subprocess for gym env

    Parameters
    ----------
    trainer_name : str
        trainer_name
    seed : int
        global seed
    offset : int
        the offset of random seed for the subprocess
    env : gym.Env, optional
        a gym env needs to set random seed, by default None

    Returns
    -------
    (int, gym.Env)
        the random seed for the subprocess, a gym env which the random seed is set
    """

    if trainer_name.split("_")[1] in ["async", "sync"]:
        print("Setting seed of a subprocess to {}".format(seed + offset))
        seed_everything(seed + offset)
        if env is not None:
            env.seed(seed + offset)
        return seed + offset, env

    else:
        if env is not None:
            env.seed(seed)
        return None, env


class FreezeParameters:
    def __init__(self, modules):
        """
      Context manager to locally freeze gradients.
      In some cases with can speed up computation because gradients aren't calculated for these listed modules.
      example:
      ```
      with FreezeParameters([module]):
          output_tensor = module(input_tensor)
      ```
      :param modules: iterable of modules. used to call .parameters() to freeze gradients.
      """
        self.modules = modules
        self.param_states = [
            p.requires_grad for p in get_parameters(self.modules)
        ]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


def get_parameters(modules):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class ModuleOnDevice:
    def __init__(self, module, device):
        self.module = module
        self.prev_device = next(module.parameters()).device.type
        self.new_device = device
        self.different_device = self.prev_device != self.new_device

    def __enter__(self):
        if self.different_device:
            self.module.to(self.new_device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.different_device:
            self.module.to(self.prev_device)


def get_args_from_json(json_file_path, args_dict):
    import json

    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict


def mp4togif(path):
    try:
        import moviepy.editor as mp
    except:
        print("If you want to convert mp4 to gif, install package `moviepy`")
        return None

    if os.path.exists(path):
        clip = mp.VideoFileClip(path)
        if path.endswith(".mp4"):
            out_path = path[:-4] + ".gif"
        else:
            out_path = path + ".gif"
        clip.write_gif(out_path)
    else:
        print(f"`{path}` dose not exist")