#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system
#  Update: 2022-12-05, Congsheng Zhang: create file


from gops.utils.common_utils import get_args_from_json
from gops.sys_simulator.sys_run import PolicyRunner
import torch
from gops.algorithm.infadp import ApproxContainer
import os
import numpy as np
import argparse


def load_args(log_policy_dir):
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args


def load_policy(log_policy_dir, trained_policy_iteration):
    # Create policy
    args = load_args(log_policy_dir)
    networks = ApproxContainer(**args)

    # Load trained policy
    log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(trained_policy_iteration)
    networks.load_state_dict(torch.load(log_path))
    return networks


# value_net = load_policy("../../results/INFADP/221115-213410", '50000').v

# def terminal_cost(obs):
#     return -value_net(obs)
runner = PolicyRunner(
    log_policy_dir_list=["../results/FHADP/idpendulum"] * 2,
    trained_policy_iteration_list=["30000", "70000"],
    is_init_info=True,
    init_info={"init_state": [-1, 0.1, -0.1, -0.3, 0.3, -0.3]},
    save_render=False,
    legend_list=["FHADP"],
    dt=0.01,
    # plot_range=[0,200],
    use_opt=False,
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 80,
        "gamma": 1,
        "minimize_options": {
            "max_iter": 200,
            "tol": 1e-3,
            "acceptable_tol": 1e0,
            "acceptable_iter": 10,
            # "print_level": 5,
        },
        "use_terminal_cost": False,
        # "terminal_cost": terminal_cost,
    },
)

runner.run()