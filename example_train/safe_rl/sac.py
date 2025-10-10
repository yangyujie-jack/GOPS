#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

import argparse

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Key Parameters
    parser.add_argument("--env_id", type=str, default="safe_rl")
    parser.add_argument("--safe_rl_env_id", type=str, default="Safety-Gymnasium-SafetyPointGoal1-v0")
    parser.add_argument("--algorithm", type=str, default="SAC")
    parser.add_argument("--enable_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    # Parameters of approximate function
    parser.add_argument("--value_func_name", type=str, default="ActionValue")
    parser.add_argument("--value_func_type", type=str, default="MLP")
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument("--value_hidden_activation", type=str, default="relu")
    parser.add_argument("--policy_func_name", type=str, default="StochaPolicy")
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_act_distribution", type=str, default="TanhGaussDistribution")
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument("--policy_hidden_activation", type=str, default="relu")
    parser.add_argument("--policy_min_log_std", type=float, default=-20.)
    parser.add_argument("--policy_max_log_std", type=float, default=2.)

    # Parameters for RL algorithm
    parser.add_argument("--q_learning_rate", type=float, default=0.0001)
    parser.add_argument("--policy_learning_rate", type=float, default=0.0001)
    parser.add_argument("--alpha_learning_rate", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--auto_alpha", type=bool, default=True)

    # Parameters for trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer")
    parser.add_argument("--max_iteration", type=int, default=1000000)
    parser.add_argument("--buffer_name", type=str, default="replay_buffer")
    parser.add_argument("--buffer_warm_size", type=int, default=10000)
    parser.add_argument("--buffer_max_size", type=int, default=1000000)
    parser.add_argument("--replay_batch_size", type=int, default=256)
    parser.add_argument("--sample_interval", type=int, default=100)
    parser.add_argument("--apprfunc_save_interval", type=int, default=100000)
    parser.add_argument("--log_save_interval", type=int, default=10000)

    # Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    parser.add_argument("--sample_batch_size", type=int, default=100)

    # Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10000)

    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    new_args = init_args(env, **{**args, "env_id": args["safe_rl_env_id"]})
    args = {**new_args, "env_id": args["env_id"]}

    # Steps to start training
    alg = create_alg(**args)
    sampler = create_sampler(**args)
    buffer = create_buffer(**args)
    evaluator = create_evaluator(**args, constraint=True)
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)
    trainer.train()
