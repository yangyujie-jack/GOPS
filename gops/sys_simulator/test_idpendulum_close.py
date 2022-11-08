from gops.create_pkg.create_env import create_env
from gops.env.tools.env_dynamic_checker import check_dynamic


env = create_env(env_id='pyth_idpendulum')

check_dynamic(env, traj_num=5,
                log_policy_dir='./results/SAC/idp_221017-174348',
                policy_iteration='27000')