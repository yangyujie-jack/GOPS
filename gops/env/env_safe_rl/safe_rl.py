import safe_rl_env
from gops.env.wrapper.constraint_info import ConstraintInfo


def env_creator(safe_rl_env_id: str, **kwargs):
    env = safe_rl_env.make(env_id=safe_rl_env_id, gym_step=True)
    env = ConstraintInfo(env)
    return env
