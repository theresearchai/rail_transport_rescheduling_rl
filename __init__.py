from gym.envs.registration import register
import yaml
import os

def get_env_config(file):
    with open(file) as f:
        env_config = yaml.safe_load(f)
    return env_config['env_config']
# print(env_config)
register(
    id='base-v0',
    entry_point='rail_transport_rescheduling_rl.envs:FlatlandBase'
)
register(
    id='small-v0',
    entry_point='rail_transport_rescheduling_rl.envs:FlatlandSparse',
    kwargs={'env_config': get_env_config('rollout_envs/small.yaml')}
)

register(
    id='medium-v0',
    entry_point='rail_transport_rescheduling_rl.envs:FlatlandSparse',
    kwargs={'env_config': get_env_config('rollout_envs/medium.yaml')}
)

## Register more environments here