from gym.envs.registration import register
import yaml

with open('/Users/stlp/Desktop/test-gym/small.yaml') as f:
    env_config = yaml.safe_load(f)
# print(env_config)
register(
    id='base-v0',
    entry_point='rail_transport_rescheduling_rl.envs:FlatlandBase'
)
register(
    id='small-v0',
    entry_point='rail_transport_rescheduling_rl.envs:FlatlandSparse',
    kwargs={'env_config': env_config['env_config']}
)