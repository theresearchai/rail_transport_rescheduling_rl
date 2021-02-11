import gym
import numpy as np

env = gym.make('rail_transport_rescheduling_rl:small-v0')
obs = env.reset()

while True:
    obs, rew, done, info = env.step({0: np.random.randint(0, 5)})
    # print(obs)
    # print(rew)
    # print(done)
    # print(info)
    env.render()
    if done:
        break

# python test_rollout.py --checkpoint /Users/stlp/Downloads/checkpoint-100 --run APEX --video-dir=/Users/stlp/Desktop/outfile --episodes 10 --env 'rail_transport_rescheduling_rl:small-v0' --config '{"model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}' 

# raise error.DependencyNotInstalled("""Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.""")

# python test_rollout.py /Users/stlp/Downloads/checkpoint-100 --run APEX  --video-dir=/Users/stlp/Desktop/test-gym/outfile --episodes 5 --env 'flatland_sparse' --config '{"env_config": {"test": "true", "generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}' 
