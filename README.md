# Reinforcement Learning in VRSP

## Introduction

## Motivation

## Setup

### Flatland Environment
You can now run the following:
```sh
conda env create -f environment-cpu.yml # creates the cpu environment
conda activate flatland-baseline-cpu-env # activates it
```

If you have access to GPU:
```sh
conda env create -f environment-gpu.yml # creates the gpu environment
conda activate flatland-baseline-gpu-env # activates it
```

If using Colab Notebook:
See [tutorial](https://colab.research.google.com/drive/1aZoKFkuNYeWKG1m_S6sadD1YG6nFnf67?usp=sharing).

### Gym Environment
The flatland enviroments are registered in `ray.tune`, which may not be compatible with some OpenAI Gym functionalities. 

You can customize and register enviroments in `gym` with the following steps.

1. Write your own class `MyFlatland` in `envs/my_flatland.py`. Skip to step 3 if use  enviroments provided by Flatland.
2. Import your class in `envs/__init__.py`
```python
from envs.my_flatland import MyFlatland
```
3. Register env in `__init__.py`

```python
#Flatland env
env_config = {
	'observation': 'tree'
    'observation_config':{
		'max_depth': 2
        'shortest_path_max_depth': 30
	}
	'generator': sparse_rail_generator
    'generator_config': small_stoch_v0
}

register(
    id='small-v0',
    entry_point='rail_transport_rescheduling_rl.envs:FlatlandSparse',
    kwargs={'env_config': env_config}
)

#Cutomized env
register(
    id='myenv-v0',
    entry_point='rail_transport_rescheduling_rl.envs:MyFlatland',
    kwargs=
)
```

4. Remember the unique id.
5. Go to the parent folder of `rail_transport_rescheduling_rl` and run `pip install -e rail_transport_rescheduling_rl`

Now you can initialize gym enviroment with your environment id.
```python
import gym
gym.make('rail_transport_rescheduling_rl:myenv-v0')
```

### Render

[WARNING] For some reason, the `Monitor` funtion in `rollout.py` only works with gym environment.

Render needs ffmpeg or avconv executables.

On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.


------------

## Train Models
Simply run `python train.py -f config.yaml` to train a model with the configuration file name `config.yaml` or [common parameters](https://docs.ray.io/en/master/rllib-training.html#common-parameters) used by RLlib.
All configuration files to run the experiments can be found in `baselines`.

### Stopping Criteria
Add the `stop` section in config file, for example:
```
stop:
	timesteps_total: 15000000
	# training_iteration: 1000
	# episode_reward_mean: 200
```
See RLlib for more stopping criteria options.

### Save Checkpoints 
Config `local_dir ` as the path to store checkpoints.

Example:
```
checkpoint_freq: 10 #save checkpoint every 10 iterations
checkpoint_at_end: True #save checkpoint after training is done
keep_checkpoints_num: 1000 #maximum number of checkpoints to be saved
local_dir: /content/gdrive/MyDrive/checkpoints

config:
	env_config:
		save_checkpoint: True #Enable checkpoint storing
```

### Resume Training from Checkpoints
Config `restore` as the path of a previously saved checkpoint to resume training. Some saved checkpoints can be found [here](https://drive.google.com/drive/folders/1AdPSM1ZiW5XWv0gl7WzQw8qFEywatfZf?usp=sharing).

Example:
```yaml
restore: /content/gdrive/MyDrive/checkpoints/apex-tree-obs-medium-v0-skip/APEX_flatland_sparse_0_2021-02-05_09-54-43o7u5dtex/checkpoint_30/checkpoint-30
```

### Sync Results to W&B
1. Install wandb, this should be done after the enviroment setup . If not, run 
```sh
pip install wandb
```
2. Create and login to your account.
```sh
wandb login
```
Follow the instruction and copy your API key to terminal.

3. Config your wandb account info in the `wandb` section.
```
config:
	env_config:
		wandb:
			project: action-masking-skipping #your project name
			entity: qye25 #your user name
			tags: ["medium_v0", "tree_obs", "apex", "skip"] #tag your model 
```

### Render 
Visualization of each training iteration will be uploaded to W&B but this will be extremely time consuming.
```
config:
	env_config:
		render: human 
```

## Rollout Trained Models
Run `rollout.py` to evaluate a trained model.

Parameters  | Comments
------------- | -------------
checkpoint | path to checkpoint
run | algorithm 
episodes | number of rollout episodes
env | enviroment
config | environment and model configuration
cfile | load configuration from file without using `--config`
video-dir | path to store rendered videos 
project | W&B project name, sync output files in `video-dir` to W&B

### Examples

1. Run with command line configuration.
```sh
python rollout.py --run APEX --episodes 5 --checkpoint=/Users/stlp/Downloads/checkpoint-100  --env 'flatland_sparse' --config '{"env_config": {"test": "true", "generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}' 
```

2. Run with a configuration file that specifies enviroment settings.
```sh
python rollout.py --run MARWIL --episodes 100 --cfile=/content/gdrive/MyDrive/checkpoints/medium.yaml --checkpoint=/content/gdrive/MyDrive/checkpoints/flatland-random-sparse-small-tree-fc-marwil-il/MARWIL_flatland_sparse_0_beta=1_2021-02-05_10-05-00014kys69/checkpoint_500/checkpoint-500 --env 'flatland_sparse' 
```

3. Use `--video-dir` to render and save rollout results. Remember to use Gym environents.
```sh
python rollout.py --checkpoint /Users/stlp/Downloads/checkpoint-100 --run APEX --video-dir=/Users/stlp/Desktop/VRSP/outfile --episodes 1 --env 'rail_transport_rescheduling_rl:small-v0' --config '{"num_workers": 4, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}' 
```


## Hyperparameter Tuning
We can use [W&B sweeps](https://docs.wandb.ai/sweeps) to automatically do hyperparameter tuning.

1. Initialize sweep with a a sweep config file
```sh
wandb sweep sweep.yaml
```
2. Copy the sweep ID printed in terminal
3. Launch agent(s)
```sh
wandb agent your-sweep-id
```

We can also use W&B sweeps to rollout multiple models or maps. See `rollout_sweep_example.yaml`.


## Reference

[1] Flatland “https://flatland.aicrowd.com”

[2] David Silver’s RL Lectures “https://www.davidsilver.uk/teaching/”

[3] "Real World Applications of Flatland": Panel Discussion with SBB, DeutschBahn, SNCF. “https://slideslive.com/38942748/real-world-applications-of-flatland-panel-discussion-with-sbb-deutschbahn-sncf”

[4] Wälter, Jonas (2020), Existing and Novel Approaches to the Vehicle Rescheduling Problem (VRSP). Masters Thesis, HSR Hochschule für Technik Rapperswil.



# 🚂 Flatland Baselines

This repository contains reinforcement learning baselines for the [NeurIPS 2020 Flatland Challenge](https://www.aicrowd.com/challenges/neurips-2020-flatland-challenge/) based on RLlib.

**Read the [baseline documentation](https://flatland.aicrowd.com/research/baselines.html) to see how to setup and use the baselines.**

>>>
Looking for something simpler? We also provide a DQN method implemented from scratch using PyTorch: https://gitlab.aicrowd.com/flatland/flatland-examples 
>>>

Notes
---

- The basic structure of this repository is adapted from [https://github.com/spMohanty/rl-experiments/](https://github.com/spMohanty/rl-experiments/)
- The baselines are under the MIT license

Main links
---

* [Flatland documentation](https://flatland.aicrowd.com/)
* [NeurIPS 2020 Challenge](https://www.aicrowd.com/challenges/neurips-2020-flatland-challenge/)

Communication
---

* [Discord Channel](https://discord.com/invite/hCR3CZG)
* [Discussion Forum](https://discourse.aicrowd.com/c/neurips-2020-flatland-challenge)
* [Issue Tracker](https://gitlab.aicrowd.com/flatland/flatland/issues/)
