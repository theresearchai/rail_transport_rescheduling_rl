### Environment Setup

#### Flatland Environment
You can now run the following:
```shell
conda env create -f environment-cpu.yml # creates the flatland-rl environment
conda activate flatland-baseline-cpu-env # activates it
```

If you have access to GPU:
```shell
conda env create -f environment-gpu.yml # creates the flatland-rl environment
conda activate flatland-baseline-gpu-env # activates it
```

If using Colab Notebook:
See tutorial

#### Gym Environment
The flatland enviroments are registered in `ray.tune`, which may not be compatible with some OpenAI Gym functionalities. 
For some reason, the `Monitor` funtion in `rollout.py` only works with gym environment.
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
5. Run `pip install -e rail_transport_rescheduling_rl`

Now you can initialize gym enviroment with your environment id.
```python
import gym
gym.make('rail_transport_rescheduling_rl:myenv-v0')
```

------------

### Train Models
Simply run `python train.py -f config.yaml` to train a model with the configuration file name `config.yaml` or [common parameters](https://docs.ray.io/en/master/rllib-training.html#common-parameters) used by RLlib.
All configuration files to run the experiments can be found in `baselines`.

#### Stopping Criteria
Add the `stop` section in config file, for example:
```yaml
stop:
	timesteps_total: 15000000
	# training_iteration: 1000
	# episode_reward_mean: 200
```
See RLlib for more stopping criteria options.

#### Save Checkpoints 
Config `local_dir ` as the path to store checkpoints.

Example:
```yaml
checkpoint_freq: 10 #save checkpoint every 10 iterations
checkpoint_at_end: True #save checkpoint after training is done
keep_checkpoints_num: 1000 #maximum number of checkpoints to be saved
local_dir: /content/gdrive/MyDrive/checkpoints

config:
	env_config:
		save_checkpoint: True #Enable checkpoint storing
```

#### Resume Training from Checkpoints
Config `restore` as the path of a previously saved checkpoint to resume training.
Example:
```shell
restore: /content/gdrive/MyDrive/checkpoints/apex-tree-obs-medium-v0-skip/APEX_flatland_sparse_0_2021-02-05_09-54-43o7u5dtex/checkpoint_30/checkpoint-30
```

#### Sync Results to W&B
1. Install wandb, this should be done after the enviroment setup . If not, run `pip install wandb`.
2. Create and login to your account.
```shell
wandb login
```
Follow the instruction and copy your API key to terminal.

3. Config your wandb account info in the `wandb` section.
```yaml
config:
	env_config:
		wandb:
			project: action-masking-skipping #your project name
			entity: qye25 #your user name
			tags: ["medium_v0", "tree_obs", "apex", "skip"] #tag your model 
```

#### Render 
Visualization of each training iteration will be uploaded to W&B but this will be extremely time consuming.
```shell
config:
	env_config:
		render: human 
```

### Rollout Trained Models
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


##### Examples

1. Run with command line configuration.
```shell
python rollout.py --checkpoint=/Users/stlp/Downloads/checkpoint-100 --run APEX  --video-dir=/Users/stlp/Desktop/test-gym/outfile --episodes 5 --env 'flatland_sparse' --config '{"env_config": {"test": "true", "generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}' 
```

2. Run with a configuration file that specifies enviroment settings.
```shell
python rollout.py --cfile=/content/gdrive/MyDrive/checkpoints/medium.yaml --checkpoint=/content/gdrive/MyDrive/checkpoints/flatland-random-sparse-small-tree-fc-marwil-il/MARWIL_flatland_sparse_0_beta=1_2021-02-05_10-05-00014kys69/checkpoint_500/checkpoint-500 --env=flatland_sparse --episodes=100 --run=MARWIL
```

### Hyperparameter Tuning
We can use [W&B sweeps](https://docs.wandb.ai/sweeps) to automatically do hyperparameter tuning.

1. Initialize sweep with a a sweep config file
```shell
wandb sweep sweep.yaml
```
2. Copy the sweep ID printed in terminal
3. Launch agent(s)
```shell
wandb agent your-sweep-id
```

We can also use W&B sweeps to rollout multiple models or maps. See `rollout_sweep_example.yaml`.


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
