# Deep Reinforcement Learning in VRSP

## Introduction

The vehicle rescheduling problem (VRSP) is a combinatorial optimization and integer programming problem seeking an optimal solution that makes all the trains arrive at their destinations with minimal total travel time when some previously assigned trips are disrupted and required to be rescheduled with minimal delay.

## Motivation

Currently, researchers are seeking systems and solutions to VRSP (Vehicle Rescheduling Problem) that can optimize large-scale traffic and quickly adapt to new environments. Deep Reinforcement Learning is demonstrated to be an alternative to Operation Research in solving railway traffic optimization problems due to its fast inference time and high adaptability. Hence, this project will focus on the novel Deep RL approach to evaluate its performance in solving VRSP.

## Data
[Flatland environment](https://flatland.aicrowd.com) is used as a simulator to generate simplified railway networks and trains along with the data samples for the RL training and testing tasks.

### Environment
Two type of maps are generated for training and testing:

- Small maps with 25 x 25 grid size and 5 trains
![Small Map](img/Map(S).gif)

- Large maps with 50 x 50 grid size and 10 trains
![Large Map](img/Map(L).gif)

### Actions

Action  | Description
------------- | -------------
DO_NOTHING | <ul> <li> If the agent is already moving, it continues moving.</li> <li>If the agent is stopped, it stays stopped.</li> <li>Special case: if the agent is at a dead-end, this action will result in the train turning around.</li> </ul>
MOVE_LEFT |<ul> <li> If the agent is at an intersection with an allowed transition to its left, it turns left. Otherwise, the action has no effect.</li> <li>If the agent is stopped, it starts moving (if allowed).</li> </ul> 
MOVE_FORWARD | <ul> <li>If the agent is at an intersection with an allowed transition straight ahead, it moves ahead. Otherwise, the action has no effect.</li> <li>If the agent is stopped, it starts moving.</li>  <li>Special case: if the agent is at a dead-end, this action will result in the train turning around.</li> </ul>
MOVE_RIGHT |<ul> <li> The same as deviate left but for right turns.</li> </ul>
STOP_MOVING | <ul> <li>This action causes the agent to stop at the current cell.</li> </ul>

### Rewards

At each time step, each agent (train) receives a combination of a local and a global reward:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=r_i(t) = \alpha r_l(t) %2b \beta r_g(t)">
</p>

Locally, the agent receives <img src="https://render.githubusercontent.com/render/math?math=r_l = 0"> after it has reached its target location, otherwise, <img src="https://render.githubusercontent.com/render/math?math=r_l = 1">. 

The global reward <img src="https://render.githubusercontent.com/render/math?math=r_g"> only returns <img src="https://render.githubusercontent.com/render/math?math=1"> when all agents have reached their targets, otherwise, <img src="https://render.githubusercontent.com/render/math?math=r_g = 0">. 

<img src="https://render.githubusercontent.com/render/math?math=\alpha"> and <img src="https://render.githubusercontent.com/render/math?math=\beta"> are factors for tuning collaborative behavior.







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


## Results
Find result analysis and presentation [here](https://docs.google.com/presentation/d/1IZWOUVTYFUjoeLVcMOUuYLKwFZEKg6PVfytiimryaIY/present?usp=sharing).

## Reference

[1] Flatland “https://flatland.aicrowd.com”

[2] David Silver’s RL Lectures “https://www.davidsilver.uk/teaching/”

[3] "Real World Applications of Flatland": Panel Discussion with SBB, DeutschBahn, SNCF. “https://slideslive.com/38942748/real-world-applications-of-flatland-panel-discussion-with-sbb-deutschbahn-sncf”

[4] Wälter, Jonas (2020), Existing and Novel Approaches to the Vehicle Rescheduling Problem (VRSP). Masters Thesis, HSR Hochschule für Technik Rapperswil.


## Notes

- The basic content of this repository is adapted from [Flatland Baselines](https://gitlab.aicrowd.com/flatland/neurips2020-flatland-baselines)
- The baselines are under the MIT license

