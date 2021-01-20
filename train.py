#!/usr/bin/env python

import os
import numpy as np

import ray
import yaml
from pathlib import Path
from ray.cluster_utils import Cluster
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune import run_experiments, Experiment
from ray.tune.logger import TBXLogger
from ray.tune.resources import resources_to_json
from ray.tune.tune import _make_scheduler

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.ppo.ppo import PPOTrainer

from algorithms.imitation_agent.imitation_trainer import ImitationAgent

from utils.argparser import create_parser
from utils.loader import load_envs, load_models, load_algorithms

from envs.flatland import get_eval_config
from ray.rllib.utils import merge_dicts

# Custom wandb logger with hotfix to allow custom callbacks
from wandblogger import WandbLogger

# Try to import both backends for flag checking/warnings.
tf = try_import_tf()
torch, _ = try_import_torch()

# Register all necessary assets in tune registries
load_envs(os.getcwd())  # Load envs
load_models(os.getcwd())  # Load models
from algorithms import CUSTOM_ALGORITHMS
load_algorithms(CUSTOM_ALGORITHMS)  # Load algorithms

MAX_ITERATIONS = 1000000

def on_episode_end(info):
    episode = info["episode"]  # type: MultiAgentEpisode

    episode_steps = 0
    episode_max_steps = 0
    episode_num_agents = 0
    episode_score = 0
    episode_done_agents = 0
    episode_num_swaps = 0

    for agent, agent_info in episode._agent_to_last_info.items():
        if episode_max_steps == 0:
            episode_max_steps = agent_info["max_episode_steps"]
            episode_num_agents = agent_info["num_agents"]
        episode_steps = max(episode_steps, agent_info["agent_step"])
        episode_score += agent_info["agent_score"]
        if "num_swaps" in agent_info:
            episode_num_swaps += agent_info["num_swaps"]
        if agent_info["agent_done"]:
            episode_done_agents += 1

    # Not a valid check when considering a single policy for multiple agents
    #assert len(episode._agent_to_last_info) == episode_num_agents

    norm_factor = 1.0 / (episode_max_steps * episode_num_agents)
    percentage_complete = float(episode_done_agents) / episode_num_agents

    episode.custom_metrics["episode_steps"] = episode_steps
    episode.custom_metrics["episode_max_steps"] = episode_max_steps
    episode.custom_metrics["episode_num_agents"] = episode_num_agents
    episode.custom_metrics["episode_return"] = episode.total_reward
    episode.custom_metrics["episode_score"] = episode_score
    episode.custom_metrics["episode_score_normalized"] = episode_score * norm_factor
    episode.custom_metrics["episode_num_swaps"] = episode_num_swaps / 2
    episode.custom_metrics["percentage_complete"] = percentage_complete


def imitation_ppo_train_fn(config,reporter=None):
    imitation_trainer = ImitationAgent(config,
    env=config.get("env"),)

    ppo_trainer = PPOTrainer(config,
    env=config.get("env"),)

    expert_ratio = config.get("env_config",{}).get("expert",{}).get('ratio', 0.5)
    expert_min_ratio = config.get("env_config",{}).get("expert",{}).get('min_ratio', expert_ratio)
    expert_ratio_decay = config.get("env_config",{}).get("expert",{}).get('ratio_decay', 1)

    for i in range(MAX_ITERATIONS):

        print("== Iteration", i, "==")

        trainer_type = np.random.binomial(size=1, n=1, p= expert_ratio)[0]

        if trainer_type:
            # improve the Imitation policy
            print("-- Imitation --")
            result_imitate = imitation_trainer.train()
            if reporter:
                reporter(**result_imitate)
            if i % checkpoint_freq == 0:
                checkpoint = imitation_trainer.save()
                print("checkpoint saved at", checkpoint)

            ppo_trainer.set_weights(imitation_trainer.get_weights())

        else:
            # improve the PPO policy
            print("-- PPO --")
            result_ppo = ppo_trainer.train()
            if reporter:
                reporter(**result_ppo)
            if i % checkpoint_freq == 0:
                checkpoint = ppo_trainer.save()
                print("checkpoint saved at", checkpoint)

        expert_ratio = max(expert_min_ratio, expert_ratio_decay * expert_ratio)

    imitation_trainer.stop()
    ppo_trainer.stop()

    print("Completed: OK")


def run(args, parser):
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)
    else:
        # Note: keep this in sync with tune/config_parser.py
        experiments = {
            args.experiment_name: {  # i.e. log to ~/ray_results/default
                "run": args.run,
                "checkpoint_freq": args.checkpoint_freq,
                "keep_checkpoints_num": args.keep_checkpoints_num,
                "checkpoint_score_attr": args.checkpoint_score_attr,
                "local_dir": args.local_dir,
                "resources_per_trial": (
                        args.resources_per_trial and
                        resources_to_json(args.resources_per_trial)),
                "stop": args.stop,
                "config": dict(args.config, env=args.env),
                "restore": args.restore,
                "num_samples": args.num_samples,
                "upload_dir": args.upload_dir,
            }
        }

    verbose = 1
    custom_fn = False
    webui_host = "localhost"
    for exp in experiments.values():
        # Bazel makes it hard to find files specified in `args` (and `data`).
        # Look for them here.
        # NOTE: Some of our yaml files don't have a `config` section.
        if exp.get("config", {}).get("input"):
            if not isinstance(exp.get("config", {}).get("input"),dict):
                if not os.path.exists(exp["config"]["input"]):
                    # This script runs in the ray/rllib dir.
                    rllib_dir = Path(__file__).parent
                    input_file = rllib_dir.absolute().joinpath(exp["config"]["input"])
                    exp["config"]["input"] = str(input_file)

        if not exp.get("run"):
            parser.error("the following arguments are required: --run")
        if not exp.get("env") and not exp.get("config", {}).get("env"):
            parser.error("the following arguments are required: --env")
        if args.eager:
            exp["config"]["eager"] = True
        if args.torch:
            exp["config"]["use_pytorch"] = True
        if args.v:
            exp["config"]["log_level"] = "INFO"
            verbose = 2
        if args.vv:
            exp["config"]["log_level"] = "DEBUG"
            verbose = 3
        if args.trace:
            if not exp["config"].get("eager"):
                raise ValueError("Must enable --eager to enable tracing.")
            exp["config"]["eager_tracing"] = True
        if args.bind_all:
            webui_host = "0.0.0.0"
        if args.log_flatland_stats:
            exp['config']['callbacks'] = {
                'on_episode_end': on_episode_end,
            }

        if args.eval:
            eval_configs_file = exp['config'].get('env_config',\
                                    {}).get('eval_generator',"default")
            if args.record:
                eval_configs_file = exp['config'].get('env_config',\
                        {}).get('eval_generator',"default_render")
            eval_configs = get_eval_config(eval_configs_file)
            eval_seed = eval_configs.get('evaluation_config',{}).get('env_config',{}).get('seed')
            eval_render = eval_configs.get('evaluation_config',{}).get('env_config',{}).get('render')

            # add evaluation config to the current config
            exp['config'] = merge_dicts(exp['config'],eval_configs)
            if exp['config'].get('evaluation_config'):
                exp['config']['evaluation_config']['env_config'] = exp['config'].get('env_config')
                eval_env_config = exp['config']['evaluation_config'].get('env_config')
                if eval_seed and eval_env_config:
                    # We override the env seed from the evaluation config
                    eval_env_config['seed'] = eval_seed
                if eval_render and eval_env_config:
                    # We override the env render from the evaluation config
                    eval_env_config['render'] = eval_render
                    # Set video_dir if it exists
                    eval_render_dir = eval_configs.get('evaluation_config',{}).get('env_config',{}).get('video_dir')
                    if eval_render_dir:
                        eval_env_config['video_dir'] = eval_render_dir
                # Remove any wandb related configs
                if eval_env_config:
                    if eval_env_config.get('wandb'):
                        del eval_env_config['wandb']

            # Remove any wandb related configs
            if exp['config']['evaluation_config'].get('wandb'):
                del exp['config']['evaluation_config']['wandb']
        if args.custom_fn:
            custom_fn = globals()[exp['config'].get("env_config",{}).get("custom_fn","imitation_ppo_train_fn")]
        if args.save_checkpoint:
            exp['config']['env_config']['save_checkpoint'] = True
        if args.config_file:
            # TODO should be in exp['config'] directly
            exp['config']['env_config']['yaml_config'] = args.config_file
        exp['loggers'] = [WandbLogger, TBXLogger]

        global checkpoint_freq,keep_checkpoints_num,checkpoint_score_attr,checkpoint_at_end
        checkpoint_freq = exp['checkpoint_freq']

        # TODO: Below checkpoints paramaters are not supported for default custom_fn
        keep_checkpoints_num = exp['keep_checkpoints_num']
        checkpoint_score_attr = exp['checkpoint_score_attr']
        checkpoint_at_end = exp['checkpoint_at_end']

    if args.ray_num_nodes:
        cluster = Cluster()
        for _ in range(args.ray_num_nodes):
            cluster.add_node(
                num_cpus=args.ray_num_cpus or 1,
                num_gpus=args.ray_num_gpus or 0,
                object_store_memory=args.ray_object_store_memory,
                memory=args.ray_memory,
                redis_max_memory=args.ray_redis_max_memory)
        ray.init(address=cluster.address)
    else:
        ray.init(
            address=args.ray_address,
            object_store_memory=args.ray_object_store_memory,
            memory=args.ray_memory,
            redis_max_memory=args.ray_redis_max_memory,
            num_cpus=args.ray_num_cpus,
            num_gpus=args.ray_num_gpus,
            webui_host=webui_host)

    if custom_fn:
        for exp in experiments.values():
            configs = with_common_config(exp["config"])
            configs['env'] = exp.get('env')
            resources = PPOTrainer.default_resource_request(configs).to_json()
            experiment_spec = Experiment(
                                custom_fn.__name__,
                                custom_fn,
                                resources_per_trial=resources,
                                config=configs,
                                stop=exp.get('stop'),
                                num_samples=exp.get('num_samples',1),
                                loggers=exp.get('loggers'),
                                restore=None)
        experiments = experiment_spec

    run_experiments(
        experiments,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials,
        resume=args.resume,
        verbose=verbose,
        concurrent=True)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
