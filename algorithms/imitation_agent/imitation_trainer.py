import numpy as np

from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.agents.dqn import ApexTrainer,DQNTrainer
from ray.rllib.utils.annotations import override

from ray.rllib.agents.ppo.ppo import PPOTrainer
import ray
from ray import tune
from ray.tune.trainable import Trainable

import numpy as np
import os
import math

import ray
import yaml
from pathlib import Path
from ray.cluster_utils import Cluster
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune import run_experiments
from ray.tune.logger import TBXLogger
from ray.tune.resources import resources_to_json
from ray.tune.tune import _make_scheduler

from ray.rllib.models.tf.tf_action_dist import Categorical
tf = try_import_tf()

from ray.tune import registry
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.optimizers import PolicyOptimizer, SyncSamplesOptimizer
from ray.rllib.models import ModelCatalog

from utils.argparser import create_parser
from utils.loader import load_envs, load_models, load_algorithms
from envs.flatland import get_eval_config
from ray.rllib.utils import merge_dicts
from ray.rllib.evaluation.metrics import collect_metrics
# Custom wandb logger with hotfix to allow custom callbacks
from wandblogger import WandbLogger
import pandas as pd

"""
Note : This implementation has been adapted from : 
    https://github.com/ray-project/ray/blob/master/rllib/contrib/random_agent/random_agent.py
"""
from ray.rllib.policy import Policy,TFPolicy
from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.policy.tf_policy_template import build_tf_policy

from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.execution.metric_ops import StandardMetricsReporting

import numpy as np
import logging
logger = logging.getLogger(__name__)

from flatland.envs.agent_utils import RailAgentStatus

import sys,os
# sys.path.insert(0, os.getcwd() + '/envs/expert')
from libs.cell_graph_dispatcher import CellGraphDispatcher

def adam_optimizer(policy, config):
    return tf.train.AdamOptimizer(
        learning_rate=config.get('lr',5e-4), epsilon=config.get('adam_epsilon',1e-8))

def default_execution_plan(workers: WorkerSet, config):
    # Collects experiences in parallel from multiple RolloutWorker actors.
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # Combine experiences batches until we hit `train_batch_size` in size.
    # Then, train the policy on those experiences and update the workers.
    train_op = rollouts \
        .combine(ConcatBatches(
            min_batch_size=config["train_batch_size"])) \
        .for_each(TrainOneStep(workers))

    # Add on the standard episode reward, etc. metrics reporting. This returns
    # a LocalIterator[metrics_dict] representing metrics for each train step.
    return StandardMetricsReporting(train_op, workers, config)

def loss_imitation(policy, model, dist_class, train_batch):
    return np.random.randint(5)

ImitationTFPolicy = build_tf_policy(
    name="ImitationTFPolicy",
    loss_fn=loss_imitation,
    optimizer_fn=adam_optimizer,
)

class ImitationMetrics(PolicyOptimizer):
    """Adding metrics."""

    @override(PolicyOptimizer)
    def step(self):
        pass


class ImitationAgent(PPOTrainer):
    """Policy that takes random actions and never learns."""

    _name = "ImitationAgent"

    @override(Trainer)
    def _init(self, config, env_creator):
        self.env = env_creator(config["env_config"])
        self.state = {}
        self._policy = ImitationTFPolicy
        action_space = self.env.action_space
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"])
        self.workers = self._make_workers(
            env_creator, self._policy, config, self.config["num_workers"])
        self.execution_plan = default_execution_plan
        self.train_exec_impl = self.execution_plan(self.workers, config)
        self.optimizer = ImitationMetrics(self.workers)

    # @override(Trainer)
    # def collect_metrics(self, selected_workers=None):

    #     weights = ray.put(self.workers.local_worker().save())
    #     self.evaluation_workers.foreach_worker(
    #         lambda w: w.restore(ray.get(weights)))

    #     self.workers.foreach_worker(
    #         lambda w: w.restore(ray.get(weights)))
    #     ray.get([
    #         w.sample.remote()
    #         for w in self.workers.remote_workers()
    #     ])

    #     metrics = collect_metrics(self.workers.local_worker(),
    #                                   self.workers.remote_workers())

    #     res = self.optimizer.collect_metrics(
    #         self.config["collect_metrics_timeout"],
    #         min_history=self.config["metrics_smoothing_episodes"],
    #         selected_workers=selected_workers)
    #     return res
    
    
    @override(Trainer)
    def _train(self):
        import tensorflow as tf
        policy = self.get_policy()        
        steps = 0
        n_episodes = 1
        for _ in range(n_episodes):
            env = self.env._env.rail_env
            obs = self.env.reset()
            num_outputs = env.action_space[0]
            n_agents = env.get_num_agents()
            dispatcher = CellGraphDispatcher(env)

            # TODO : Update max_steps as per latest version
            # https://gitlab.aicrowd.com/flatland/flatland-examples/blob/master/reinforcement_learning/multi_agent_training.py
            # max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities))) - 1
            max_steps = int(4 * 2 * (20 + env.height + env.width))
            episode_steps = 0
            episode_max_steps = 0
            episode_num_agents = 0
            episode_score = 0
            episode_done_agents = 0
            done = {}
            done["__all__"] = False

            # TODO: Support for batch update
            # batch_size = 2
            # logits, _ = policy.model.forward({"obs": np.vstack([obs[a],obs[a]])}, [], None)

            for step in range(max_steps):
                action_dict = dispatcher.step(env._elapsed_steps)

                with tf.GradientTape() as tape:
                    imitation_loss = 0
                    active_agents = 0
                    for a in range(n_agents):
                        if not done.get(a) and obs.get(a) is not None:
                            active_agents += 1
                            expert_action = action_dict[a].value
                            input_dict = {"obs": np.expand_dims(obs[a],0)}
                            input_dict['obs_flat'] = input_dict['obs']
                            logits, _ = policy.model.forward(input_dict, [], None)
                            model_logits = tf.squeeze(logits)
                            expert_logits = tf.cast(expert_action, tf.int32)

                            action_dist = Categorical(logits, policy.model.model_config)

                            imitation_loss += tf.reduce_mean(-action_dist.logp(tf.expand_dims(expert_logits,0)))
                    imitation_loss = imitation_loss/max(active_agents,1)            

                gradients = tape.gradient(imitation_loss, policy.model.trainable_variables())
 
                self.workers.local_worker().apply_gradients(gradients)
                weights = ray.put(self.workers.local_worker().get_weights())
                # print(self.workers.local_worker().get_weights()['default_policy'][0][:4])
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)

                obs, all_rewards, done, info = self.env.step(action_dict)
                steps += 1

                for agent, agent_info in info.items():
                    if agent_info["agent_done"]:
                        episode_done_agents += 1
                
                if done["__all__"]:
                    for agent, agent_info in info.items():
                        if episode_max_steps == 0:
                            episode_max_steps = agent_info["max_episode_steps"]
                            episode_num_agents = agent_info["num_agents"]
                        episode_steps = max(episode_steps, agent_info["agent_step"])
                        episode_score += agent_info["agent_score"]
                    print(float(episode_done_agents) / episode_num_agents)
                    break

        norm_factor = 1.0 / (episode_max_steps * episode_num_agents)

        result = {
            "expert_episode_reward_mean": episode_score,
            "episode_reward_mean" : episode_score,
            "expert_episode_completion_mean": float(episode_done_agents) / episode_num_agents,
            "expert_episode_score_normalized": episode_score * norm_factor,
            "episodes_this_iter": n_episodes,
            "timesteps_this_iter": steps,
        }

        # Code taken from _train method of trainer_template.py - TODO: Not working
        # res = self.collect_metrics()
        # res = {}
        # res.update(
        #     optimizer_steps_this_iter=steps,
        #     episode_reward_mean=episode_score,
        #     info=res.get("info", {}))
        # res.update(expert_scores = result)

        return result
        

if __name__ == "__main__":

    # Copy this file to the root folder to run

    from train import on_episode_end
    exp = {}
    exp['run']= "ImitationAgent"
    exp['env']= "flatland_sparse"
    # exp['stop'] = {"timesteps_total": 15000}
    exp['stop'] = {"iterations": 4}
    exp['checkpoint_freq'] =  2
    # exp['checkpoint_at_end'] = True
    # exp['keep_checkpoints_num']= 100
    # exp['checkpoint_score_attr']: "episode_reward_mean"
    # exp['num_samples']= 3

    config = {
    "num_workers": 1,
    "num_envs_per_worker": 1,
    "num_gpus": 0,
    "clip_rewards": False,
    "vf_clip_param": 500.0,
    "entropy_coeff": 0.01,
    # effective batch_size: train_batch_size * num_agents_in_each_environment [5, 10]
    # see https://github.com/ray-project/ray/issues/4628
    "train_batch_size": 1000, # 5000
    "rollout_fragment_length": 50,  # 100
    "sgd_minibatch_size": 100,  # 500
    "vf_share_layers": False,
    "env_config" : {
        "observation": "tree",
        "observation_config":{
            "max_depth": 2,
            "shortest_path_max_depth": 30},
        "generator": "sparse_rail_generator",
        "generator_config": "small_v0",
        "eval_generator": "test"},
    "model" : {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [256, 256],
        "vf_share_layers": True }}


    exp['config'] = config
    exp['config']['callbacks'] = {
        'on_episode_end': on_episode_end,
    }

    eval_configs = get_eval_config(exp['config'].get('env_config',\
                    {}).get('eval_generator',"default"))
    eval_seed = eval_configs.get('evaluation_config',{}).get('env_config',{}).get('seed')

    # add evaluation config to the current config
    exp['config'] = merge_dicts(exp['config'],eval_configs)
    if exp['config'].get('evaluation_config'):
        exp['config']['evaluation_config']['env_config'] = exp['config'].get('env_config')
        eval_env_config = exp['config']['evaluation_config'].get('env_config')
        if eval_seed and eval_env_config:
            # We override the env seed from the evaluation config
            eval_env_config['seed'] = eval_seed

    exp["config"]["eager"] = True
    exp["config"]["use_pytorch"] = False
    exp["config"]["log_level"] = "INFO"
    verbose = 2
    exp["config"]["eager_tracing"] = True
    webui_host = "0.0.0.0"
    # TODO should be in exp['config'] directly
    exp['config']['env_config']['yaml_config'] = config
    exp['loggers'] = [TBXLogger]

    _default_config = with_common_config(
       exp["config"])


    ray.init(num_cpus=4,num_gpus=0)
    trainer = ImitationAgent(_default_config,
    env=exp['env'],)

    # trainer = PPOTrainer(_default_config,
    # env="flatland_sparse",)
    for i in range(exp.get("stop",{}).get("iterations",5)):
        result = trainer.train()
        print("Results:",result)
        if i % exp['checkpoint_freq']==0:
            # eval_results = trainer._evaluate()
            # print("Eval Results:",eval_results)
            checkpoint = trainer.save()
            # TODO: Loads weights but not optimizer state
            # Could be done by overriding _save by using model.save_weight(checkpoint)
            # Also override _restore. Ideally use workers to save/load weights.
            # trainer.restore(checkpoint)
            print("checkpoint saved at", checkpoint)

    trainer.stop()
    
    print("Test: OK")



