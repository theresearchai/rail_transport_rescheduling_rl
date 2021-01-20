"""
Registry of custom implemented algorithms names
Please refer to the following examples to add your custom algorithms : 
- AlphaZero : https://github.com/ray-project/ray/tree/master/rllib/contrib/alpha_zero
- bandits : https://github.com/ray-project/ray/tree/master/rllib/contrib/bandits
- maddpg : https://github.com/ray-project/ray/tree/master/rllib/contrib/maddpg
- random_agent: https://github.com/ray-project/ray/tree/master/rllib/contrib/random_agent
An example integration of the random agent is shown here : 
- https://github.com/AIcrowd/neurips2020-procgen-starter-kit/tree/master/algorithms/custom_random_agent
"""

def _import_imitation_trainer():
    from .imitation_agent.imitation_trainer import ImitationAgent
    return ImitationAgent

def _import_custom_trainer():
    from .custom_agent.custom_trainer import CustomAgent
    return CustomAgent

CUSTOM_ALGORITHMS = {
    "ImitationAgent": _import_imitation_trainer,
    "CustomAgent": _import_custom_trainer
}