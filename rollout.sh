echo "===================="
echo "APEX TREE OBS"
echo "===================="

python rollout.py baselines/checkpoints/apex-tree-obs-small-v0-0/checkpoint_400/checkpoint-400 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/apex_tree_obs_small_v0-1/checkpoint_400/checkpoint-400 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/apex_tree_obs_small_v0-2/checkpoint_250/checkpoint-250 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

echo "===================="
echo "PPO TREE OBS"
echo "===================="

python rollout.py baselines/checkpoints/ppo-tree-obs-small-v0-0/checkpoint_1800/checkpoint-1800 --run PPO --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/ppo_tree_obs_small_v0-1/checkpoint_1200/checkpoint-1200 --run PPO --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/ppo_tree_obs_small_v0-2/checkpoint_1800/checkpoint-1800 --run PPO --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

echo "===================="
echo "MARWIL TREE OBS"
echo "===================="

python rollout.py baselines/checkpoints/marwil-tree-obs-small-v0-0/checkpoint_183874/checkpoint-183874 --run MARWIL --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/marwil-tree-obs-small-v0-1/checkpoint_183860/checkpoint-183860 --run MARWIL --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/marwil-tree-obs-small-v0-2/checkpoint_183882/checkpoint-183882 --run MARWIL --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

echo "===================="
echo "PPO SKIP TREE OBS"
echo "===================="

python rollout.py baselines/checkpoints/ppo_tree_obs_small_v0_skip-0/checkpoint_1800/checkpoint-1800 --run PPO --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/ppo_tree_obs_small_v0_skip-1/checkpoint_2650/checkpoint-2650 --run PPO --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/ppo_tree_obs_small_v0_skip-2/checkpoint_400/checkpoint-400 --run PPO --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

echo "===================="
echo "APEX SKIP TREE OBS"
echo "===================="

python rollout.py baselines/checkpoints/apex_tree_obs_small_v0_skip-0/checkpoint_450/checkpoint-450 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/apex_tree_obs_small_v0_skip-1/checkpoint_450/checkpoint-450 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/apex_tree_obs_small_v0_skip-2/checkpoint_300/checkpoint-300 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

echo "===================="
echo "APEX MIXED IL TREE OBS"
echo "===================="

python rollout.py baselines/checkpoints/apex-dqfd-25-tree-obs-small-v0-0/checkpoint_500/checkpoint-500 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/apex-dqfd-25-tree-obs-small-v0-1/checkpoint_1600/checkpoint-1600 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/apex-dqfd-25-tree-obs-small-v0-2/checkpoint_200/checkpoint-200 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

echo "===================="
echo "PURE ONLINE IL TREE OBS"
echo "===================="

python rollout.py baselines/checkpoints/pure_imitation_tree_obs-0/checkpoint_45300/checkpoint-45300 --run ImitationAgent --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "False"}}' --eager

python rollout.py baselines/checkpoints/pure_imitation_tree_obs-1/checkpoint_11500/checkpoint-11500 --run ImitationAgent --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "False"}}' --eager

python rollout.py baselines/checkpoints/pure_imitation_tree_obs-2/checkpoint_23350/checkpoint-23350 --run ImitationAgent --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "False"}}' --eager

echo "===================="
echo "PPO + ONLINE IL TREE OBS"
echo "===================="

python rollout.py baselines/checkpoints/ppo_imitation_tree_obs-0/checkpoint_5392/checkpoint-5392 --run ImitationAgent --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "False"}}' --eager

python rollout.py baselines/checkpoints/ppo_imitation_tree_obs-1/checkpoint_8630/checkpoint-8630 --run ImitationAgent --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "False"}}' --eager

python rollout.py baselines/checkpoints/ppo_imitation_tree_obs-2/checkpoint_8466/checkpoint-8466 --run ImitationAgent --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "False"}}' --eager

echo "===================="
echo "CPPO"
echo "===================="

python rollout.py baselines/checkpoints/ccppo-tree-obs-0/checkpoint_6084/checkpoint-6084 --run CcTransformer --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"sparse_reward":"True","done_reward":1, "not_finished_reward": -1, "seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"custom_model": "cc_transformer","fcnet_activation": "relu", "fcnet_hiddens":[512,512,512],"vf_share_layers": "True" , "custom_options": {"max_num_agents": 15,"actor":{"activation_fn": "relu","hidden_layers": [512,512,512]},"critic":{"centralized": "True", "embedding_size": 32, "num_heads": 4, "d_model": 32, "use_scale": "True", "activation_fn": "relu","hidden_layers": [512,512,512]},"embedding":{"activation_fn": "relu","hidden_layers": [512,512,512]}}}}'

python rollout.py baselines/checkpoints/ccppo-tree-obs-1/checkpoint_5740/checkpoint-5740 --run CcTransformer --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"sparse_reward":"True","done_reward":1, "not_finished_reward": -1, "seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"custom_model": "cc_transformer","fcnet_activation": "relu", "fcnet_hiddens":[512,512,512],"vf_share_layers": "True" , "custom_options": {"max_num_agents": 15,"actor":{"activation_fn": "relu","hidden_layers": [512,512,512]},"critic":{"centralized": "True", "embedding_size": 32, "num_heads": 4, "d_model": 32, "use_scale": "True", "activation_fn": "relu","hidden_layers": [512,512,512]},"embedding":{"activation_fn": "relu","hidden_layers": [512,512,512]}}}}'

python rollout.py baselines/checkpoints/ccppo-tree-obs-2/checkpoint_5820/checkpoint-5820 --run CcTransformer --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"sparse_reward":"True","done_reward":1, "not_finished_reward": -1, "seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"custom_model": "cc_transformer","fcnet_activation": "relu", "fcnet_hiddens":[512,512,512],"vf_share_layers": "True" , "custom_options": {"max_num_agents": 15,"actor":{"activation_fn": "relu","hidden_layers": [512,512,512]},"critic":{"centralized": "True", "embedding_size": 32, "num_heads": 4, "d_model": 32, "use_scale": "True", "activation_fn": "relu","hidden_layers": [512,512,512]},"embedding":{"activation_fn": "relu","hidden_layers": [512,512,512]}}}}'

echo "===================="
echo "CPPO Base"
echo "===================="

python rollout.py baselines/checkpoints/ccppo-transformer-tree-obs-0/checkpoint_5933/checkpoint-5933 --run CcConcatenate --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"custom_model": "cc_concatenate","fcnet_activation": "relu", "fcnet_hiddens":[512,512,512],"vf_share_layers": "True" , "custom_options": {"max_num_agents": 15,"actor":{"activation_fn": "relu","hidden_layers": [512,512,512]},"critic":{"centralized": "True", "embedding_size": 32, "num_heads": 4, "d_model": 32, "use_scale": "True", "activation_fn": "relu","hidden_layers": [512,512,512]},"embedding":{"activation_fn": "relu","hidden_layers": [512,512,512]}}}}'

python rollout.py baselines/checkpoints/ccppo-transformer-tree-obs-1/checkpoint_5914/checkpoint-5914 --run CcConcatenate --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"custom_model": "cc_concatenate","fcnet_activation": "relu", "fcnet_hiddens":[512,512,512],"vf_share_layers": "True" , "custom_options": {"max_num_agents": 15,"actor":{"activation_fn": "relu","hidden_layers": [512,512,512]},"critic":{"centralized": "True", "embedding_size": 32, "num_heads": 4, "d_model": 32, "use_scale": "True", "activation_fn": "relu","hidden_layers": [512,512,512]},"embedding":{"activation_fn": "relu","hidden_layers": [512,512,512]}}}}'

python rollout.py baselines/checkpoints/ccppo-transformer-tree-obs-2/checkpoint_5847/checkpoint-5847 --run CcConcatenate --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"custom_model": "cc_concatenate","fcnet_activation": "relu", "fcnet_hiddens":[512,512,512],"vf_share_layers": "True" , "custom_options": {"max_num_agents": 15,"actor":{"activation_fn": "relu","hidden_layers": [512,512,512]},"critic":{"centralized": "True", "embedding_size": 32, "num_heads": 4, "d_model": 32, "use_scale": "True", "activation_fn": "relu","hidden_layers": [512,512,512]},"embedding":{"activation_fn": "relu","hidden_layers": [512,512,512]}}}}'

echo "===================="
echo "PPO MASK TREE OBS"
echo "===================="

python rollout.py baselines/checkpoints/ppo-tree-obs-small-v0-mask-0/checkpoint_650/checkpoint-650 --run PPO --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"available_actions_obs":"True","allow_noop":"False","seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"custom_model": "fully_connected_model", "custom_options": {"layers": [256, 256],"activation":"relu","layer_norm":"False", "mask_unavailable_actions":"True"}}}'

python rollout.py baselines/checkpoints/ppo-tree-obs-small-v0-mask-1/checkpoint_300/checkpoint-300 --run PPO --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"available_actions_obs":"True","allow_noop":"False","seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"custom_model": "fully_connected_model", "custom_options": {"layers": [256, 256],"activation":"relu","layer_norm":"False", "mask_unavailable_actions":"True"}}}'

python rollout.py baselines/checkpoints/ppo-tree-obs-small-v0-mask-2/checkpoint_2500/checkpoint-2500 --run PPO --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"available_actions_obs":"True","allow_noop":"False","seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"custom_model": "fully_connected_model", "custom_options": {"layers": [256, 256],"activation":"relu","layer_norm":"False", "mask_unavailable_actions":"True"}}}'

echo "===================="
echo "APEX Global Density OBS"
echo "===================="

python rollout.py baselines/checkpoints/apex-global-density-obs-small-v0-0/checkpoint_200/checkpoint-200 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"hiddens":[],"dueling":"False","env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "density", "observation_config": {"width": 25, "height": 25,"max_t": 1000, "encoding": "exp_decay"}}, "model": {"custom_model": "global_dens_obs_model","custom_options": {"architecture": "impala","architecture_options":{"residual_layers":[[16, 2], [32, 4]]}}}}'

python rollout.py baselines/checkpoints/apex-global-density-obs-small-v0-1/checkpoint_450/checkpoint-450 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"hiddens":[],"dueling":"False","env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "density", "observation_config": {"width": 25, "height": 25,"max_t": 1000, "encoding": "exp_decay"}}, "model": {"custom_model": "global_dens_obs_model","custom_options": {"architecture": "impala","architecture_options":{"residual_layers":[[16, 2], [32, 4]]}}}}'

python rollout.py baselines/checkpoints/apex-global-density-obs-small-v0-2/checkpoint_400/checkpoint-400 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"hiddens":[],"dueling":"False","env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "density", "observation_config": {"width": 25, "height": 25,"max_t": 1000, "encoding": "exp_decay"}}, "model": {"custom_model": "global_dens_obs_model","custom_options": {"architecture": "impala","architecture_options":{"residual_layers":[[16, 2], [32, 4]]}}}}'

echo "===================="
echo "APEX PURE IL TREE OBS"
echo "===================="

python rollout.py baselines/checkpoints/apex_pure_il-tree-obs-0/checkpoint_50/checkpoint-50 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/apex_pure_il-tree-obs-1/checkpoint_50/checkpoint-50 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

python rollout.py baselines/checkpoints/apex_pure_il-tree-obs-2/checkpoint_100/checkpoint-100 --run APEX --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "tree", "observation_config": {"max_depth": 2, "shortest_path_max_depth": 30}}, "model": {"fcnet_activation": "relu", "fcnet_hiddens": [256, 256], "vf_share_layers": "True"}}'

echo "===================="
echo "RANDOM AGENT"
echo "===================="

python rollout.py baselines/checkpoints/random/checkpoint/ --run CustomAgent --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "random_action"}}'

echo "===================="
echo "SHORTEST PATH AGENT"
echo "===================="

python rollout.py baselines/checkpoints/shortest_path/checkpoint/ --run CustomAgent --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "shortest_path_action"}}'

echo "===================="
echo "FORWARD PATH AGENT"
echo "===================="

python rollout.py baselines/checkpoints/forward_path/checkpoint/ --run CustomAgent --no-render --episodes 50 --env 'flatland_sparse' --config '{"env_config": {"seed":1000000000,"generator": "sparse_rail_generator", "generator_config": "small_v0", "observation": "forward_action"}}'
