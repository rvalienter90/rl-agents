# Approximate Robust Control of Uncertain Dynamical Systems
# https://eleurent.github.io/robust-control/

benchmark
configs/RoundaboutEnv/benchmark_robust_control.json
--test --episodes=100 --processes=4


# Social Attention for Autonomous Decision-Making in Dense Traffic
## MLP/List
evaluate 
configs/IntersectionEnv/env.json 
configs/IntersectionEnv/agents/DQNAgent/baseline.json 
--train --episodes=4000 --name-from-config

## CNN/Grid
evaluate 
configs/IntersectionEnv/env_grid_dense.json 
configs/IntersectionEnv/agents/DQNAgent/grid_convnet.json 
--train --episodes=4000 --name-from-config

## Ego-Attention
evaluate 
configs/IntersectionEnv/env.json 
configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json 
--train --episodes=4000 --name-from-config


evaluate
--environment
configs/IntersectionEnv/env.json 
--agent
configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json
--no-display
--train
--episodes
4000
--test
--episodes_test
400
--video_save_freq
100
--model_save_freq
1000
--name-from-envconfig
--create_episode_log
--individual_episode_log_level
2

## Visualize the results
tensorboard --logdir out/IntersectionEnv
or python analyze.py run out/IntersectionEnv/DQNAgent/


## Highway
evaluate 
configs/HighwayEnv/env.json 
configs/HighwayEnv/agents/DQNAgent/dqn.json 
--train --episodes=4000 --name-from-config

evaluate
--environment
configs/HighwayEnv/env.json
--agent
configs/HighwayEnv/agents/DQNAgent/ddqn.json
--no-display
--train
--episodes
4000
--test
--episodes_test
400
--video_save_freq
100
--model_save_freq
1000
--name-from-envconfig
--create_episode_log
--individual_episode_log_level
2


# Multi env
evaluate
--no-display
--train
--episodes
1000
--test
--episodes_test
50
--video_save_freq
1
--model_save_freq
500
--create_episode_log
--individual_episode_log_level
2
--environment
configs/experiments/complex/exp_merge_complex_grid_ma.json
--name-from-envconfig


configs/experiments/complex/exp_merge_complex_300.json



#test
evaluate
--agent
configs/HighwayEnv/agents/DQNAgent/dqn.json 
--no-display
--test
--episodes_test
400
--video_save_freq
20
--model_save_freq
500
--create_episode_log
--individual_episode_log_level
2
--environment
configs/HighwayEnv/env.json 
--recover-from
out/models/checkpoint-2999_dqn.tar