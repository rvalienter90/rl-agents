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

## Visualize the results
tensorboard --logdir out/IntersectionEnv
or python analyze.py run out/IntersectionEnv/DQNAgent/