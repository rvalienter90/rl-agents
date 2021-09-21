#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/home/cavrel/PycharmProjects/coop_repo"
#cd ./scripts/rl_agents_scripts/out/HighwayEnv/DQNAgent/run_20200913-201728_23011/checkpoint-final.tar

#tensorboard --logdir ./scripts/rl_agents_scripts/out/HighwayEnv/Data/500s

#win
#D:\Rodolfo\Data\IEEE_Access\1400series\1400s\train
tensorboard --logdir ./scripts/rl_agents_scripts/out/HighwayEnv/DQNAgent/
#tensorboard --logdir ./scripts/rl_agents_scripts/out/HighwayEnv/IEEE_Access/320s
#cd ./scripts/rl_agents_scripts

#python3 analyze.py run out/HighwayEnv/DQNAgent/