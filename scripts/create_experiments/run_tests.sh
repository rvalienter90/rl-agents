#!/bin/bash
##export PYTHONPATH="${PYTHONPATH}:/home/cavrel/PycharmProjects/coop_repo"
##source /home/cavrel/PycharmProjects/venv/coop_repo/bin/activate
cd /home/toghi/Toghi_WS/RL/coop_repo/scripts/rl_agents_scripts/
#
## shellcheck disable=SC1068
##common="source /home/cavrel/PycharmProjects/venv/coop_repo/bin/activate; "
#common=""
#common+="python3 experiments.py evaluate --agent configs/experiments/agents/DQNAgent/dqn.json  "
#common+=" --no-display --test --episodes 500"
#common+=" --video_save_freq 100 --model_save_freq 500"
#common+=" --create_episode_log  --individual_episode_log_level 2"
#common+=" --processes 12"
#
##for experiments in 207
#for scenario in "run_20201210-122132_1372_exp_merge_02"
#do
#  readarray -d _ -t strarr <<< "$scenario"
#
#
#  experiments="${strarr[${#strarr[*]} -1]}"
#  experiments=${experiments%?};
##  echo $experiments
#  exp="$common  --environment configs/experiments/exp_merge_$experiments.json   --recover-from  out/HighwayEnv/DQNAgent/$scenario/checkpoint-final.tar"
#
#  echo $exp
#  xterm -e "$exp" &
#done
#

#python3 experiments.py evaluate --agent configs/experiments/agents/DQNAgent/dqn.json --no-display --test --episodes_test 2000 --train --episodes 1 --video_save_freq 500 --model_save_freq 500 --create_episode_log  --individual_episode_log_level 2 --processes 12 --environment configs/experiments/IEEE_Access/exp_merge_300.json   --recover-from  out/HighwayEnv/IEEE_Access/300s/train/run_20210121-180816_19528_exp_merge_300/checkpoint-final.tar
#python3 experiments.py evaluate --agent configs/experiments/agents/DQNAgent/dqn.json --no-display --test --episodes_test 2000 --train --episodes 1  --video_save_freq 500 --model_save_freq 500 --create_episode_log  --individual_episode_log_level 2 --processes 12 --environment configs/experiments/IEEE_Access/exp_merge_301.json   --recover-from  out/HighwayEnv/IEEE_Access/300s/train/run_20210121-180832_14628_exp_merge_301/checkpoint-final.tar
#python3 experiments.py evaluate --agent configs/experiments/agents/DQNAgent/dqn.json --no-display --test --episodes_test 2000 --train --episodes 1  --video_save_freq 500 --model_save_freq 500 --create_episode_log  --individual_episode_log_level 2 --processes 12 --environment configs/experiments/IEEE_Access/exp_merge_302.json   --recover-from  out/HighwayEnv/IEEE_Access/300s/train/run_20210121-180846_7540_exp_merge_302/checkpoint-final.tar
#python3 experiments.py evaluate --agent configs/experiments/agents/DQNAgent/dqn.json --no-display --test --episodes_test 2000 --train --episodes 1  --video_save_freq 500 --model_save_freq 500 --create_episode_log  --individual_episode_log_level 2 --processes 12 --environment configs/experiments/IEEE_Access/exp_merge_303.json   --recover-from  out/HighwayEnv/IEEE_Access/300s/train/run_20210121-180928_2492_exp_merge_303/checkpoint-final.tar
#python3 experiments.py evaluate --agent configs/experiments/agents/DQNAgent/dqn.json --no-display --test --episodes_test 2000 --train --episodes 1  --video_save_freq 500 --model_save_freq 500 --create_episode_log  --individual_episode_log_level 2 --processes 12 --environment configs/experiments/IEEE_Access/exp_merge_304.json   --recover-from  out/HighwayEnv/IEEE_Access/300s/train/run_20210121-164729_32486_exp_merge_304/checkpoint-final.tar
#python3 experiments.py evaluate --agent configs/experiments/agents/DQNAgent/dqn.json --no-display --test --episodes_test 2000 --train --episodes 1  --video_save_freq 500 --model_save_freq 500 --create_episode_log  --individual_episode_log_level 2 --processes 12 --environment configs/experiments/IEEE_Access/exp_merge_305.json   --recover-from  out/HighwayEnv/IEEE_Access/300s/train/run_20210121-164729_32471_exp_merge_305/checkpoint-final.tar
#python3 experiments.py evaluate --agent configs/experiments/agents/DQNAgent/dqn.json --no-display --test --episodes_test 2000 --train --episodes 1  --video_save_freq 500 --model_save_freq 500 --create_episode_log  --individual_episode_log_level 2 --processes 12 --environment configs/experiments/IEEE_Access/exp_merge_306.json   --recover-from  out/HighwayEnv/IEEE_Access/300s/train/run_20210121-164729_32479_exp_merge_306/checkpoint-final.tar
python3 experiments.py evaluate --agent configs/experiments/agents/DQNAgent/dqn.json --no-display --test --episodes_test 2000 --train --episodes 1  --video_save_freq 500 --model_save_freq 500 --create_episode_log  --individual_episode_log_level 2 --processes 12 --environment configs/experiments/IEEE_Access/exp_merge_307.json   --recover-from  out/HighwayEnv/IEEE_Access/300s/train/run_20210121-164729_32472_exp_merge_307/checkpoint-final.tar
