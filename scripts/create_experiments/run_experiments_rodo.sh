#!/bin/bash
#export PYTHONPATH="${PYTHONPATH}:/home/cavrel/PycharmProjects/coop_repo"
#source /home/cavrel/PycharmProjects/venv/coop_repo/bin/activate
cd /home/cavrel/PycharmProjects/coop_repo/scripts/rl_agents_scripts/
#cd C:\Users\rvali\PycharmProjects\coop_repo
# shellcheck disable=SC1068
#common="source /home/cavrel/PycharmProjects/venv/coop_repo/bin/activate; "

common=""
common+="python3 experiments.py evaluate --agent configs/experiments/agents/DQNAgent/dqn.json"
common+=" --no-display --train --episodes 4000"
common+=" --video_save_freq 500 --model_save_freq 500"
common+=" --create_episode_log  --individual_episode_log_level 2"
common+=" --processes 12"
common+=" --test "
common+=" --episodes_test 1000"

#for experiments in 207
for experiments in {1100..1103}
do
  exp="$common  --environment configs/experiments/IEEE_Access/exp_merge_$experiments.json";
  echo $exp
  xterm -e "$exp" &
done
