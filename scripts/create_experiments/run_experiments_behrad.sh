#!/bin/bash
cd /home/toghi/Toghi_WS/RL/coop_repo/scripts/rl_agents_scripts/

common=""
common+="python3 experiments.py evaluate --agent configs/experiments/agents/DQNAgent/convnetatari.json"
common+=" --no-display --train --episodes 10000"
common+=" --video_save_freq 50 --model_save_freq 500"
common+=" --create_episode_log --create_timestep_log  --individual_episode_log_level 2"
common+=" --processes 12"
#common+=" --test "
#common+=" --episodes_test 1000"
#
##for experiments in 207
#for experiments in 3000
#do
#  exp="$common  --environment configs/experiments/IROS/exp_merge_IROS_$experiments.json";
#  echo $exp
#  xterm -maximized -e "$exp" &
#done



exp="$common  --environment configs/experiments/IROS/exp_test_behrad.json";
echo $exp
xterm -maximized -e "$exp" &


#for experiments in {1520..1522}
#do
#  exp="$common  --environment configs/experiments/IEEE_Access/exp_merge_$experiments.json";
#  echo $exp
#  xterm -maximized -e "$exp" &10000
#done
#
#for experiments in 101
#do
#  exp="$common  --environment configs/experiments/IROS/exp_merge_IROS101.json";
#  echo $exp
#  xterm -maximized -e "$exp" &
#done


