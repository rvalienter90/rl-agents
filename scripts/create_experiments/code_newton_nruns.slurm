#!/bin/bash                                   
#SBATCH --nodes=1				#Get one node
#SBATCH --cpus-per-task=6			#8 cores per task
#SBATCH --ntasks=1				#But only one task
#SBATCH --gres=gpu:1				#And one GPU
#SBATCH --gres-flags=enforce-binding		#Insist on good CPU/GPU alignment
#SBATCH --time=40:00:00			#Run for 10 minutes, at most
#SBATCH --job-name=IEEE_sim			#Name the job so I can see it in squeue
#SBATCH --output=out/IEEE_simrun-train-slurm-%J.out
#SBATCH --mem=65536 

#Give this process 1 task (per GPU, but only one GPU), then
#assign eight 8per task (so 8 cores overall). Then enforce
#that slurm assigns only CPUs that are on the socket closest
#to the GPU you get
# Other comands
#--mem-per-cpu=16384
#--mem-per-cpu=65536 32768

# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=1
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the TensorFlow module
# module load tensorflow/tensorflow-1.6.0
module load ffmpeg
# List the modules that are loaded
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m
nvidia-smi

echo

# Activate the GPU version of TensorFlow
#source activate tensorflow-gpu

# Run Code
# common+="python3 experiments.py evaluate --agent ./configs/experiments/agents/DQNAgent/convolutionalbaseline.json"

source /home/rvalienteromero/Coop/envs/coop_venv/bin/activate
# cd /home/rvalienteromero/Coop/coop_repo/
cd /home/rvalienteromero/Coop/coop_repo_multi/rl-agents/scripts
echo

common=""
common+="python3 experiments.py evaluate" 


# *******  train ******* 
common+=" --no-display --train --episodes 10000"
common+=" --video_save_freq 200 --model_save_freq 4000"
common+=" --create_episode_log  --individual_episode_log_level 2"
common+=" --test "
common+=" --episodes_test 1000"
common+=" --output_folder out"
common+=" --offscreen_rendering"

# *******  train ******* 
time $common --environment configs/experiments/Multienv/exp_generalization_100.json  &  $common --environment configs/experiments/Multienv/exp_generalization_101.json &  $common --environment configs/experiments/Multienv/exp_generalization_102.json 
echo


# ******* test ******* 
# common+=" --no-display" 
#common+=" --recover-from models/exit_agressive.tar"
#common+=" --video_save_freq 100 --model_save_freq 2000"
#common+=" --create_episode_log  --individual_episode_log_level 2"
#common+=" --test "
#common+=" --episodes_test 1000"
#common+=" --output_folder outagg"
#common+=" --offscreen_rendering"

# *******  test ******* 
# time $common --environment configs/experiments/IROS/exp_merge_IROS_9120.json --recover-from models/checkpoint-final_IROS9100.tar & $common --environment configs/experiments/IROS/exp_merge_IROS_9121.json --recover-from models/checkpoint-final_IROS9103.tar & $common --environment configs/experiments/IROS/exp_merge_IROS_9122.json --recover-from models/checkpoint-final_IROS9106.tar

# Done!
echo "Ending script ..."
date