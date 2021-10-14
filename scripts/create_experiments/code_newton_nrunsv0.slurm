#!/bin/bash                                   
#SBATCH --nodes=1				#Get one node
#SBATCH --cpus-per-task=8			#8 cores per task
#SBATCH --ntasks=1				#But only one task
#SBATCH --gres=gpu:1				#And one GPU
#SBATCH --gres-flags=enforce-binding		#Insist on good CPU/GPU alignment
#SBATCH --time=90:00:00				#Run for 10 minutes, at most
#SBATCH --job-name=IEEE_sim			#Name the job so I can see it in squeue
#SBATCH --output=out/IEEE_simrun-train-slurm-%J.out
#SBATCH --mem=65536 

#Give this process 1 task (per GPU, but only one GPU), then
#assign eight 8per task (so 8 cores overall). Then enforce
#that slurm assigns only CPUs that are on the socket closest
#to the GPU you get
# Other comands
#--mem-per-cpu=16384
#--mem-per-cpu=65536

# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=1
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the TensorFlow module
# module load tensorflow/tensorflow-1.6.0
# List the modules that are loaded
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m
nvidia-smi

echo

# Activate the GPU version of TensorFlow
#source activate tensorflow-gpu

# Run Code
source /home/rvalienteromero/Coop/envs/coop_venv/bin/activate
cd /home/rvalienteromero/Coop/coop_repo/
export PATH=$PATH:/home/rvalienteromero/Coop/coop_repo/ffmpeg-git-20210110-amd64-static/

echo
common=""
common+="python3 experiments.py evaluate 
common+=" --no-display --train --episodes 10000"
common+=" --video_save_freq 500 --model_save_freq 500"
common+=" --create_episode_log  --individual_episode_log_level 2"
common+=" --test "
common+=" --episodes_test 1000"

time $common --environment configs/experiments/Behavior/exp_behavior_MLP.json 
echo

# Done!
echo "Ending script ..."
date