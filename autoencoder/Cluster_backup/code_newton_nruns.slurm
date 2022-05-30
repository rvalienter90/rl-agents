#!/bin/bash                                   
#SBATCH --nodes=1				#Get one node
#SBATCH --cpus-per-task=6			#8 cores per task
#SBATCH --ntasks=1				#But only one task
#SBATCH --gres=gpu:1				#And one GPU
#SBATCH --gres-flags=enforce-binding		#Insist on good CPU/GPU alignment
#SBATCH --time=40:00:00			#Run for 10 minutes, at most
#SBATCH --job-name=Autoencoder		#Name the job so I can see it in squeue
#SBATCH --output=out/Autoencoder-train-slurm-%J.out
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
module load cuda/cuda-11.4
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

source /home/rvalienteromero/Coop/autoencoder/envcuda/bin/activate
cd /home/rvalienteromero/Coop/autoencoder/

echo
common=""
common+="python3 train_state_AE.py"
# train
# common+=" --latent_space_dim 16"
common+=" --pathbase  /home/rvalienteromero/Coop/autoencoder/Dataset/Image"
common+=" --datatype Image"
common+=" --samples None "
common+=" --learning_rate 0.0005"
common+=" --batch_size 64"
common+=" --epochs 100"


time $common --latent_space_dim 64 
echo


# Done!
echo "Ending script ..."
date