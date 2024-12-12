#!/bin/bash
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=2-00:00:00         # Set a time limit for the job (e.g., 2 days)
#SBATCH --mem=32G                 # Set memory limit (e.g., 32 GB)
#SBATCH --job-name=cyclic_2         # Set a name for the job
#SBATCH --cpus-per-task=8         # Allocate 8 CPU cores for data loading
#SBATCH --output=/om2/user/raduba/vision-transformers/log/output_vit_with_cyclic_2.log  # Save stdout in log folder
#SBATCH --error=/om2/user/raduba/vision-transformers/log/error_vit_with_cyclic_2.log    # Save stderr in log folder

# Create the logs directory if it doesn't exist
mkdir -p /om2/user/raduba/vision-transformers/log

# Load necessary environment and activate Conda
source /om2/user/raduba/anaconda/etc/profile.d/conda.sh
conda activate DL

# Change directory to your project
cd /om2/user/raduba/vision-transformers

# Run the Python script with cyclic cosine lambda and specified initial lambda
srun python train_cifar10.py \
  --net vit_small \
  --opt adamW \
  --use_lambda_scheduler \
  --initial-lambda 0.001 \
  --n_epochs 800 \
  &> /om2/user/raduba/vision-transformers/log/train_vit_with_cyclic_2.log

# srun python train_cifar10.py --net vit_small --opt adamW --use_lambda_scheduler --initial-lambda 0.001 --nowandb --n_epochs 400 &> /om2/user/raduba/vision-transformers/log/train_vit_with_cyclic_no_wandb.log