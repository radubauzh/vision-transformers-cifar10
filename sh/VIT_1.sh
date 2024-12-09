#!/bin/bash
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=2-00:00:00         # Set a time limit for the job (e.g., 2 days)
#SBATCH --mem=32G                 # Set memory limit (e.g., 32 GB)
#SBATCH --job-name=ViT_small      # Set a name for the job
#SBATCH --cpus-per-task=8         # Allocate 8 CPU cores for data loading
#SBATCH --output=/om2/user/raduba/vision-transformers/log/output_vit_small.log  # Save stdout in log folder
#SBATCH --error=/om2/user/raduba/vision-transformers/log/error_vit_small.log    # Save stderr in log folder

# Create the logs directory if it doesn't exist
mkdir -p /om2/user/raduba/vision-transformers/log

# Load necessary environment and activate Conda
source /om2/user/raduba/anaconda/etc/profile.d/conda.sh
conda activate DL

# Change directory to your project
cd /om2/user/raduba/vision-transformers

# Run the Python script
srun python train_cifar10.py --net vit_small --nowandb --n_epochs 400 &> /om2/user/raduba/vision-transformers/log/train_vit_small.log
