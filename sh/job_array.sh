#!/bin/bash
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=2-00:00:00         # Set a time limit for the job (e.g., 2 days)
#SBATCH --mem=32G                 # Set memory limit (e.g., 32 GB)
#SBATCH --job-name=lambda_array   # Set a name for the job array
#SBATCH --cpus-per-task=8         # Allocate 8 CPU cores for data loading
#SBATCH --array=0-17              # Define the job array indices (18 experiments: 6 lambdas * 3 settings)
#SBATCH --output=/om2/user/raduba/vision-transformers/log/output_%A_%a.log  # Save stdout with array ID
#SBATCH --error=/om2/user/raduba/vision-transformers/log/error_%A_%a.log    # Save stderr with array ID

# Create the logs directory if it doesn't exist
mkdir -p /om2/user/raduba/vision-transformers/log

# Load necessary environment and activate Conda
source /om2/user/raduba/anaconda/etc/profile.d/conda.sh
conda activate DL

# Change directory to your project
cd /om2/user/raduba/vision-transformers

# Array of lambda values
LAMBDAS=(0.1 0.01 0.001 0.0001 0.00001 0.000001)

# Determine experiment configuration
# 0-5: no scheduler
# 6-11: cyclic scheduler
# 12-17: sqrt scheduler
SETTING=$(( SLURM_ARRAY_TASK_ID / 6 ))
LAMBDA_INDEX=$(( SLURM_ARRAY_TASK_ID % 6 ))
INITIAL_LAMBDA=${LAMBDAS[$LAMBDA_INDEX]}

if [ "$SETTING" -eq 0 ]; then
    SCHEDULER_FLAG=""
    JOB_NAME="no_scheduler"
elif [ "$SETTING" -eq 1 ]; then
    SCHEDULER_FLAG="--use_lambda_scheduler"
    JOB_NAME="cyclic"
else
    SCHEDULER_FLAG="--use_sqrt_lambda_scheduler"
    JOB_NAME="sqrt"
fi

# Run the Python script with the selected configuration
srun python train_cifar10.py \
  --net vit_small \
  --opt adamW \
  $SCHEDULER_FLAG \
  --initial-lambda $INITIAL_LAMBDA \
  --n_epochs 800 \
  &> /om2/user/raduba/vision-transformers/log/train_vit_${JOB_NAME}_lambda_${INITIAL_LAMBDA}.log