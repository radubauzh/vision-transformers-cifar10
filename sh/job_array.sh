#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --job-name=lambda_array
#SBATCH --cpus-per-task=8
#SBATCH --array=0-17
#SBATCH --output=/om2/user/raduba/vision-transformers/log/output_%A_%a.log
#SBATCH --error=/om2/user/raduba/vision-transformers/log/error_%A_%a.log

mkdir -p /om2/user/raduba/vision-transformers/log

source /om2/user/raduba/anaconda/etc/profile.d/conda.sh
conda activate DL

cd /om2/user/raduba/vision-transformers

LAMBDAS=(0.1 0.01 0.001 0.0001 0.00001 0.000001)

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

# Use the multiplicative regularization directly
# For example, we set lambda_mul = INITIAL_LAMBDA (and lambda_sum=0.0) and enable features_normalization
srun python train_cifar10.py \
  --net vit_small \
  --opt adamW \
  $SCHEDULER_FLAG \
  --initial-lambda $INITIAL_LAMBDA \
  --n_epochs 800 \
  --lambda_mul $INITIAL_LAMBDA \
  --lambda_sum 0.0 \
  --features_normalization \
  &> /om2/user/raduba/vision-transformers/log/train_vit_${JOB_NAME}_lambda_${INITIAL_LAMBDA}.log
