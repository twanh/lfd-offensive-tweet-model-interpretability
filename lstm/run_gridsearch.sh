#!/bin/bash
#SBATCH --job-name=lstm_offensive_grid_search
#SBATCH --output=/scratch/%u/lfd_final/lstm/grid_search/slurm_logs/slurm-%A_%a.out
#SBATCH --error=/scratch/%u/lfd_final/lstm/grid_search/slurm_logs/slurm-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

set -e
set -x

mkdir -p /scratch/$USER/lfd_final/lstm/grid_search/slurm_logs

module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
source $HOME/venvs/lfd_final/bin/activate

DATA_DIR="$HOME/workspace/lfd-offensive-tweet-model-interpretability/data/clean"
TRAIN_FILE="$DATA_DIR/train.tsv"
DEV_FILE="$DATA_DIR/dev.tsv"
TEST_FILE="$DATA_DIR/test.tsv"
BASE_OUTPUT_DIR="/scratch/$USER/lfd_final/lstm/grid_search"
RESULTS_DIR="$BASE_OUTPUT_DIR/results"

mkdir -p $BASE_OUTPUT_DIR
mkdir -p $RESULTS_DIR

echo "--- Starting Grid Search ---"

python train.py \
    $TRAIN_FILE \
    $DEV_FILE \
    -t $TEST_FILE \
    --embeddings "/scratch/$USER/fasttext/cc.en.300.bin"  \
    --save-model-dir "$RESULTS_DIR/grid_search_best_model/" \
    --epochs 20 \
    --batch_size 32 \
    --grid-search

    >> "$RESULTS_DIR/grid_search_log.txt" 2>&1

echo "Finished job $SLURM_ARRAY_TASK_ID."
