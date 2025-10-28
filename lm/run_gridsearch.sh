#!/bin/bash
#SBATCH --job-name=bert_offensive_grid_search
#SBATCH --output=/scratch/%u/lfd_final/grid_search/slurm_logs/slurm-%A_%a.out
#SBATCH --output=/scratch/%u/lfd_final/grid_search/slurm_logs/slurm-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-71

# Basic setup
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"
# Exit if non 0 exit code
set -e
# Print all commands to stdout
set -x

# Setup modules and python
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
source $HOME/venvs/lfd_final/bin/activate

# Defaults
TRAIN_FILE="train.tsv"
DEV_FILE="dev.tsv"
TEST_FILE="test.tsv"
MODEL="bert-base-uncased"
EPOCHS="15"
BASE_OUTPUT_DIR="/scratch/$USER/lfd_final/grid_search/"

# Define the grid
LRS=(1e-5 2e-5 3e-5 5e-5)
BATCH_SIZES=(8 16 32)
MAX_LENGTHS=(64 128 256)
PATIENCE_VALUES=(2 3)

# Calculate the total number of combinations
NUM_LRS=${#LRS[@]}                 # 4
NUM_BATCH_SIZES=${#BATCH_SIZES[@]} # 3
NUM_MAX_LENGTHS=${#MAX_LENGTHS[@]} # 3
NUM_PATIENCE_VALUES=${#PATIENCE_VALUES[@]} # 2

# Map the SLURM_ARRAY_TASK_ID to the grid indices
idx=$SLURM_ARRAY_TASK_ID
PATIENCE_IDX=$((idx % NUM_PATIENCE_VALUES))
MAX_LENGTH_IDX=$(((idx / NUM_PATIENCE_VALUES) % NUM_MAX_LENGTHS))
BATCH_SIZE_IDX=$(((idx / (NUM_PATIENCE_VALUES * NUM_MAX_LENGTHS)) % NUM_BATCH_SIZES))
LR_IDX=$(((idx / (NUM_PATIENCE_VALUES * NUM_MAX_LENGTHS * NUM_BATCH_SIZES)) % NUM_LRS))

# Get the parameters for this specific job
LR=${LRS[$LR_IDX]}
BATCH_SIZE=${BATCH_SIZES[$BATCH_SIZE_IDX]}
MAX_LENGTH=${MAX_LENGTHS[$MAX_LENGTH_IDX]}
PATIENCE=${PATIENCE_VALUES[$PATIENCE_IDX]}

# Create a unique output directory for this run
OUTPUT_DIR="$BASE_OUTPUT_DIR/lr_${LR}__bs_${BATCH_SIZE}__len_${MAX_LENGTH}__pat_${PATIENCE}"
mkdir -p $OUTPUT_DIR

# Define the results file path
RESULTS_FILE="$OUTPUT_DIR/results.txt"

# Write the hyperparameter config to the results.txt file
echo "--- Grid Search Run ---" > $RESULTS_FILE
echo "Job ID: $SLURM_ARRAY_TASK_ID" >> $RESULTS_FILE
echo "Model: $MODEL" >> $RESULTS_FILE
echo "Learning Rate: $LR" >> $RESULTS_FILE
echo "Batch Size: $BATCH_SIZE" >> $RESULTS_FILE
echo "Max Epochs: $EPOCHS" >> $RESULTS_FILE
echo "Max Length: $MAX_LENGTH" >> $RESULTS_FILE
echo "Patience: $PATIENCE" >> $RESULTS_FILE
echo "Output Dir: $OUTPUT_DIR" >> $RESULTS_FILE
echo "-----------------------" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "--- Training Log ---" >> $RESULTS_FILE

# Add all stdout and stderr from the python script to results.txt
python train.py \
    $TRAIN_FILE \
    $DEV_FILE \
    --test-file $TEST_FILE \
    --model_name "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $EPOCHS \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --early_stopping_patience $PATIENCE \
    --load_best_model_at_end \
    --metric_for_best_model "eval_f1_macro" \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --confusion_matrix "$OUTPUT_DIR/test_confusion_matrix.png" \
    >> $RESULTS_FILE 2>&1

echo "Job finished. Results and model saved to $OUTPUT_DIR"
