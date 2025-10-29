#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --array=0-71%15 # 72 jobs (4x3x1x3x2), max 15 run at once
# NEW: Store SLURM logs in a subfolder of the main output directory
#SBATCH --output=/scratch/%u/LFD/output/slurm_logs/slurm-%A_%a.out
#SBATCH --error=/scratch/%u/LFD/output/slurm_logs/slurm-%A_%a.err

module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/lfd3_env/bin/activate

# --- Define Hyperparameters ---
learning_rates=(1e-5 2e-5 3e-5 5e-5)
batch_sizes=(8 16 32)
epochs=(15)
max_lengths=(64 128 256)
patience_values=(2 3)

# --- Define Output Paths ---
# NEW: Define a root output directory in scratch space and a single log file
OUTPUT_ROOT="/scratch/$USER/LFD/output"
LOG_FILE="$OUTPUT_ROOT/results.log"
# NEW: Ensure the directories for logs and results exist
mkdir -p "$OUTPUT_ROOT/slurm_logs"

# --- Calculate Job Indices (no changes here) ---
total_lrs=${#learning_rates[@]}
total_bs=${#batch_sizes[@]}
total_epochs=${#epochs[@]}
total_max_lengths=${#max_lengths[@]}
total_patience=${#patience_values[@]}
id=$SLURM_ARRAY_TASK_ID
patience_index=$(( id % total_patience )); id=$(( id / total_patience ))
ml_index=$(( id % total_max_lengths )); id=$(( id / total_max_lengths ))
epoch_index=$(( id % total_epochs )); id=$(( id / total_epochs ))
bs_index=$(( id % total_bs )); id=$(( id / total_bs ))
lr_index=$(( id % total_lrs ))

# --- Get Hyperparameter Values ---
lr=${learning_rates[$lr_index]}
batch_size=${batch_sizes[$bs_index]}
epoch=${epochs[$epoch_index]}
max_len=${max_lengths[$ml_index]}
patience=${patience_values[$patience_index]}

# NEW: The checkpoint directory for this specific job
checkpoint_dir="$OUTPUT_ROOT/lr_${lr}_bs_${batch_size}_ml_${max_len}_pat_${patience}"

# NEW: Construct the full command as a variable for logging
COMMAND="python3 ./lfd_assignment3_lm.py \
    --model_name 'bert-base-uncased' \
    --output_dir $checkpoint_dir \
    --num_train_epochs $epoch \
    --learning_rate $lr \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --max_length $max_len \
    --seed 42 \
    --eval_strategy 'epoch' \
    --load_best_model_at_end \
    --early_stopping_patience $patience \
    --metric_for_best_model 'eval_f1_macro'"

echo "Job ${SLURM_ARRAY_TASK_ID}: Executing..."

python3 ./lfd_assignment3_lm.py \
    --model_name 'bert-base-uncased' \
    --output_dir $checkpoint_dir \
    --num_train_epochs $epoch \
    --learning_rate $lr \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --max_length $max_len \
    --seed 42 \
    --eval_strategy 'epoch' \
    --load_best_model_at_end \
    --early_stopping_patience $patience \
    --metric_for_best_model 'eval_f1_macro'

# echo "Job ${SLURM_ARRAY_TASK_ID}: Executing..."
# echo "$COMMAND"

# NEW: Execute the command and capture its entire output
# RUN_OUTPUT=$($COMMAND)
# RUN_OUTPUT=$($COMMAND 2>&1)

# NEW: Extract the specific metrics line from the output
# FINAL_METRICS=$(echo "$RUN_OUTPUT" | grep "FINAL_METRICS")

# NEW: Use a lock to safely append the command and results to the log file
# This prevents different jobs from writing at the same time and corrupting the file
# while ! mkdir "$LOG_FILE.lock" 2>/dev/null; do
#     sleep 1
# done

# {
#     echo "-----------------------------------------------------------"
#     echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
#     echo "Finished at: $(date)"
#     echo "Command:"
#     echo "$COMMAND"
#     echo "Metrics: $FINAL_METRICS"
#     echo ""
# } >> "$LOG_FILE"

# rmdir "$LOG_FILE.lock"

echo "Done."
deactivate
