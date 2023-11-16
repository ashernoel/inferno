#!/bin/bash
#SBATCH --gres=gpu:1                                # Number of GPUs
#SBATCH -t 0-24:00                                  # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu  # Partition to submit to
#SBATCH --mem=32000                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o my_prune_myoutput_%j.out                          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e my_prune_myerrors_%j.err                          # File to which STDERR will be written, %j inserts jobid
#SBATCH --constraint=a100
TOKENIZER_SOURCE_PATH="/n/idreos_lab/users/anoel/llama/llama-hf/"
BASE_PATH="/n/idreos_lab/users/anoel/pruned_llama/product/"

# Declare an array to hold the job details
declare -a JOB_DETAILS

for PERCENTAGE in 0.7
# for PERCENTAGE in 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
    if [ "$PERCENTAGE" = "0.1" ]; then
        PRUNING_PATH="ten_percent_pruning"
    elif [ "$PERCENTAGE" = "0.2" ]; then
        PRUNING_PATH="twenty_percent_pruning"
    elif [ "$PERCENTAGE" = "0.3" ]; then
        PRUNING_PATH="thirty_percent_pruning"
    elif [ "$PERCENTAGE" = "0.4" ]; then
        PRUNING_PATH="forty_percent_pruning"
    elif [ "$PERCENTAGE" = "0.5" ]; then
        PRUNING_PATH="fifty_percent_pruning"
    elif [ "$PERCENTAGE" = "0.6" ]; then
        PRUNING_PATH="sixty_percent_pruning"
    elif [ "$PERCENTAGE" = "0.7" ]; then
        PRUNING_PATH="seventy_percent_pruning"
    elif [ "$PERCENTAGE" = "0.8" ]; then
        PRUNING_PATH="eighty_percent_pruning"
    else
        PRUNING_PATH="other_pruning"
    fi

    # for MODE in "attn_and_mlp/with_prompt" 
    # for MODE in "attn_and_mlp/magnitude_based"
    # for MODE in "attn_and_mlp_2" "attn_and_mlp_2/with_prompt" 
    # for MODE in "skewness/magnitude_based" 
    for MODE in "traditional"
    # for MODE in "skewness" "skewness/with_prompt" "traditional" 
    do
        # Submitting job to slurm and waiting for it to finish
#         JOB_ID=$(sbatch <<EOL | awk '{print $4}'
# #!/bin/bash
# #SBATCH -c 10                                       # Number of cores (-c)
# #SBATCH --gres=gpu:1                                # Number of GPUs
# #SBATCH -t 0-24:00                                  # Runtime in D-HH:MM, minimum of 10 minutes
# #SBATCH -p seas_gpu                                 # Partition to submit to
# #SBATCH --mem=32000                                 # Memory pool for all cores (see also --mem-per-cpu)
# #SBATCH -o llama_training_myoutput_%j.out                          # File to which STDOUT will be written, %j inserts jobid
# #SBATCH -e llama_training_myerrors_%j.err                          # File to which STDERR will be written, %j inserts jobid
# #SBATCH --constraint=a100
# module load Anaconda2/2019.10-fasrc01 cuda/12.2.0-fasrc01 cudnn
# export LIBRARY_PATH=/n/sw/eb/apps/centos7/Anaconda2/2019.10-fasrc01/lib/:$LIBRARY_PATH
# export LD_LIBRARY_PATH=/n/sw/eb/apps/centos7/Anaconda2/2019.10-fasrc01/lib/:$LD_LIBRARY_PATH
# source activate nnet_calculator

# python3 lm-evaluation-harness/product_baesd_pruning.py --percentage $PERCENTAGE --mode $MODE

# EOL
# )
   # Store the job details as a string
    PRUNED_MODEL_PATH="${BASE_PATH}${PRUNING_PATH}/${MODE}"
    
    # # Wait for the job to complete
    # while squeue -j $JOB_ID | grep -q $JOB_ID; do
    #     echo "Job $JOB_ID is still running"
    #     sleep 10
    # done

    echo "PATH $PRUNED_MODEL_PATH READY"
    
# Moving the tokenizer to the newly created directory
    cp ${TOKENIZER_SOURCE_PATH}tokenizer.json ${PRUNED_MODEL_PATH} || echo "Failed to copy tokenizer.json"
    cp ${TOKENIZER_SOURCE_PATH}tokenizer.model ${PRUNED_MODEL_PATH} || echo "Failed to copy tokenizer.model"
    cp ${TOKENIZER_SOURCE_PATH}tokenizer_config.json ${PRUNED_MODEL_PATH} || echo "Failed to copy tokenizer_config.json"
        
    # Submitting the Python evaluation command as a separate sbatch job
    sbatch <<EOL
#!/bin/bash
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -t 0-24:00
#SBATCH -p seas_gpu
#SBATCH --mem=32000
#SBATCH -o LLAMA_evaluation_output_%j.out
#SBATCH -e LLAMA_evaluation_errors_%j.err
#SBATCH --constraint=a100
module load Anaconda2/2019.10-fasrc01 cuda/12.2.0-fasrc01 cudnn
export LIBRARY_PATH=/n/sw/eb/apps/centos7/Anaconda2/2019.10-fasrc01/lib/:$LIBRARY_PATH
export LD_LIBRARY_PATH=/n/sw/eb/apps/centos7/Anaconda2/2019.10-fasrc01/lib/:$LD_LIBRARY_PATH
source activate nnet_calculator

python3 lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=${PRUNED_MODEL_PATH} \
    --tasks hellaswag \
    --device cuda:0

EOL
        

    done
done
