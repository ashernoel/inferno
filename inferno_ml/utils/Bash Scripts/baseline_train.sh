#!/bin/bash

# Define the CSV filename
csvfile="/n/home10/anoel/configurations_ALL_flops.csv"

# Ensure the script is called with sbatch
if [ "$SLURM_JOB_ID" == "" ]; then
    echo "Script must be submitted through sbatch"
    exit 1
fi

# Read CSV and submit jobs
{
    read # Skip the header
    while IFS=, read -r block depth width flops
    do
        # You can use $block, $depth, $width here as your arguments
        echo "Submitting job with Block=$block, Depth=$depth, Width=$width"
        
        # Construct and execute the sbatch command with the appropriate arguments
        sbatch <<EOT
#!/bin/bash
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -t 0-48:00
#SBATCH -p seas_gpu
#SBATCH --mem=32000
#SBATCH -o myoutput_${block}_${depth}_${width}_%j.out
#SBATCH -e myerrors_${block}_${depth}_${width}_%j.err
#SBATCH --constraint=a100|v100
module load Anaconda2/2019.10-fasrc01 cuda/12.2.0-fasrc01 cudnn
export LIBRARY_PATH=/n/sw/eb/apps/centos7/Anaconda2/2019.10-fasrc01/lib/:$LIBRARY_PATH
export LD_LIBRARY_PATH=/n/sw/eb/apps/centos7/Anaconda2/2019.10-fasrc01/lib/:$LD_LIBRARY_PATH
source activate nnet_calculator

# Call your python script with the parameters
python3 run_models_target_flops.py --block $block --depth $depth --width $width
EOT

    done
} < $csvfile
