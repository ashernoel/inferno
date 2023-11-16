#!/bin/bash
#SBATCH -c 10                                       # Number of cores (-c)
#SBATCH --gres=gpu:4                                # Number of GPUs
#SBATCH -t 0-130:00                                  # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu  # Partition to submit to
#SBATCH --mem=256000                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out                          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err                          # File to which STDERR will be written, %j inserts jobid
#SBATCH --constraint=a100
module load Anaconda2/2019.10-fasrc01 cuda/12.2.0-fasrc01 cudnn
export LIBRARY_PATH=/n/sw/eb/apps/centos7/Anaconda2/2019.10-fasrc01/lib/:$LIBRARY_PATH
export LD_LIBRARY_PATH=/n/sw/eb/apps/centos7/Anaconda2/2019.10-fasrc01/lib/:$LD_LIBRARY_PATH
source activate nnet_calculator
# python3 transfer_learning.py
python3 gpu_training_script.py
