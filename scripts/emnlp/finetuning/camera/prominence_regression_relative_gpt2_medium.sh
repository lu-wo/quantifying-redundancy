#!/bin/bash

#SBATCH --output=/om/user/luwo/projects/MIT_prosody/slurm_log/%j.out
#SBATCH --error=/om/user/luwo/projects/MIT_prosody/slurm_log/%j.err

#SBATCH -p evlab
#SBATCH -t 47:00:00 
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Loop to run the experiment with seeds from 1 to 5
for i in {1..10}
do
  echo "Running experiment with seed=$i"

  # Binary or script to execute, with seed passed as an argument
  python src/train.py experiment=emnlp/finetuning/prominence_regression_relative_gpt2_medium seed=$i

  echo "Finished experiment with seed=$i at: $(date)"
done

# Send more noteworthy information to the output log
echo "All experiments finished at: $(date)"

# End the script with exit code 0
exit 0



