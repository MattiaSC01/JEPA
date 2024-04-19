#!/bin/bash

#SBATCH --job-name=sweep
#SBATCH --output=outputs/%x/%j.out
#SBATCH --error=outputs/%x/%j.err
#SBATCH --mail-user=mattia.scardecchia@studbocconi.it
#SBATCH --mail-type=None
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode03


# load modules
module load cuda/12.3

# activate conda environment
source /home/3144860/miniconda3/bin/activate jepa

# go to the directory where the script is located
cd /home/3144860/JEPA/scripts

# run the python script
python ae_sweep.py --project jepa-prove --config ../config/ae_sweep.yaml --count 15

# run the wandb cleanup script
cd /home/3144860/JEPA/slurm
./wandb_cleanup.sh

# deactivate conda environment
conda deactivate
