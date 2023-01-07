#!/bin/bash -l
#
# time
#SBATCH --time=23:00:00
#
# job name 
#SBATCH --job-name=heraus-test1
#
# cluster
#SBATCH --clusters=tinygpu
#
# Essential for FAU HPC system - do not change
# --------------------------------------------
# do not export environment variables 
#SBATCH --export=NONE 
#
# do not export environment variables
unset SLURM_EXPORT_ENV
# --------------------------------------------
#
#load required modules (compiler, ...)
module load python/3.8-anaconda
module load cuda
#
# anaconda
source activate base
#
# run
python nnOOD_run_training.py fullres nnOODTrainerDS heraus_png FPI 1
