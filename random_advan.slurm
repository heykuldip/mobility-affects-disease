#!/bin/bash

#Submit this script with: sbatch random_advan.slurm

#SBATCH --job-name=random_advan ##name that will show up in the queue
#SBATCH --output=./output/%x-%N-%j.out   ##filename of the output
#SBATCH --error=./output/%x-%N-%j.err   ##filename of the error output
#SBATCH --nodes=1  ##number of nodes to use
#SBATCH --ntasks=64  ##number of tasks to run
#SBATCH --time=2-00:00:00  ##time for analysis (day-hour:min:sec)
#SBATCH --cpus-per-task=1  ##the number of threads the code will use
#SBATCH --mem=400GB  ##memory allocation per node
#SBATCH --partition normal ##the partition to run in

## Commands to Load Modules:
module load gnu9 intel mpich git singularity

## Run your programs here.
mpirun -n 64 singularity run /containers/hopper/UserContainers/$USER/repast4py_latest.sif python ./random_advan.py ./config/random_advan.yaml
