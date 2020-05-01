#!/bin/bash 


#SBATCH --partition=main	# Partition (job queue) 
#SBATCH --requeue 		# Return job to the queue if preempted 
#SBATCH --job-name=process	# Assign an short name to your job 
#SBATCH --nodes=1 		# Number of nodes you require 
#SBATCH --ntasks=1		# Total # of tasks across all nodes
#SBATCH --cpus-per-task=5 	# Cores per task (>1 if multithread tasks) 
#SBATCH --mem-per-cpu=8000		# Real memory (RAM) required (MB) 
#SBATCH --time=50:00:00 	# Total run time limit (HH:MM:SS) 
#SBATCH --output=slurm.%N.%j.out 	# STDOUT output file 
#SBATCH --error=slurm.%N.%j.err 	# STDERR output file (optional) 
#SBATCH --export=ALL	 # Export you current env to the job 

env cd /scratch/bx25 
module purge 
module load python/3.5.2
module load intel/18.0.5


python resizeImg.py