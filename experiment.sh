#!/bin/bash 

#SBATCH --partition=main	# Partition (job queue) 
#SBATCH --requeue 		# Return job to the queue if preempted 
#SBATCH --job-name=cifar	# Assign an short name to your job 
#SBATCH --nodes=1		# Number of nodes you require 
#SBATCH --ntasks=20		# Total # of tasks across all nodes
#SBATCH --cpus-per-task=1 	# Cores per task (>1 if multithread tasks) 
#SBATCH --mem-per-cpu=1000		# Real memory (RAM) required (MB) 
#SBATCH --time=3:00:00 	# Total run time limit (HH:MM:SS) 
#SBATCH --output=slurm.%N.%j.out 	# STDOUT output file 
#SBATCH --error=slurm.%N.%j.err 	# STDERR output file (optional) 
#SBATCH --export=ALL	 # Export you current env to the job 

env cd /scratch/bx25 
module purge 
module load python/3.5.2
module load intel/18.0.5
module load openmpi/2.1.1


hostname
srun --mpi=pmi2 -n 20 python /scratch/bx25/Thesis_mpi_obj/run.py


sacct -o Elapsed,Start,End,job,jobName,NTasks,MaxRSS,MaxVMSize,AveRSS,AveVMSize,AveCPU -j $SLURM_JOBID