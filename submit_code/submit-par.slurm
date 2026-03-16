#! /bin/sh
##SBATCH --account=next
#SBATCH --job-name=Oceananigan.jl-MOC
#
#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#
#SBATCH --qos=standby
#SBATCH --time=5:00:00 
#
#
#SBATCH --output=sbatch.64.%j.out
#SBATCH --error=sbatch.1.%j.err
#
#SBATCH --mail-type=TIME_LIMIT

# Flush all output
export JULIA_UNBUFFERED_IO=1

time srun -n1 --unbuffered julia --threads auto src/main.jl
