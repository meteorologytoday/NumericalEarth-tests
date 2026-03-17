#! /bin/sh
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#
#SBATCH --job-name=SPG-Oceananigan
#SBATCH --time=05:00:00 
#SBATCH --output=sbatch.64.%j.out
#SBATCH --error=sbatch.1.%j.err
#
#SBATCH --mail-type=TIME_LIMIT

# Flush all output
export JULIA_UNBUFFERED_IO=1

echo "Test if CUDA is functional in Julia"
srun -n1 julia -e 'using CUDA; println(CUDA.functional())'

echo "Print CUDA.versioninfo() in Julia"
srun -n1 julia -e 'using CUDA; CUDA.versioninfo()'

time srun -n1 --unbuffered julia --threads $SLURM_CPUS_PER_TASK src/Oceananigan/main_SPG.jl
