#!/bin/bash
#SBATCH --job-name=mpi_hello_world
#SBATCH --partition=main      # Specify the partition
#SBATCH --ntasks=4
#SBATCH --time=00:10:00
#SBATCH --output=mpi_output.txt

# Load the MPI module if needed
module load gcc

# Run the MPI program
srun ./mpi_hello_world
