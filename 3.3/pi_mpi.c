#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SEED     921
#define NUM_ITER 1000000000

int main(int argc, char* argv[])
{
    int rank, num_ranks, iter, provided;
    double x, y, z, pi;
    int local_count = 0, total_count;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    double start_time = MPI_Wtime();

    srand(SEED * rank); // Unique seed for each rank
    int local_iter = NUM_ITER / num_ranks; // Divide the total iterations by the number of ranks

    for (iter = 0; iter < local_iter; iter++)
    {
        x = (double)rand() / (double)RAND_MAX;
        y = (double)rand() / (double)RAND_MAX;
        z = sqrt((x * x) + (y * y));

        if (z <= 1.0)
        {
            local_count++;
        }
    }

    // Use MPI_Reduce to sum all local counts into total_count on rank 0
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        pi = ((double)total_count / (double)(NUM_ITER)) * 4.0;
        double end_time = MPI_Wtime();
        printf("The result is %f\n", pi);
        printf("Time taken: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
