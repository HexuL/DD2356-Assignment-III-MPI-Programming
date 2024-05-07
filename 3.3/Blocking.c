#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, size, nxc = 128; // Ensure nxc is divisible by size
    double L = 2 * M_PI; // Length of the domain
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    int nxn_loc = nxc / size + 2; // +2 for ghost cells
    double dx = L / nxc;
    double *f = calloc(nxn_loc, sizeof(double));
    double *dfdx = calloc(nxn_loc, sizeof(double));

 
    for (int i = 1; i < nxn_loc - 1; i++) {
        f[i] = sin((rank * (nxn_loc - 2) + (i - 1)) * dx);
    }


    MPI_Status status;
    int left = rank - 1;
    int right = rank + 1;
    if (rank == 0) left = size - 1;
    if (rank == size - 1) right = 0;

    MPI_Sendrecv(&f[nxn_loc - 2], 1, MPI_DOUBLE, right, 0,
                 &f[0], 1, MPI_DOUBLE, left, 0,
                 MPI_COMM_WORLD, &status);


    MPI_Sendrecv(&f[1], 1, MPI_DOUBLE, left, 1,
                 &f[nxn_loc - 1], 1, MPI_DOUBLE, right, 1,
                 MPI_COMM_WORLD, &status);

   
    for (int i = 1; i < nxn_loc - 1; i++) {
        dfdx[i] = (f[i + 1] - f[i - 1]) / (2 * dx);
    }

   
    if (rank == 0) {
        printf("Rank %d, f and dfdx values:\n", rank);
        for (int i = 0; i < nxn_loc; i++)
            printf("f[%d] = %f, dfdx[%d] = %f\n", i, f[i], i, dfdx[i]);
    }

    MPI_Finalize();
    free(f);
    free(dfdx);
    return 0;
}
