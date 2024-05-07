#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, size, nxc = 128; 
    double L = 2 * M_PI; 
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

    MPI_Request request[4];
    MPI_Status status[4];
    int left = rank - 1;
    int right = rank + 1;
    if (rank == 0) left = size - 1;
    if (rank == size - 1) right = 0;

  
    MPI_Isend(&f[nxn_loc - 2], 1, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(&f[0], 1, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &request[1]);

   
    MPI_Isend(&f[1], 1, MPI_DOUBLE, left, 1, MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(&f[nxn_loc - 1], 1, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &request[3]);

    
    MPI_Waitall(4, request, status);

    
    for (int i = 1; i < nxn_loc - 1; i++) {
        dfdx[i] = (f[i + 1] - f[i - 1]) / (2 * dx);
    }

   
    if (rank == 0) { // print only rank 0 for convenience
        printf("Rank %d, f and dfdx values:\n", rank);
        for (int i = 0; i < nxn_loc; i++)
            printf("f[%d] = %f, dfdx[%d] = %f\n", i, f[i], i, dfdx[i]);
    }

    MPI_Finalize();
    free(f);
    free(dfdx);
    return 0;
}
