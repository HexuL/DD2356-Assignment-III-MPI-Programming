#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]){
    int rank, size, i, provided;
    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int nxc = 128; // Ensure nxc is divisible by size
    double L = 2 * 3.1415; // Length of the domain
    int nxn_loc = nxc / size + 2; // Number of nodes is number of cells + 1; add 2 ghost cells
    double L_loc = L / size;
    double dx = L / nxc;
    
    double *f = calloc(nxn_loc, sizeof(double)); 
    double *dfdx = calloc(nxn_loc, sizeof(double)); 


    for (i = 1; i < nxn_loc-1; i++) {
        f[i] = sin(L_loc * rank + (i-1) * dx);
    }


    int left = rank - 1;
    int right = rank + 1;
    if (rank == 0) left = size - 1;
    if (rank == size - 1) right = 0;

    MPI_Sendrecv(&f[nxn_loc-2], 1, MPI_DOUBLE, right, 0,
                 &f[0], 1, MPI_DOUBLE, left, 0, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&f[1], 1, MPI_DOUBLE, left, 1,
                 &f[nxn_loc-1], 1, MPI_DOUBLE, right, 1, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

 
    for (i = 1; i < nxn_loc-1; i++) {
        dfdx[i] = (f[i+1] - f[i-1]) / (2 * dx);
    }


    if (rank == 0) {
        printf("My rank %d of %d\n", rank, size);
        printf("Here are my values for f including ghost cells\n");
        for (i = 0; i < nxn_loc; i++)
            printf("%f\n", f[i]);
        printf("\n");
    }

    free(f);
    free(dfdx);
    MPI_Finalize();
    return 0;
}
