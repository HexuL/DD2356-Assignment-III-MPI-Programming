#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h> 

#define TILE_SIZE (matrix_size / p)  


void read_input_matrices(double** A, double** B, int* matrix_size) {
    *matrix_size = 16;

    double A_data[] = {
        45, 48, 65, 68, 68, 10, 84, 22, 37, 88, 71, 89, 89, 13, 59, 66,
        40, 88, 47, 89, 82, 38, 26, 78, 73, 10, 21, 81, 70, 80, 48, 65,
        83, 100, 89, 50, 30, 20, 20, 15, 40, 33, 66, 10, 58, 33, 32, 75,
        24, 36, 76, 56, 29, 35, 1, 1, 37, 54, 6, 39, 18, 80, 5, 43, 59,
        32, 2, 66, 42, 58, 36, 12, 47, 83, 92, 1, 15, 100, 54, 13, 43, 85,
        76, 69, 7, 69, 48, 4, 77, 53, 79, 16, 21, 100, 59, 24, 80, 14, 86,
        49, 50, 70, 42, 36, 65, 96, 70, 95, 1, 51, 37, 35, 49, 94, 4, 99,
        43, 78, 22, 74, 1, 11, 44, 59, 24, 60, 3, 99, 63, 36, 95, 68, 83,
        47, 100, 21, 82, 51, 28, 15, 42, 59, 66, 37, 11, 87, 44, 12, 3, 52,
        81, 33, 55, 1, 39, 20, 47, 43, 57, 61, 78, 31, 25, 3, 4, 95, 99,
        14, 41, 73, 20, 96, 73, 27, 67, 53, 68, 62, 15, 97, 5, 68, 12, 87,
        78, 76, 57, 17, 25, 30, 22, 26, 81, 61, 62, 84, 34, 33, 71, 86, 32,
        14, 72, 57, 25, 80, 42, 19, 41, 55, 80, 12, 39, 94, 2, 96, 45, 89,
        25, 68, 83, 4, 77, 36, 87, 62, 70, 88, 44, 33, 12, 85, 11, 55, 38,
        29, 3, 28, 84, 90, 24, 54, 52, 47, 21, 54, 30, 68, 36, 40, 10, 74,
        42, 24, 4, 47
    };

    double B_data[] = {
        91, 51, 4, 32, 10, 11, 28, 46, 72, 40, 62, 86, 98, 45, 35, 35,
        89, 34, 6, 37, 1, 76, 35, 70, 54, 81, 63, 9, 62, 2, 82, 36,
        92, 41, 37, 49, 26, 68, 36, 31, 30, 34, 19, 18, 94, 85, 3, 70,
        13, 45, 67, 92, 86, 40, 40, 76, 23, 31, 18, 71, 72, 19, 93, 44,
        84, 50, 42, 94, 47, 22, 74, 90, 97, 92, 74, 29, 82, 59, 1, 87,
        64, 17, 37, 95, 25, 64, 68, 52, 9, 57, 92, 94, 88, 33, 20, 73,
        72, 88, 14, 59, 82, 56, 65, 76, 93, 37, 26, 33, 43, 15, 87, 29,
        21, 83, 69, 23, 100, 84, 8, 73, 62, 14, 6, 1, 9, 80, 80, 54, 12,
        5, 40, 93, 46, 27, 75, 53, 50, 92, 52, 100, 19, 35, 52, 31, 54,
        59, 44, 56, 19, 46, 88, 66, 71, 54, 49, 95, 60, 81, 27, 36, 59,
        50, 74, 45, 14, 71, 39, 40, 9, 14, 8, 81, 23, 80, 90, 9, 100,
        7, 82, 72, 85, 90, 67, 61, 17, 57, 24, 25, 5, 50, 88, 31, 55,
        26, 21, 98, 58, 24, 28, 30, 34, 54, 52, 87, 8, 10, 55, 1, 84,
        37, 82, 21, 4, 43, 66, 21, 37, 69, 81, 48, 11, 95, 92, 44, 64,
        32, 21, 71, 10, 61, 92, 36, 84, 77, 19, 75, 99, 98, 44, 4, 13,
        59, 2, 1, 40, 25, 59, 37, 100, 70, 7, 4, 99, 41, 61, 34
    };


int dim = *matrix_size;  

double* temp_A = (double*)malloc(dim * dim * sizeof(double));
if (temp_A == NULL) {
}

double* temp_B = (double*)malloc(dim * dim * sizeof(double));
if (temp_B == NULL) { 
}

    for (int i = 0; i < (*matrix_size) * (*matrix_size); i++) {
        temp_A[i] = A_data[i];
        temp_B[i] = B_data[i];
    }

    *A = temp_A;
    *B = temp_B;
}

int main(int argc, char** argv) {
    int processes, rank;
    int p;                        
    int grid_rank;                 
    int grid_coords[2];
    MPI_Comm grid_comm;        
    MPI_Comm row_comm;
    MPI_Comm col_comm;      
    double start_time, end_time;  

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);


    p = (int)sqrt(processes);
    if (p * p != processes) {
        if (rank == 0) {
            printf("[ERROR] The number of processes must be a integer square, is %i.", p);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    double *A, *B;              
    double *local_A, *local_B;  
    int matrix_size = 0;       

    if (rank == 0) {
        read_input_matrices(&A, &B, &matrix_size);

        printf("Got matrix size: %d\n", matrix_size);
        printf("Number of processes: %d\n", processes);
        printf("Time will start now...");

        if (matrix_size % p != 0) {
            printf("[ERROR] The matrix size must be divisible by the root of the number of processes.");

            MPI_Finalize();
            return EXIT_FAILURE;
        }
        else {
        }
    }

    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

local_A = (double*)malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
local_B = (double*)malloc(TILE_SIZE * TILE_SIZE * sizeof(double));

    int ndims = 2;           
    int dims[2] = {p, p};        
    int periods[2] = {1, 1};      
    int reorder = 1;                
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &grid_comm);
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_coords(grid_comm, grid_rank, ndims, grid_coords);

    int remain_dims_row[2] = {0, 1};    
    MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);

    int remain_dims_col[2] = {1, 0};
    MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

    MPI_Barrier(row_comm);
    MPI_Barrier(col_comm);

 double* local_C = (double*)malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
    double* C_full = NULL;
    if (rank == 0) {
         C_full = (double*)malloc(matrix_size * matrix_size * sizeof(double));
    }

     MPI_Datatype row_type, block_type;
    MPI_Type_contiguous(TILE_SIZE, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);
    MPI_Type_vector(TILE_SIZE, 1, matrix_size / TILE_SIZE, row_type, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double) * TILE_SIZE, &block_type);
    MPI_Type_commit(&block_type);

    // 计算 sendcounts 和 displacements
    int* sendcounts = NULL;
    int* displacements = NULL;
    if (rank == 0) {
        int coords[2];
        sendcounts = malloc(processes * sizeof(int));
        displacements = malloc(processes * sizeof(int));

        for (int i = 0; i < processes; i++) {
            sendcounts[i] = 1; 
            MPI_Cart_coords(grid_comm, i, 2, coords);
            displacements[i] = coords[0] * matrix_size + coords[1]; 
        }
    }

    // 分发数据块
    MPI_Scatterv(A, sendcounts, displacements, block_type, local_A, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, 0, grid_comm);
    MPI_Scatterv(B, sendcounts, displacements, block_type, local_B, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, 0, grid_comm);

    // 释放内存
    if (rank == 0) {
        free(sendcounts);
        free(displacements);
    }

    // 释放 MPI_Datatype
    MPI_Type_free(&row_type);
    MPI_Type_free(&block_type);

    int source, dest;
    source = (grid_coords[0] + 1) % p;
    dest = (grid_coords[0] + p - 1) % p;

    double* temp_A = (double*)malloc(TILE_SIZE * TILE_SIZE * sizeof(double));

    for (int step = 0; step < p; step++) {

        int root = (grid_coords[0] + step) % p;

if (root == grid_coords[1]) {
    MPI_Bcast(local_A, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, root, row_comm);
    for (int x = 0; x < TILE_SIZE; x++) {
        for (int y = 0; y < TILE_SIZE; y++) {
            for (int z = 0; z < TILE_SIZE; z++) {
                local_C[x * TILE_SIZE + y] += local_A[x * TILE_SIZE + z] * local_B[z * TILE_SIZE + y];
            }
        }
    }
}
else {
    MPI_Bcast(temp_A, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, root, row_comm);
    for (int x = 0; x < TILE_SIZE; x++) {
        for (int y = 0; y < TILE_SIZE; y++) {
            for (int z = 0; z < TILE_SIZE; z++) {
                local_C[x * TILE_SIZE + y] += temp_A[x * TILE_SIZE + z] * local_B[z * TILE_SIZE + y];
            }
        }
    }
}


        MPI_Sendrecv_replace(local_B, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, dest, 0, source, 0, col_comm, MPI_STATUS_IGNORE);
    }

    free(temp_A);

    MPI_Gather(local_C, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, C_full, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, 0, grid_comm);


    if (rank == 0) {
        end_time = MPI_Wtime();

        double* C_fixed = (double*)malloc(matrix_size * matrix_size * sizeof(double));
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                int block_start = (i * p + j) * TILE_SIZE * TILE_SIZE;
                for (int k = 0; k < TILE_SIZE; k++) {
                    memcpy(&C_fixed[(i * TILE_SIZE + k) * matrix_size + j * TILE_SIZE], &C_full[block_start + k * TILE_SIZE], TILE_SIZE * sizeof(double));
                }
            }
        }

         printf("Result matrix:\n");
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            printf("%6.2f ", C_fixed[i * matrix_size + j]);
        }
        printf("\n");
    }
    printf("\n");
        free(C_fixed);

        if (rank == 0) {
            double runtime = end_time - start_time;
            printf("Number of processes: %d, Runtime: %f seconds\n", processes, runtime);
        }
    }

    free(local_A);
    free(local_B);
    free(local_C);
    if (rank == 0) {
        free(A);
        free(B);
        free(C_full);
    }
 
    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&row_comm);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
