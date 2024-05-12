#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define TILE_SIZE (matrix_size / p)

void read_input_matrices(double** A, double** B, int* matrix_size) {
    *matrix_size = 16; 

    int total_elements = (*matrix_size) * (*matrix_size); 

    
    *A = (double*)malloc(total_elements * sizeof(double));
    *B = (double*)malloc(total_elements * sizeof(double));

    // 使用从 0 开始的递增值填充矩阵 A 和 B
    for (int i = 0; i < total_elements; i++) {
        (*A)[i] = (double)i;
        (*B)[i] = (double)i; 
    }
}
void copy_matrix_block(double *dest, double *src, int dest_start, int src_start, int num_rows, int dest_stride, int src_stride, int block_size) {
    for (int k = 0; k < num_rows; k++) {
        memcpy(&dest[dest_start + k * dest_stride], &src[src_start + k * src_stride], block_size * sizeof(double));
    }
}

// 封装 MPI_Scatterv 操作
void scatter_matrix(void *matrix, void *local_matrix, int *sendcounts, int *displacements, MPI_Datatype block_type, int tile_size, MPI_Comm grid_comm) {
    MPI_Scatterv(matrix, sendcounts, displacements, block_type, local_matrix, tile_size * tile_size, MPI_DOUBLE, 0, grid_comm);
}

// 封装资源释放操作
void cleanup_resources(int *sendcounts, int *displacements, MPI_Datatype *row_type, MPI_Datatype *block_type, int rank) {
    if (rank == 0) {
        free(sendcounts);
        free(displacements);
    }
    MPI_Type_free(row_type);
    MPI_Type_free(block_type);
}

int main(int argc, char **argv)
{
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

    double *A, *B;
    double *local_A, *local_B;
    int matrix_size = 0;

    if (rank == 0)
    {
        read_input_matrices(&A, &B, &matrix_size);
        printf("matrix size: %d\n", matrix_size);
        printf("processes: %d\n", processes);
    }

    if (rank == 0)
    {
        start_time = MPI_Wtime();
    }

    MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_A = (double *)malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
    local_B = (double *)malloc(TILE_SIZE * TILE_SIZE * sizeof(double));

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

    double *local_C = (double *)malloc(TILE_SIZE * TILE_SIZE * sizeof(double));
    double *C_full = NULL;
    if (rank == 0)
    {
        C_full = (double *)malloc(matrix_size * matrix_size * sizeof(double));
    }

    MPI_Datatype row_type, block_type;
    MPI_Type_contiguous(TILE_SIZE, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);


    // 计算 sendcounts 和 displacements
int *sendcounts = NULL;
int *displacements = NULL;
    MPI_Type_vector(TILE_SIZE, 1, matrix_size / TILE_SIZE, row_type, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double) * TILE_SIZE, &block_type);
    MPI_Type_commit(&block_type);
if (rank == 0) {
    sendcounts = (int *)malloc(processes * sizeof(int));
    displacements = (int *)malloc(processes * sizeof(int));
    int coords[2];

    for (int i = 0; i < processes; i++) {
        sendcounts[i] = 1;  // 每个进程发送一个数据块
        MPI_Cart_coords(grid_comm, i, 2, coords);
        displacements[i] = coords[0] * matrix_size + coords[1];  // 计算每个进程的位移
    }
}

    // 分发数据块
    scatter_matrix(A, local_A, sendcounts, displacements, block_type, TILE_SIZE, grid_comm);
    scatter_matrix(B, local_B, sendcounts, displacements, block_type, TILE_SIZE, grid_comm);

    // 释放内存和数据类型
    cleanup_resources(sendcounts, displacements, &row_type, &block_type, rank);



    int source, dest;
    source = (grid_coords[0] + 1) % p;
    dest = (grid_coords[0] + p - 1) % p;

    double *temp_A = (double *)malloc(TILE_SIZE * TILE_SIZE * sizeof(double));

    for (int step = 0; step < p; step++)
    {

        int root = (grid_coords[0] + step) % p;

        if (root == grid_coords[1])
        {
            MPI_Bcast(local_A, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, root, row_comm);
            for (int x = 0; x < TILE_SIZE; x++)
            {
                for (int y = 0; y < TILE_SIZE; y++)
                {
                    for (int z = 0; z < TILE_SIZE; z++)
                    {
                        local_C[x * TILE_SIZE + y] += local_A[x * TILE_SIZE + z] * local_B[z * TILE_SIZE + y];
                    }
                }
            }
        }
        else
        {
            MPI_Bcast(temp_A, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, root, row_comm);
            for (int x = 0; x < TILE_SIZE; x++)
            {
                for (int y = 0; y < TILE_SIZE; y++)
                {
                    for (int z = 0; z < TILE_SIZE; z++)
                    {
                        local_C[x * TILE_SIZE + y] += temp_A[x * TILE_SIZE + z] * local_B[z * TILE_SIZE + y];
                    }
                }
            }
        }

        MPI_Sendrecv_replace(local_B, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, dest, 0, source, 0, col_comm, MPI_STATUS_IGNORE);
    }

    free(temp_A);

    MPI_Gather(local_C, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, C_full, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, 0, grid_comm);

    if (rank == 0)
    {
        end_time = MPI_Wtime();

        double *C_fixed = (double *)malloc(matrix_size * matrix_size * sizeof(double));
        for (int i = 0; i < p; i++) {
        for (int j = 0; j < p; j++) {
            int block_start = (i * p + j) * TILE_SIZE * TILE_SIZE;
            int dest_start = (i * TILE_SIZE) * matrix_size + j * TILE_SIZE;
            int src_start = block_start;

            // 调用辅助函数以复制子矩阵块
            copy_matrix_block(C_fixed, C_full, dest_start, src_start, TILE_SIZE, matrix_size, TILE_SIZE, TILE_SIZE);
        }
    }

        printf("Result matrix:\n");
        for (int i = 0; i < matrix_size; i++)
        {
            for (int j = 0; j < matrix_size; j++)
            {
                printf("%6.2f ", C_fixed[i * matrix_size + j]);
            }
            printf("\n");
        }
        printf("\n");
        free(C_fixed);

        if (rank == 0)
        {
            double runtime = end_time - start_time;
            printf("Number of processes: %d, Runtime: %f seconds\n", processes, runtime);
        }
    }

    free(local_A);
    free(local_B);
    free(local_C);
    if (rank == 0)
    {
        free(A);
        free(B);
        free(C_full);
    }

    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&row_comm);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
