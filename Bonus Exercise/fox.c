#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h> 

#define TILE_SIZE (matrix_size / p)  

void print_matrix(double *matrix, int matrix_size) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            printf("%6.2f ", matrix[i * matrix_size + j]);
        }
        printf("\n");
    }
}

double* allocate_matrix(int dim) {
    double* matrix = (double*)malloc(dim * dim * sizeof(double));
    return matrix;
}

void multiply_accumalate(double* A, double* B, double* C, int size) {
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            for (int z = 0; z < size; z++) {
                C[x * size + y] += A[x * size + z] * B[z * size + y];
            }
        }
    }
}

void distribute_blocks(double* A, double* B, double* local_A, double* local_B, int matrix_size, int rank, int processes, int block_size, MPI_Comm grid_comm) {

    MPI_Datatype row_type, block_type;

    MPI_Type_contiguous(block_size, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);
    MPI_Type_vector(block_size, 1, matrix_size/block_size, row_type, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double) * block_size, &block_type);
    MPI_Type_commit(&block_type);

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

  

    MPI_Scatterv(A, sendcounts, displacements, block_type, local_A, block_size * block_size, MPI_DOUBLE, 0, grid_comm);
    MPI_Scatterv(B, sendcounts, displacements, block_type, local_B, block_size * block_size, MPI_DOUBLE, 0, grid_comm);

    if (rank == 0) {
        free(sendcounts);
        free(displacements);
    }

    MPI_Type_free(&row_type);
    MPI_Type_free(&block_type);
}

void gather_results(double *local_C, double *C_full, int tile_size, MPI_Comm grid_comm) {
    MPI_Gather(local_C, tile_size * tile_size, MPI_DOUBLE, C_full, tile_size * tile_size, MPI_DOUBLE, 0, grid_comm);
}

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

    double* temp_A = allocate_matrix(*matrix_size);
    double* temp_B = allocate_matrix(*matrix_size);

    for (int i = 0; i < (*matrix_size) * (*matrix_size); i++) {
        temp_A[i] = A_data[i];
        temp_B[i] = B_data[i];
    }

    *A = temp_A;
    *B = temp_B;
}


void read_expected_matrix(double** matrix, int matrix_size) {
    double expected_data[] = {
        56999.00, 40567.00, 35559.00, 56683.00, 40947.00, 46013.00, 51271.00, 50790.00, 
        50521.00, 49337.00, 32745.00, 49098.00, 49949.00, 45466.00, 52601.00, 31969.00, 
        55025.00, 37195.00, 39710.00, 55812.00, 42287.00, 47013.00, 48510.00, 51470.00, 
        48437.00, 54562.00, 39112.00, 44049.00, 47592.00, 45148.00, 56749.00, 37963.00, 
        46987.00, 32072.00, 24392.00, 39544.00, 23651.00, 35707.00, 36347.00, 37640.00, 
        39564.00, 40925.00, 29994.00, 38676.00, 47243.00, 36159.00, 41392.00, 27915.00, 
        34080.00, 20369.00, 24099.00, 30494.00, 18749.00, 26097.00, 30662.00, 26950.00, 
        25216.00, 31424.00, 24972.00, 28141.00, 31322.00, 29769.00, 28671.00, 24455.00, 
        42767.00, 30723.00, 32263.00, 41748.00, 21368.00, 33505.00, 42211.00, 37125.00, 
        35376.00, 38665.00, 33620.00, 48223.00, 38757.00, 41288.00, 42374.00, 26218.00, 
        51601.00, 32507.00, 34410.00, 47937.00, 24998.00, 43024.00, 46661.00, 41143.00, 
        37931.00, 45958.00, 36943.00, 49689.00, 46367.00, 44190.00, 49371.00, 31150.00, 
        56291.00, 35977.00, 39475.00, 51698.00, 34309.00, 45748.00, 50545.00, 47114.00, 
        42681.00, 45856.00, 35861.00, 50600.00, 49487.00, 50956.00, 45889.00, 36731.00, 
        43163.00, 33916.00, 31950.00, 39663.00, 33885.00, 40082.00, 36735.00, 43371.00, 
        37926.00, 40293.00, 31409.00, 33313.00, 43072.00, 33664.00, 50591.00, 32498.00, 
        55043.00, 39378.00, 32619.00, 55809.00, 35580.00, 41201.00, 51230.00, 48569.00, 
        51053.00, 50639.00, 38218.00, 49698.00, 54182.00, 45903.00, 45549.00, 37084.00, 
        38706.00, 27424.00, 21061.00, 35574.00, 24459.00, 34986.00, 34775.00, 32794.00, 
        35013.00, 37018.00, 25663.00, 30980.00, 34707.00, 29153.00, 40424.00, 22266.00, 
        52508.00, 32233.00, 27123.00, 48952.00, 32645.00, 40772.00, 47420.00, 41384.00, 
        42881.00, 44356.00, 34812.00, 47916.00, 44063.00, 37512.00, 44294.00, 27024.00, 
        37554.00, 32080.00, 32170.00, 43735.00, 31198.00, 37863.00, 37102.00, 42155.00, 
        35038.00, 37238.00, 30587.00, 46014.00, 39315.00, 36599.00, 45322.00, 28886.00, 
        53600.00, 39234.00, 38092.00, 46348.00, 37093.00, 45990.00, 41282.00, 46374.00, 
        41897.00, 41124.00, 32474.00, 42140.00, 43623.00, 45282.00, 54866.00, 33407.00, 
        51512.00, 33569.00, 32928.00, 52204.00, 34455.00, 46566.00, 45143.00, 44771.00, 
        42353.00, 45930.00, 34793.00, 46454.00, 47237.00, 46512.00, 40568.00, 33763.00, 
        46783.00, 38762.00, 30661.00, 41375.00, 31971.00, 41626.00, 41666.00, 44527.00, 
        47171.00, 42202.00, 28245.00, 38800.00, 55165.00, 46268.00, 45027.00, 33417.00, 
        36968.00, 25998.00, 26521.00, 36393.00, 30583.00, 34735.00, 31503.00, 35119.00, 
        30154.00, 32983.00, 24655.00, 28920.00, 31681.00, 29431.00, 36871.00, 26455.00
    };

    double* temp_matrix = allocate_matrix(matrix_size);

    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            temp_matrix[i * matrix_size + j] = expected_data[i * matrix_size + j];
        }
    }

    *matrix = temp_matrix;
}


bool compare_matrices(double* calculated_matrix, double* expected_matrix, int matrix_size, double tolerance) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            if (fabs(calculated_matrix[i * matrix_size + j] - expected_matrix[i * matrix_size + j]) > tolerance) {
                return false;
            }
        }
    }
    return true; 
}

void test_matrix_corectness(double* calculated_matrix, int matrix_size) {
    double* expected_matrix;

    read_expected_matrix(&expected_matrix, matrix_size);
    double tolerance = 1e-6;
    bool matrices_match = compare_matrices(calculated_matrix, expected_matrix, matrix_size, tolerance);

    if (matrices_match) {
        printf("\n[TEST PASS] The calculated matrix matches the expected matrix.\n");

    } else {
        printf("\n[TEST FAIL] The calculated matrix does not match the expected matrix.\nExpected matrix:\n");
        print_matrix(expected_matrix, matrix_size);
    }

    free(expected_matrix);
}


void print_result_matrix(double *matrix, int matrix_size) {
    printf("Result matrix:\n");
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            printf("%6.2f ", matrix[i * matrix_size + j]);
        }
        printf("\n");
    }
    printf("\n");
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
        // quit
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

    local_A = allocate_matrix(TILE_SIZE);
    local_B = allocate_matrix(TILE_SIZE);

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

    double* local_C = allocate_matrix(TILE_SIZE);
    double* C_full = NULL;
    if (rank == 0) {
        C_full = allocate_matrix(matrix_size);
    }

    distribute_blocks(A, B, local_A, local_B, matrix_size, rank, processes, TILE_SIZE, grid_comm);

    int source, dest;
    source = (grid_coords[0] + 1) % p;
    dest = (grid_coords[0] + p - 1) % p;

    double* temp_A = allocate_matrix(TILE_SIZE);

    for (int step = 0; step < p; step++) {

        int root = (grid_coords[0] + step) % p;

        if (root == grid_coords[1]) {
            MPI_Bcast(local_A, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, root, row_comm);
            multiply_accumalate(local_A, local_B, local_C, TILE_SIZE);
        }
        else {
            MPI_Bcast(temp_A, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, root, row_comm);
            multiply_accumalate(temp_A, local_B, local_C, TILE_SIZE);
        }

        MPI_Sendrecv_replace(local_B, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, dest, 0, source, 0, col_comm, MPI_STATUS_IGNORE);
    }

    free(temp_A);

    gather_results(local_C, C_full, TILE_SIZE, grid_comm);

    if (rank == 0) {
        end_time = MPI_Wtime();

        double* C_fixed = allocate_matrix(matrix_size);
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                int block_start = (i * p + j) * TILE_SIZE * TILE_SIZE;
                for (int k = 0; k < TILE_SIZE; k++) {
                    memcpy(&C_fixed[(i * TILE_SIZE + k) * matrix_size + j * TILE_SIZE], &C_full[block_start + k * TILE_SIZE], TILE_SIZE * sizeof(double));
                }
            }
        }

        print_result_matrix(C_fixed, matrix_size); 
        test_matrix_corectness(C_fixed, matrix_size);
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


