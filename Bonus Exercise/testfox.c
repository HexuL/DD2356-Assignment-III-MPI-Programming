//This file outputs the matrix multiplication serial calculation results and runtime.
//Use to verify the result of fox.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


double* allocate_matrix(int dim) {
    return (double*)malloc(dim * dim * sizeof(double));
}


void multiply_matrices(double* A, double* B, double* C, int matrix_size) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            C[i * matrix_size + j] = 0; 
            for (int k = 0; k < matrix_size; k++) {
                C[i * matrix_size + j] += A[i * matrix_size + k] * B[k * matrix_size + j];
            }
        }
    }
}


void print_matrix(double* matrix, int matrix_size) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            printf("%6.2f ", matrix[i * matrix_size + j]);
        }
        printf("\n");
    }
}


void read_input_matrices(double** A, double** B, int* matrix_size) {
    *matrix_size = 16;  

    *A = allocate_matrix(*matrix_size);
    *B = allocate_matrix(*matrix_size);

    int total_elements = (*matrix_size) * (*matrix_size);
    for (int i = 0; i < total_elements; i++) {
        (*A)[i] = i;  
        (*B)[i] = i;  
    }
}

int main() {
    double *A, *B, *C;
    int matrix_size;

    read_input_matrices(&A, &B, &matrix_size);
    C = allocate_matrix(matrix_size);  

    clock_t start = clock();
    multiply_matrices(A, B, C, matrix_size); 
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;  

    printf("Got matrix size: %d \n",matrix_size);
    printf("Result Matrix C:\n");
    print_matrix(C, matrix_size);
    printf("Run time: %f seconds\n", time_spent);

    
    free(A);
    free(B);
    free(C);

    return 0;
}
