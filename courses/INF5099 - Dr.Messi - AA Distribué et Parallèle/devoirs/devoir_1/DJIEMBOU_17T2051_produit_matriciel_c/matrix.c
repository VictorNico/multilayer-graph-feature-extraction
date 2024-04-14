#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <assert.h>

#include "headers/matrix.h"

// matrix generator
double** generate_random_matrix(double** matrix, int rows, int cols) {
    int i, j;
    srand(time(NULL));  // Initialiser le générateur de nombres aléatoires avec une graine basée sur l'heure actuelle

    for (i = 0; i < rows; i++) {
        #ifdef NDEBUG
        printf("i:%d\t",i);
        #endif
        for (j = 0; j < cols; j++) {
            #ifdef NDEBUG
            printf("j:%d \n",j);
            #endif
            matrix[i][j] = (double)rand() / RAND_MAX;  // Générer un nombre réel aléatoire entre 0 et 1
        }
    }
    return matrix;
}

// matrix printing
void print_matrix(double** matrix, int rows, int cols) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
}

// sequential matrix multiplication
double** matrix_multiply(double **A, double **B, double **C, int cols1_rows2, int cols2, int rows1) {
    int i, j, k;

    for (i = 0; i < rows1; i++) {
        for (j = 0; j < cols2; j++) {
            C[i][j] = 0;
            for (k = 0; k < cols1_rows2; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}


void* modulo_matrix_multiply_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int thread_id = args->thread_id;
    double** A = args->A;
    double** B = args->B;
    double** C = args->C;
    int cols1_rows2 = args->cols1_rows2;
    int cols2 = args->cols2;
    int rows1 = args->rows1;
    int NUM_THREADS = args->NUM_THREADS;

    int start_row = (thread_id * rows1) / NUM_THREADS;

    int i, j, k;
    for (i = start_row; i < rows1; i+=NUM_THREADS) {
        for (j = 0; j < cols2; j++) {
            C[i][j] = 0;
            for (k = 0; k < cols1_rows2; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    pthread_exit(NULL);
}

void* block_matrix_multiply_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int thread_id = args->thread_id;
    double** A = args->A;
    double** B = args->B;
    double** C = args->C;
    int cols1_rows2 = args->cols1_rows2;
    int cols2 = args->cols2;
    int rows1 = args->rows1;
    int NUM_THREADS = args->NUM_THREADS;

    int start_row = (thread_id * rows1) / NUM_THREADS;
    int end_row = ((thread_id + 1) * rows1) / NUM_THREADS;

    int i, j, k;
    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < cols2; j++) {
            C[i][j] = 0;
            for (k = 0; k < cols1_rows2; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    pthread_exit(NULL);
}

void modulo_par_matrix_multiply(double** A, double** B, double** C, int cols1_rows2, int cols2, int rows1, int NUM_THREADS) {
    pthread_t threads[NUM_THREADS];
    ThreadArgs thread_args[NUM_THREADS];
    int i;

    for (i = 0; i < NUM_THREADS; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].A = A;
        thread_args[i].B = B;
        thread_args[i].C = C;
        thread_args[i].cols1_rows2 = cols1_rows2;
        thread_args[i].cols2 = cols2;
        thread_args[i].rows1 = rows1;
        thread_args[i].NUM_THREADS = NUM_THREADS;
        pthread_create(&threads[i], NULL, modulo_matrix_multiply_thread, (void*)&thread_args[i]);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}

void block_par_matrix_multiply(double** A, double** B, double** C, int cols1_rows2, int cols2, int rows1, int NUM_THREADS) {
    pthread_t threads[NUM_THREADS];
    ThreadArgs thread_args[NUM_THREADS];
    int i;

    for (i = 0; i < NUM_THREADS; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].A = A;
        thread_args[i].B = B;
        thread_args[i].C = C;
        thread_args[i].cols1_rows2 = cols1_rows2;
        thread_args[i].cols2 = cols2;
        thread_args[i].rows1 = rows1;
        thread_args[i].NUM_THREADS = NUM_THREADS;
        pthread_create(&threads[i], NULL, block_matrix_multiply_thread, (void*)&thread_args[i]);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}

double**  generate_matrix(double** M, int rows, int cols) {
    int i;
    // Allocate memory for matrices M
    M = (double**)malloc(rows * sizeof(double*));
    for (i = 0; i < rows; i++) {
        #ifdef NDEBUG
        printf("i:%d \n",i);
        #endif
        M[i] = (double*)malloc(cols * sizeof(double));    
    }
    return M;
}

void free_matrix(double** M, int rows) {
    int i;
    // Free memory
    for (i = 0; i < rows; i++) {
        #ifdef NDEBUG
        printf("i:%d \n",i);
        #endif
        free(M[i]);
    }
    free(M);
}


int matrices_identiques(double** A, double** B, int rows, int cols) {
    int flag = 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (A[i][j] != B[i][j]) {
                flag = 0;
            }
        }
    }
    return flag;
}
