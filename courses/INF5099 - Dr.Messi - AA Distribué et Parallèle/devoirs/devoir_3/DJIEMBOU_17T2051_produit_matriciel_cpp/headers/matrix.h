#ifndef MATRIX_H
#define MATRIX_H

#pragma once

#include <iostream>
#include <thread>
#include <random>
#include <mutex>

/** \struct ThreadArgs
 * \brief Storage structure
 * 
 * This structure store all usefull variable needs for theads matrix multiply operation
 */

typedef struct {
    // int thread_id; /** Thread Identifier */
    std::vector<std::vector<double>> A; /** Matrix 2 */
    std::vector<std::vector<double>> B; /** Matrix 1 */
    std::vector<std::vector<double>>* D; /** Result Matrix */
    std::vector<std::vector<double>>* E; /** Result Matrix */
    int cols1_rows2; /** Number of share rows & columns between matrix 1 and 2 */
    int cols2; /** Number of columns of matrix 2 */
    int rows1; /** Number of rows of the matrix 1 */
    int NUM_THREADS; /** Number of thread to use */
} ThreadArgs;

// global instance
extern ThreadArgs _ThreadArgs;

/** \methods matrices_identiques
 * \brief Identic Matrix
 * 
 * This method check whether two matrix are identic
 * 
 * \param A The matrix 1
 * \param B The Matrix 2
 * \param rows The number of rows
 * \param cols The number of columns
 * 
 * \return 0 if they are identic else 1
 */
int matrices_identiques(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B, int rows, int cols);

/** \methods generate_matrix
 * \brief Matrix Generation (Allocation)
 * 
 * This method generate a matrix
 * 
 * \param rows The number of rows
 * \param cols The number of columns
 * 
 * \return The generated matrix
 */
std::vector<std::vector<double>>  generate_matrix(int rows, int cols);

/** \methods block_par_matrix_multiply
 * \brief Thread Matrix multiply
 * 
 * This method create threads which will execute matrix multiply with the block repartition approach
 * 
 * \param A The matrix 1
 * \param B The Matrix 2
 * \param C The Result matrix pointer
 * \param rows The number of rows
 * \param cols The number of columns
 * \param NUM_THREADS The number of thread to use
 * 
 */
void block_par_matrix_multiply(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B, std::vector<std::vector<double>>* C, int cols1_rows2, int cols2, int rows1, int NUM_THREADS);

/** \methods modulo_par_matrix_multiply
 * \brief Thread Matrix multiply
 * 
 * This method create threads which will execute matrix multiply with the modulo repartition approach
 * 
 * \param A The matrix 1
 * \param B The Matrix 2
 * \param C The Result matrix pointer
 * \param rows The number of rows
 * \param cols The number of columns
 * \param NUM_THREADS The number of thread to use
 * 
 */
void modulo_par_matrix_multiply(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B, std::vector<std::vector<double>>* C, int cols1_rows2, int cols2, int rows1, int NUM_THREADS);

/** \methods block_matrix_multiply_thread
 * \brief Thread Matrix multiply executor
 * 
 * This method apply matrix multiply on a block of the matrix 1 knowing the thread argument
 * 
 * \param ThreadArgs The arguments of thread
 * \param mtx The mutex variable used for lock and unlock access to a memoire space
 * 
 */
void block_matrix_multiply_thread(int thread_id);

/** \methods modulo_matrix_multiply_thread
 * \brief Thread Matrix multiply executor
 * 
 * This method apply matrix multiply on some line of the matrix 1 such as the line%num_thread=thread_id knowing the thread argument
 * 
 * \param ThreadArgs The arguments of thread
 * \param mtx The mutex variable used for lock and unlock access to a memoire space
 * 
 */
void modulo_matrix_multiply_thread(int thread_id);

/** \methods matrix_multiply
 * \brief Matrix multiply
 * 
 * This method create threads which will execute matrix multiply with sequential approach
 * 
 * \param A The matrix 1
 * \param B The Matrix 2
 * \param C The Result matrix
 * \param cols1_rows2 The number of common row and col of matrix 1 and 2 respectively
 * \param cols2 The number of columns of matrix 2
 * \param rows1 The number of rows of matrix 1
 * 
 * \return The resulting matrix
 */
std::vector<std::vector<double>> matrix_multiply(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B, std::vector<std::vector<double>> C, int cols1_rows2, int cols2, int rows1);

/** \methods generate_random_matrix
 * \brief Matrix printing
 * 
 * This method print a matrix in console
 * 
 * \param matrix The matrix to print
 * \param cols The number of columns of the matrix
 * \param rows The number of rows of the matrix
 * 
 */
void print_matrix(std::vector<std::vector<double>> matrix, int rows, int cols);

/** \methods matrix_multiply
 * \brief Matrix content generation (content-full)
 * 
 * This method randomly affect value to matrix cells
 * 
 * \param matrix The matrix to print
 * \param cols The number of columns of the matrix
 * \param rows The number of rows of the matrix
 * 
 * \return The filled matrix
 */
std::vector<std::vector<double>> generate_random_matrix(std::vector<std::vector<double>> matrix, int rows, int cols);



#endif