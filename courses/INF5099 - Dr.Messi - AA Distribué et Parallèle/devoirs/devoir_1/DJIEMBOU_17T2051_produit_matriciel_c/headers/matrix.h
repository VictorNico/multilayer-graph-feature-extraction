#ifndef MATRIX_H
#define MATRIX_H

/** \struct ThreadArgs
 * \brief Storage structure
 * 
 * This structure store all usefull variable needs for theads matrix multiply operation
 */
typedef struct {
    int thread_id; /** Thread Identifier */
    double** A; /** Matrix 2 */
    double** B; /** Matrix 1 */
    double** C; /** Result Matrix */
    int cols1_rows2; /** Number of share rows & columns between matrix 1 and 2 */
    int cols2; /** Number of columns of matrix 2 */
    int rows1; /** Number of rows of the matrix 1 */
    int NUM_THREADS; /** Number of thread to use */
} ThreadArgs;

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
double**  generate_random_matrix(double** matrix, int rows, int cols);

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
void print_matrix(double** matrix, int rows, int cols);

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
double** matrix_multiply(double **A, double **B, double **C, int cols1_rows2, int cols2, int rows1);

/** \methods modulo_matrix_multiply_thread
 * \brief Thread Matrix multiply executor
 * 
 * This method apply matrix multiply on some line of the matrix 1 such as the line%num_thread=thread_id knowing the thread argument
 * 
 * \param arg The arguments of thread
 * 
 */
void* modulo_matrix_multiply_thread(void* arg);

/** \methods block_matrix_multiply_thread
 * \brief Thread Matrix multiply executor
 * 
 * This method apply matrix multiply on a block of the matrix 1 knowing the thread argument
 * 
 * \param arg The arguments of thread
 * 
 */
void* block_matrix_multiply_thread(void* arg);

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
void modulo_par_matrix_multiply(double** A, double** B, double** C, int cols1_rows2, int cols2, int rows1, int NUM_THREADS);

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
void block_par_matrix_multiply(double** A, double** B, double** C, int cols1_rows2, int cols2, int rows1, int NUM_THREADS);

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
double** generate_matrix(double** M, int rows, int cols);


/** \methods free_matrix
 * \brief Memory Management
 * 
 * This method free allocate memory matrix
 * 
 * \param M The matrix to free memory
 * \param rows The number of rows
 * 
 */
void free_matrix(double** M, int rows);

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
int matrices_identiques(double** A, double** B, int cols, int rows);

#endif