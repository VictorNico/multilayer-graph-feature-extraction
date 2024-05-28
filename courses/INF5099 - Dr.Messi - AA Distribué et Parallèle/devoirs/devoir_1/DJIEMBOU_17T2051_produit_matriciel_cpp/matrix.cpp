#include "headers/matrix.h"


using namespace std;

void matrix_multiply_thread(ThreadArgs*  arg, mutex& mtx);

// matrix generator
vector<vector<double>> generate_random_matrix(vector<vector<double>> matrix, int rows, int cols) {
    int i, j;
    srand(time(nullptr));  // Initialiser le générateur de nombres aléatoires avec une graine basée sur l'heure actuelle

    for (i = 0; i < rows; i++) {
        #ifdef NDEBUG
        cout <<"i:"<<i<<"\t"<<endl;
        #endif
        for (j = 0; j < cols; j++) {
            #ifdef NDEBUG
            cout <<"j:"<<j<<"\n"<<endl;;
            #endif
            matrix[i][j] = (double)rand() / RAND_MAX;  // Générer un nombre réel aléatoire entre 0 et 1
        }
    }
    return matrix;
}

// matrix printing
void print_matrix(vector<vector<double>> matrix, int rows, int cols) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            cout <<matrix[i][j]<<"\t";
        }
        cout <<endl;
    }
}

// sequential matrix multiplication
vector<vector<double>> matrix_multiply(vector<vector<double>> A, vector<vector<double>> B, vector<vector<double>> C, int cols1_rows2, int cols2, int rows1) {
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


void block_matrix_multiply_thread(ThreadArgs*  args, mutex& mtx) {
    // ThreadArgs* args = (ThreadArgs*)arg;
    int thread_id = args->thread_id;    
    int cols1_rows2 = args->cols1_rows2;
    int cols2 = args->cols2;
    int rows1 = args->rows1;
    // std::vector<std::vector<double>>* C = &(args->C);
    int NUM_THREADS = args->NUM_THREADS;

    int start_row = (thread_id * rows1) / NUM_THREADS;
    int end_row = ((thread_id + 1) * rows1) / NUM_THREADS;

    int i, j, k;
    double sum;
    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < cols2; j++) {
            // cout << "<--i:" <<i<<",j:"<<j << endl;
            sum = 0.0;
            for (k = 0; k < cols1_rows2; k++) {
                sum += args->A[i][k] * args->B[k][j];
                // cout << "#" <<args->A[i][k] << ">"<< args->B[k][j] << "="<<sum << endl;
            }
            lock_guard<mutex> lock(mtx);
            (*args->C)[i][j] = sum;
            // cout << "**"<<sum << "-->" << endl;
        }
    }
}

void modulo_matrix_multiply_thread(ThreadArgs*  args, mutex& mtx) {
    // ThreadArgs* args = (ThreadArgs*)arg;
    int thread_id = args->thread_id;    
    int cols1_rows2 = args->cols1_rows2;
    int cols2 = args->cols2;
    int rows1 = args->rows1;
    // std::vector<std::vector<double>>* C = &(args->C);
    int NUM_THREADS = args->NUM_THREADS;

    int start_row = (thread_id * rows1) / NUM_THREADS;
    int end_row = ((thread_id + 1) * rows1) / NUM_THREADS;

    int i, j, k;
    double sum;
    for (i = start_row; i < rows1; i+=NUM_THREADS) {
        for (j = 0; j < cols2; j++) {
            // cout << "<--i:" <<i<<",j:"<<j << endl;
            sum = 0.0;
            for (k = 0; k < cols1_rows2; k++) {
                sum += args->A[i][k] * args->B[k][j];
                // cout << "#" <<args->A[i][k] << ">"<< args->B[k][j] << "="<<sum << endl;
            }
            lock_guard<mutex> lock(mtx);
            (*args->C)[i][j] = sum;
            // cout << "**"<<sum << "-->" << endl;
        }
    }
}

void block_par_matrix_multiply(vector<vector<double>> A, vector<vector<double>> B, vector<vector<double>>* C, int cols1_rows2, int cols2, int rows1, int NUM_THREADS) {
    // thread threads[NUM_THREADS];
    vector<std::thread> threads(NUM_THREADS);
    vector<ThreadArgs> thread_args(NUM_THREADS);
    mutex mtx;  // Mutex pour synchroniser l'accès à la matrice résultante
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
        threads[i] = thread(block_matrix_multiply_thread, &thread_args[i], ref(mtx));
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void modulo_par_matrix_multiply(vector<vector<double>> A, vector<vector<double>> B, vector<vector<double>>* C, int cols1_rows2, int cols2, int rows1, int NUM_THREADS) {
    // thread threads[NUM_THREADS];
    vector<std::thread> threads(NUM_THREADS);
    vector<ThreadArgs> thread_args(NUM_THREADS);
    mutex mtx;  // Mutex pour synchroniser l'accès à la matrice résultante
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
        threads[i] = thread(modulo_matrix_multiply_thread, &thread_args[i], ref(mtx));
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

vector<vector<double>>  generate_matrix(int rows, int cols) {
    // Allocate memory for matrices M
    vector<vector<double>> M(rows, vector<double>(cols));  
    return M;
}



int matrices_identiques(vector<vector<double>> A, vector<vector<double>> B, int rows, int cols) {
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
