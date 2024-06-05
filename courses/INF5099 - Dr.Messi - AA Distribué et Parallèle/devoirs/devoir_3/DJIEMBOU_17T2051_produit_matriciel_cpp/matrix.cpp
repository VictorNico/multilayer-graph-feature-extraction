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


void block_matrix_multiply_thread(int thread_id) {
    // ThreadArgs* args = (ThreadArgs*)arg;
    // int thread_id = args->thread_id;    
    int cols1_rows2 = _ThreadArgs.cols1_rows2;
    int cols2 = _ThreadArgs.cols2;
    int rows1 = _ThreadArgs.rows1;
    // std::vector<std::vector<double>>* C = &(_ThreadArgs.C);
    int NUM_THREADS = _ThreadArgs.NUM_THREADS;

    int start_row = (thread_id * rows1) / NUM_THREADS;
    int end_row = ((thread_id + 1) * rows1) / NUM_THREADS;

    int i, j, k;
    double sum;
    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < cols2; j++) {
            // cout << "<--i:" <<i<<",j:"<<j << endl;
            (*_ThreadArgs.D)[i][j] = 0.0;
            for (k = 0; k < cols1_rows2; k++) {
                (*_ThreadArgs.D)[i][j] += _ThreadArgs.A[i][k] * _ThreadArgs.B[k][j];
                // cout << "#" <<args->A[i][k] << ">"<< args->B[k][j] << "="<<sum << endl;
            }
            // lock_guard<mutex> lock(mtx);
            // (*args->C)[i][j] = sum;
            // cout << "**"<<sum << "-->" << endl;
        }
    }
}

void modulo_matrix_multiply_thread(int thread_id) {
    // ThreadArgs* args = (ThreadArgs*)arg;
    // int thread_id = args->thread_id;    
    int cols1_rows2 = _ThreadArgs.cols1_rows2;
    int cols2 = _ThreadArgs.cols2;
    int rows1 = _ThreadArgs.rows1;
    // std::vector<std::vector<double>>* C = &(args->C);
    int NUM_THREADS = _ThreadArgs.NUM_THREADS;

    int start_row = (thread_id * rows1) / NUM_THREADS;
    int end_row = ((thread_id + 1) * rows1) / NUM_THREADS;

    int i, j, k;
    double sum;
    for (i = start_row; i < rows1; i+=NUM_THREADS) {
        for (j = 0; j < cols2; j++) {
            // cout << "<--i:" <<i<<",j:"<<j << endl;
            (*_ThreadArgs.E)[i][j] = 0.0;
            for (k = 0; k < cols1_rows2; k++) {
                (*_ThreadArgs.E)[i][j] += _ThreadArgs.A[i][k] * _ThreadArgs.B[k][j];
                // cout << "#" <<args->A[i][k] << ">"<< args->B[k][j] << "="<<sum << endl;
            }
            // lock_guard<mutex> lock(mtx);
            // (*_ThreadArgs.E)[i][j] = sum;
            // cout << "**"<<sum << "-->" << endl;
        }
    }
}

void block_par_matrix_multiply(vector<vector<double>> A, vector<vector<double>> B, vector<vector<double>>* C, int cols1_rows2, int cols2, int rows1, int NUM_THREADS) {
    // thread threads[NUM_THREADS];
    vector<std::thread> threads(NUM_THREADS);
    // vector<ThreadArgs> thread_args(NUM_THREADS);
    // mutex mtx;  // Mutex pour synchroniser l'accès à la matrice résultante
    int i;

    _ThreadArgs.A = A;
    _ThreadArgs.B = B;
    _ThreadArgs.D = C;
    _ThreadArgs.cols1_rows2 = cols1_rows2;
    _ThreadArgs.cols2 = cols2;
    _ThreadArgs.rows1 = rows1;
    _ThreadArgs.NUM_THREADS = NUM_THREADS;

    for (i = 0; i < NUM_THREADS; i++) {
        threads[i] = thread(block_matrix_multiply_thread, i);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void modulo_par_matrix_multiply(vector<vector<double>> A, vector<vector<double>> B, vector<vector<double>>* C, int cols1_rows2, int cols2, int rows1, int NUM_THREADS) {
    // thread threads[NUM_THREADS];
    vector<std::thread> threads(NUM_THREADS);
    // vector<ThreadArgs> thread_args(NUM_THREADS);
    // mutex mtx;  // Mutex pour synchroniser l'accès à la matrice résultante
    int i;

    _ThreadArgs.A = A;
    _ThreadArgs.B = B;
    _ThreadArgs.E = C;
    _ThreadArgs.cols1_rows2 = cols1_rows2;
    _ThreadArgs.cols2 = cols2;
    _ThreadArgs.rows1 = rows1;
    _ThreadArgs.NUM_THREADS = NUM_THREADS;

    for (i = 0; i < NUM_THREADS; i++) {
        threads[i] = thread(modulo_matrix_multiply_thread, i);
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
