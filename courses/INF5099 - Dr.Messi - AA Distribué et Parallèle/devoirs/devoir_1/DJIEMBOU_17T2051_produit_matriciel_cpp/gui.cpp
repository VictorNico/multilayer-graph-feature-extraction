#include <iostream>
#include <ctime>
#include <chrono>

#include "headers/gui.h"
#include "headers/matrix.h"

using namespace std;


bool is_integer(const char *chaine) {
    char *fin;
    strtol(chaine, &fin, 10);
    return !*fin;
}


int get_integer() {
    char input[100];
    int output;

    do {
        cout << "Hit an integer : " <<endl;
        fgets(input, sizeof(input), stdin);

        // delete new line characters when exists
        input[strcspn(input, "\n")] = '\0';

        if (is_integer(input)) {
            output = atoi(input);
        }
    } while (is_integer(input) == false);

    return output;
}


void gui() {
    int ch;
    vector<vector<double>> A,  B, C, D, E;
    int i, j, Ncol1_row2, Nrow1, Ncol2, Nthread;
    unsigned long temps;
    vector<int> stages(3);
    unsigned int numProcessors;

    // Boucle principale
    do {

        // Afficher un message
        cout << "#######################################################################"<<endl;
        cout << "#   0 - Exit"<<endl;
        cout << "#   1 - Setup Matrix Dimension"<<endl;
        cout << "#   2 - Allocate Matrix"<<endl;
        cout << "#   3 - Generate Randomized data inside Matrix"<<endl;
        cout << "#   4 - Block Parallel Execution"<<endl;
        cout << "#   5 - Modulo Parallel Execution"<<endl;
        cout << "#   6 - Sequential Execution"<<endl;
        cout << "#   7 - Block Parallel and Sequential Execution"<<endl;
        cout << "#   8 - Modulo Parallel and Sequential Execution"<<endl;
        cout << "#   9 - Block Parallel and Modulo Parallel Execution"<<endl;
        cout << "#   10 - Block Parallel, Modulo Parallel and Sequential Execution"<<endl;
        cout << "#   11 - Print Matrix"<<endl;

        ch = get_integer();
        switch (ch) {
            case 0:
                cout << "Thanks you for your fidelity. Soonly..."<<endl;
                break;
            case 1:{
                stages[0] = 1;
                // get Matrix 1 cols and matrix 2 rows
                cout << " Matrix Setup ...."<<endl;
                cout << " set Matrix 1 cols and matrix 2 rows"<<endl;
                Ncol1_row2 = get_integer();
                // get Matrix 1 rows
                cout << " set Matrix 1 rows"<<endl;
                Nrow1 = get_integer();
                // get Matrix 2 cols
                cout << " set Matrix 2 cols"<<endl;
                Ncol2 = get_integer();
                // get number of threads
                do {
                    numProcessors = std::thread::hardware_concurrency();
                    cout << " set Number of threads to use ("<< numProcessors << " are available)"<<endl;
                    Nthread = abs(get_integer());
                } while(Nthread <=0 || Nthread>numProcessors);
                break;
            }
            case 2:{
                if (stages[0]){
                    stages[1] = 1;
                    // allocate matrix
                    cout << " Memory allocation of matrix 1 ("<<Nrow1<<","<<Ncol1_row2<<")"<<endl;
                    A = generate_matrix(Nrow1, Ncol1_row2);
                    cout << " Memory allocation of matrix 2 ("<<Ncol1_row2<<","<<Ncol2<<")"<<endl;
                    B = generate_matrix(Ncol1_row2, Ncol2);
                    cout << " Memory allocation of result matrix ("<<Nrow1<<","<<Ncol2<<")"<<endl;
                    C = generate_matrix(Nrow1, Ncol2);
                    D = generate_matrix(Nrow1, Ncol2);
                    E = generate_matrix(Nrow1, Ncol2);
                }
                else {
                    cout << "need 1)" << endl;
                }
                break;
            }
            case 3:{
                if (stages[1]){
                    stages[2] = 1;
                    // rondomly generate matrix A and B
                    cout << " Generate content of matrix 1 as type double ("<< Nrow1<< ","<< Ncol1_row2<<")"<<endl;
                    A = generate_random_matrix(A, Nrow1, Ncol1_row2);
                    cout << " Generate content of matrix 2 as type double ("<< Ncol1_row2<<","<<Ncol2<<")"<<endl;
                    B = generate_random_matrix(B, Ncol1_row2, Ncol2);
                }
                else {
                    cout << "need 2)" << endl;
                }
                break;
            }
            case 4:{
                if (stages[2]){
                    auto start = chrono::high_resolution_clock::now();
                    block_par_matrix_multiply(A, B, &D, Ncol1_row2, Ncol2, Nrow1, Nthread);
                    auto stop = chrono::high_resolution_clock::now();
                    auto elapsed = chrono::duration_cast<chrono::milliseconds>(stop - start).count();
                    cout << "time block par = "<< elapsed<<"ms"<<endl;
                }
                else {
                    cout << "need 3)" << endl;
                }
                break;
            }
            case 5:{
                if (stages[2]){
                    auto start = chrono::high_resolution_clock::now();
                    modulo_par_matrix_multiply(A, B, &E, Ncol1_row2, Ncol2, Nrow1, Nthread);
                    auto stop = chrono::high_resolution_clock::now();
                    auto elapsed = chrono::duration_cast<chrono::milliseconds>(stop - start).count();
                    cout << "time modulo par = "<< elapsed<<"ms"<<endl;
                }
                else {
                    cout << "need 3)" << endl;
                }
                break;
            }
            case 6:{
                if (stages[2]){
                    auto start1 = chrono::high_resolution_clock::now();
                    C = matrix_multiply(A, B, C, Ncol1_row2, Ncol2, Nrow1);
                    auto stop1 = chrono::high_resolution_clock::now();
                    auto elapsed1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1).count();
                    cout << "time seq = "<< elapsed1<<"ms"<<endl;
                }
                else {
                    cout << "need 3)" << endl;
                }
                break;
            }
            case 7:{
                if (stages[2]){
                    auto start2 = chrono::high_resolution_clock::now();
                    block_par_matrix_multiply(A, B, &D, Ncol1_row2, Ncol2, Nrow1, Nthread);
                    auto stop2 = chrono::high_resolution_clock::now();
                    auto elapsed2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2).count();
                    cout << "time block par = "<< elapsed2<<"ms"<<endl;



                    auto start3 = chrono::high_resolution_clock::now();
                    C = matrix_multiply(A, B, C, Ncol1_row2, Ncol2, Nrow1);
                    auto stop3 = chrono::high_resolution_clock::now();
                    auto elapsed3 = chrono::duration_cast<chrono::milliseconds>(stop3 - start3).count();
                    cout << "time seq = "<< elapsed3<<"ms"<<endl;
                }
                else {
                    cout << "need 3)" << endl;
                }
                break;
            }
            case 8:{
                if (stages[2]){
                    auto start2 = chrono::high_resolution_clock::now();
                    modulo_par_matrix_multiply(A, B, &D, Ncol1_row2, Ncol2, Nrow1, Nthread);
                    auto stop2 = chrono::high_resolution_clock::now();
                    auto elapsed2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2).count();
                    cout << "time modulo par = "<< elapsed2<<"ms"<<endl;



                    auto start3 = chrono::high_resolution_clock::now();
                    C = matrix_multiply(A, B, C, Ncol1_row2, Ncol2, Nrow1);
                    auto stop3 = chrono::high_resolution_clock::now();
                    auto elapsed3 = chrono::duration_cast<chrono::milliseconds>(stop3 - start3).count();
                    cout << "time seq = "<< elapsed3<<"ms"<<endl;
                }
                else {
                    cout << "need 3)" << endl;
                }
                break;
            }
            case 9:{
                if (stages[2]){
                    auto start2 = chrono::high_resolution_clock::now();
                    modulo_par_matrix_multiply(A, B, &E, Ncol1_row2, Ncol2, Nrow1, Nthread);
                    auto stop2 = chrono::high_resolution_clock::now();
                    auto elapsed2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2).count();
                    cout << "time modulo par = "<< elapsed2<<"ms"<<endl;



                    auto start3 = chrono::high_resolution_clock::now();
                    block_par_matrix_multiply(A, B, &D, Ncol1_row2, Ncol2, Nrow1, Nthread);
                    auto stop3 = chrono::high_resolution_clock::now();
                    auto elapsed3 = chrono::duration_cast<chrono::milliseconds>(stop3 - start3).count();
                    cout << "time block par = "<< elapsed3<<"ms"<<endl;
                }
                else {
                    cout << "need 3)" << endl;
                }
                break;
            }
            case 10:{
                if (stages[2]){
                    auto start2 = chrono::high_resolution_clock::now();
                    modulo_par_matrix_multiply(A, B, &E, Ncol1_row2, Ncol2, Nrow1, Nthread);
                    auto stop2 = chrono::high_resolution_clock::now();
                    auto elapsed2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2).count();
                    cout << "time modulo par = "<< elapsed2<<"ms"<<endl;



                    auto start3 = chrono::high_resolution_clock::now();
                    block_par_matrix_multiply(A, B, &D, Ncol1_row2, Ncol2, Nrow1, Nthread);
                    auto stop3 = chrono::high_resolution_clock::now();
                    auto elapsed3 = chrono::duration_cast<chrono::milliseconds>(stop3 - start3).count();
                    cout << "time block par = "<< elapsed3<<"ms"<<endl;



                    auto start4 = chrono::high_resolution_clock::now();
                    C = matrix_multiply(A, B, C, Ncol1_row2, Ncol2, Nrow1);
                    auto stop4 = chrono::high_resolution_clock::now();
                    auto elapsed4 = chrono::duration_cast<chrono::milliseconds>(stop4 - start4).count();
                    cout << "time seq = "<< elapsed4<<"ms"<<endl;
                }
                else {
                    cout << "need 3)" << endl;
                }
                break;
            }
            case 11:{
                if (stages[2]){
                    int choose;
                    do {
                        cout << " Matrix Printing ...."<<endl;
                        cout << " 12 - Matrix 1"<<endl;
                        cout << " 13 - Matrix 2"<<endl;
                        cout << " 14 - Result Matrix"<<endl;
                        cout << " 15 - Result identity"<<endl;
                        choose = get_integer();
                    } while(choose <12 || choose >15);
                    switch (choose) {
                        case 12:
                            print_matrix(A, Nrow1, Ncol1_row2);
                            break;
                        case 13:
                            print_matrix(B, Ncol1_row2, Ncol2);
                            break;
                        case 14:
                            cout << " Sequential result"<<endl;
                            print_matrix(C, Nrow1, Ncol2);
                            cout << "\n\n"<<endl;
                            cout << " Block Parallel result"<<endl;
                            print_matrix(D, Nrow1, Ncol2);
                            cout << "\n\n"<<endl;
                            cout << " Modulo Parallel result"<<endl;
                            print_matrix(E, Nrow1, Ncol2);
                            break;
                        case 15:
                            cout << "\n\nR_block_par and R_modulo have a state of identity = "<<matrices_identiques(E, D, Nrow1, Ncol2)<<endl;  
                            cout << "\n\nR_seq and R_modulo have a state of identity = "<<matrices_identiques(E, C, Nrow1, Ncol2)<<endl;  
                            break;
                    }
                }
                else {
                    cout << "need 3)" << endl;
                }
                break;
            }
        }
    } while (ch != 0);
}
