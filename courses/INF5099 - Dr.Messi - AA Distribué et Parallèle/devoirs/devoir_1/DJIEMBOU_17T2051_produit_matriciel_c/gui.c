#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>

#include <unistd.h>
#include <pthread.h>


#include "headers/gui.h"
#include "headers/matrix.h"
// #include "headers/timer.h"

/** \methods is_integer
 * \brief Type checking
 * 
 * This check wheter an input chaine can be cast to int
 */
bool is_integer(const char *chaine);

/** \methods get_integer
 * \brief IHM
 * 
 * This get an integer input from user
 */
int get_integer();

typedef struct timezone timezone_t;
typedef struct timeval timeval_t;

timeval_t t1, t2;
timezone_t tz;


static struct timeval _t1, _t2;
static struct timezone _tz;
timeval_t t1, t2;
timezone_t tz;

static unsigned long _temps_residuel = 0;
#define top1() gettimeofday(&_t1, &_tz)
#define top2() gettimeofday(&_t2, &_tz)

void init_cpu_time(void)
{
   top1(); top2();
   _temps_residuel = 1000000L * _t2.tv_sec + _t2.tv_usec -
                     (1000000L * _t1.tv_sec + _t1.tv_usec );
}

unsigned long cpu_time(void) /* retourne des microsecondes */
{
   return 1000000L * _t2.tv_sec + _t2.tv_usec -
           (1000000L * _t1.tv_sec + _t1.tv_usec ) - _temps_residuel;
}






bool is_integer(const char *chaine) {
    char *fin;
    strtol(chaine, &fin, 10);
    return !*fin;
}


int get_integer() {
    char input[100];
    int output;

    do {
        printf("Hit an integer : ");
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
    double** A, ** B, ** C, ** D, ** E;
    int i, j, Ncol1_row2, Nrow1, Ncol2, Nthread;
    unsigned long temps;
    int stage_2=0, stage_3=0;

    // Boucle principale
    do {

        // Afficher un message
        printf("#######################################################################\n");
        printf("#   0 - Exit\n");
        printf("#   1 - Setup Matrix Dimension\n");
        printf("#   2 - Allocate Matrix\n");
        printf("#   3 - Generate Randomized data inside Matrix\n");
        printf("#   4 - Block Parallel Execution\n");
        printf("#   5 - Modulo Parallel Execution\n");
        printf("#   6 - Sequential Execution\n");
        printf("#   7 - Block Parallel and Sequential Execution\n");
        printf("#   8 - Modulo Parallel and Sequential Execution\n");
        printf("#   9 - Block Parallel and Modulo Parallel Execution\n");
        printf("#   10 - Block Parallel, Modulo Parallel and Sequential Execution\n");
        printf("#   11 - Print Matrix\n");

        ch = get_integer();
        switch (ch) {
            case 0:{
                if (stage_2){
                    free_matrix(A, Nrow1); 
                    free_matrix(B, Ncol1_row2); 
                    free_matrix(C, Nrow1); 
                    free_matrix(D, Nrow1); 
                    free_matrix(E, Nrow1); 
                }
                printf("Thanks you for your fidelity. Soonly...\n");
                break;
            }
            case 1:{
                // get Matrix 1 cols and matrix 2 rows
                printf(" Matrix Setup ....\n");
                printf(" set Matrix 1 cols and matrix 2 rows\n");
                Ncol1_row2 = get_integer();
                // get Matrix 1 rows
                printf(" set Matrix 1 rows\n");
                Nrow1 = get_integer();
                // get Matrix 2 cols
                printf(" set Matrix 2 cols\n");
                Ncol2 = get_integer();
                // get number of threads
                int numProcessors = sysconf(_SC_NPROCESSORS_ONLN);
                do {
                    printf(" set Number of threads to use (%d are avaible) \n",numProcessors);
                    Nthread = abs(get_integer());
                } while(Nthread <=0 || Nthread>numProcessors);
                break;
            }
            case 2:{
                // allocate matrix
                printf(" Memory allocation of matrix 1 (%d,%d)\n", Nrow1, Ncol1_row2);
                A = generate_matrix(A, Nrow1, Ncol1_row2);
                printf(" Memory allocation of matrix 2 (%d,%d)\n", Ncol1_row2, Ncol2);
                B = generate_matrix(B, Ncol1_row2, Ncol2);
                printf(" Memory allocation of result matrix (%d,%d)\n",Nrow1, Ncol2);
                C = generate_matrix(C, Nrow1, Ncol2);
                D = generate_matrix(D, Nrow1, Ncol2);
                E = generate_matrix(E, Nrow1, Ncol2);
                stage_2 = 1;
                break;
            }
            case 3:{
                // rondomly generate matrix A and B
                printf(" Generate content of matrix 1 as type double (%d,%d)\n", Nrow1, Ncol1_row2);
                A = generate_random_matrix(A, Nrow1, Ncol1_row2);
                printf(" Generate content of matrix 2 as type double (%d,%d)\n", Ncol1_row2, Ncol2);
                B = generate_random_matrix(B, Ncol1_row2, Ncol2);
                stage_3=1;
                break;
            }
            case 4:{
                top1();
                block_par_matrix_multiply(A, B, D, Ncol1_row2, Ncol2, Nrow1, Nthread);
                top2();
                temps = cpu_time();
                printf("\ntime block par = %ld.%03ldms\n", temps/1000, temps%1000);
                break;
            }
            case 5:{
                top1();
                modulo_par_matrix_multiply(A, B, E, Ncol1_row2, Ncol2, Nrow1, Nthread);
                top2();
                temps = cpu_time();
                printf("\ntime modulo par = %ld.%03ldms\n", temps/1000, temps%1000);
                break;
            }
            case 6:{
                top1();
                C = matrix_multiply(A, B, C, Ncol1_row2, Ncol2, Nrow1);
                top2();
                temps = cpu_time();
                printf("\ntime seq = %ld.%03ldms\n", temps/1000, temps%1000);
                break;
            }
            case 7:{
                top1();
                block_par_matrix_multiply(A, B, D, Ncol1_row2, Ncol2, Nrow1, Nthread);
                top2();
                temps = cpu_time();
                printf("\ntime block par = %ld.%03ldms\n", temps/1000, temps%1000);



                top1();
                C = matrix_multiply(A, B, C, Ncol1_row2, Ncol2, Nrow1);
                top2();
                temps = cpu_time();
                printf("\ntime seq = %ld.%03ldms\n", temps/1000, temps%1000);
                break;
            }
            case 8:{
                top1();
                modulo_par_matrix_multiply(A, B, E, Ncol1_row2, Ncol2, Nrow1, Nthread);
                top2();
                temps = cpu_time();
                printf("\ntime modulo par = %ld.%03ldms\n", temps/1000, temps%1000);



                top1();
                C = matrix_multiply(A, B, C, Ncol1_row2, Ncol2, Nrow1);
                top2();
                temps = cpu_time();
                printf("\ntime seq = %ld.%03ldms\n", temps/1000, temps%1000);
                break;
            }
            case 9:{
                top1();
                block_par_matrix_multiply(A, B, D, Ncol1_row2, Ncol2, Nrow1, Nthread);
                top2();
                temps = cpu_time();
                printf("\ntime block par = %ld.%03ldms\n", temps/1000, temps%1000);



                top1();
                modulo_par_matrix_multiply(A, B, E, Ncol1_row2, Ncol2, Nrow1, Nthread);
                top2();
                temps = cpu_time();
                printf("\ntime modulo par = %ld.%03ldms\n", temps/1000, temps%1000);
                break;
            }
            case 10:{
                top1();
                block_par_matrix_multiply(A, B, D, Ncol1_row2, Ncol2, Nrow1, Nthread);
                top2();
                temps = cpu_time();
                printf("\ntime block par = %ld.%03ldms\n", temps/1000, temps%1000);


                top1();
                modulo_par_matrix_multiply(A, B, E, Ncol1_row2, Ncol2, Nrow1, Nthread);
                top2();
                temps = cpu_time();
                printf("\ntime modulo par = %ld.%03ldms\n", temps/1000, temps%1000);


                top1();
                C = matrix_multiply(A, B, C, Ncol1_row2, Ncol2, Nrow1);
                top2();
                temps = cpu_time();
                printf("\ntime seq = %ld.%03ldms\n", temps/1000, temps%1000);
                break;
            }
            case 11:{
                int choose;
                do {
                    printf(" Matrix Printing ....\n");
                    printf(" 12 - Matrix 1\n");
                    printf(" 13 - Matrix 2\n");
                    printf(" 14 - Result Matrix\n");
                    printf(" 15 - Result Matrix\n");
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
                        printf(" Sequential result\n");
                        print_matrix(C, Nrow1, Ncol2);
                        printf("\n\n");
                        printf(" Block Parallel result\n");
                        print_matrix(D, Nrow1, Ncol2);
                        printf("\n\n");
                        printf(" Modulo Parallel result\n");
                        print_matrix(E, Nrow1, Ncol2);
                        break;
                    case 15:
                        printf("\n\nR_block_par and R_modulo have a state of identity =  %d\n",matrices_identiques(E, D, Nrow1, Ncol2));  
                        printf("\n\nR_seq and R_modulo have a state of identity = %d\n",matrices_identiques(C, E, Nrow1, Ncol2));  
                        break;
                }
                break;
            }
        }
    } while (ch != 0);
}
