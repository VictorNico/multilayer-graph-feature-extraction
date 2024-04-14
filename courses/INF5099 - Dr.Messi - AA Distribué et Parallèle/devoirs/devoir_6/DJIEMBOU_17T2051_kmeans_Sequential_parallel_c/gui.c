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
#include "headers/kmeans.h"
#include "headers/file.h"
#include "headers/time.h"

/** \methods is_integer
 * \brief Type checking
 * 
 * This check wheter an input chaine can be cast to int
 */
bool is_integer(const char *chaine);
bool is_double(const char *chaine);

/** \methods get_integer
 * \brief IHM
 * 
 * This get an integer input from user
 */
int get_integer();
int get_double();

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
bool is_double(const char *chaine) {
    char *fin;
    strtod(chaine, &fin);
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

int get_double() {
    char input[100];
    int output;

    do {
        printf("Hit a double : ");
        fgets(input, sizeof(input), stdin);

        // delete new line characters when exists
        input[strcspn(input, "\n")] = '\0';

        if (is_double(input)) {
            output = strtod(input, NULL);
        }
    } while (is_double(input) == false);

    return output;
}


void gui() {
    int ch;
    double** A, ** B, ** C, ** D, ** E;
    int i, j, Ncol1_row2, Nrow1, Ncol2, Nthread;
    unsigned long temps, temps2;
    int stage_2=0, stage_3=0;

    // const char* currentDate = getCurrentDate();
    // const char* filename = "result_bloc_modulo"+currentDate+".csv";
    const char* path = "outputs/result_bloc_modulo.csv";
    const char* path_1 = "outputs/result_bloc_seq.csv";
    const char* path_2 = "outputs/result_seq_modulo.csv";
    const char* line = "lines,bloc_time,modulo_time,n_thread";
    const char* line_1 = "lines,bloc_time,sec_time";
    const char* line_2 = "lines,sec_time,modulo_time";


    if (fileExists(path)) {
        printf("%s exists\n",path);
    } else {
        write_csv(line, path);
    }
    if (fileExists(path_1)) {
        printf("%s exists\n",path_1);
    } else {
        write_csv(line_1, path_1);
    }
    if (fileExists(path_2)) {
        printf("%s exists\n",path_2);
    } else {
        write_csv(line_2, path_2);
    }

    // Boucle principale
    do {

        // Afficher un message
        printf("#######################################################################\n");
        printf("#   0 - Exit\n");
        printf("#   1 - Setup kmeans hyperparameters\n");
        printf("#   2 - Allocate vectors\n");
        printf("#   3 - Generate Randomized data point\n");
        printf("#   4 - Block Parallel Execution\n");
        printf("#   5 - Modulo Parallel Execution\n");
        printf("#   6 - Sequential Execution\n");
        printf("#   7 - Block Parallel and Sequential Execution\n");
        printf("#   8 - Modulo Parallel and Sequential Execution\n");
        printf("#   9 - Block Parallel and Modulo Parallel Execution\n");
        printf("#   10 - Block Parallel, Modulo Parallel and Sequential Execution\n");
        printf("#   11 - Print clusters\n");

        ch = get_integer();
        switch (ch) {
            case 0:{
                if (stage_2){
                    free_structure();
                }
                printf("Thanks you for your fidelity. Soonly...\n");
                break;
            }
            case 1:{
                printf(" Kmeans hyperparameters Setup ....\n");
                do {
                    printf(" set the points dimension between 2 and k \n");
                    _ThreadArgs.dim = abs(get_integer());
                } while(_ThreadArgs.dim <=1);
                printf(" set length of data points\n");
                _ThreadArgs.dataSize = get_integer();

                printf(" set number of targeted clusters\n");
                _ThreadArgs.k = get_integer();

                printf(" set Maximum iteration threshold\n");
                _ThreadArgs.max_iter = get_integer();

                printf(" set convergence tolerance\n");
                _ThreadArgs.tolerance = get_double();
                // get number of threads
                int numProcessors = sysconf(_SC_NPROCESSORS_ONLN);
                do {
                    printf(" set Number of threads to use (%d are avaible) \n",numProcessors);
                    _ThreadArgs.NUM_THREADS = abs(get_integer());
                } while(_ThreadArgs.NUM_THREADS <=0 || _ThreadArgs.NUM_THREADS>numProcessors);
                break;
            }
            case 2:{
                // allocate matrix
                printf(" Memory allocation of data points (%d)\n", _ThreadArgs.dataSize);
                // Données d'entrée
                _ThreadArgs.points = generate_points(_ThreadArgs.dataSize);
                

                printf(" Memory allocation of centroids points (%d)\n", _ThreadArgs.dataSize);
                _ThreadArgs.centroids = generate_points(_ThreadArgs.k);
                
                _ThreadArgs.centroids_b = generate_points(_ThreadArgs.k);
                
                _ThreadArgs.centroids_m = generate_points(_ThreadArgs.k);
                
                printf(" Memory allocation of result matrix (%d)\n",_ThreadArgs.dataSize);
                _ThreadArgs.assignments = malloc(_ThreadArgs.dataSize * sizeof(int));
                _ThreadArgs.assignments_m = malloc(_ThreadArgs.dataSize * sizeof(int));
                _ThreadArgs.assignments_b = malloc(_ThreadArgs.dataSize * sizeof(int));
                stage_2 = 1;
                stage_3 = 0;
                break;
            }
            case 3:{
                // rondomly generate matrix A and B
                printf(" Generate content of data points and centroids (%d)\n", _ThreadArgs.dataSize);
                generate_random_points(_ThreadArgs.dataSize, _ThreadArgs.dim, _ThreadArgs.points);

                generate_random_centroids(_ThreadArgs.k, _ThreadArgs.dataSize, _ThreadArgs.dim, _ThreadArgs.centroids, _ThreadArgs.points);
                
                for (int i = 0; i<_ThreadArgs.k; i++) {
                    _ThreadArgs.centroids_b[i].dim = malloc(_ThreadArgs.dataSize * sizeof(double));
                    _ThreadArgs.centroids_m[i].dim = malloc(_ThreadArgs.dataSize * sizeof(double));
                    for (int j = 0; j < _ThreadArgs.dataSize; j++) {
                        _ThreadArgs.centroids_b[i].dim[j] = _ThreadArgs.centroids[i].dim[j];
                        _ThreadArgs.centroids_m[i].dim[j] = _ThreadArgs.centroids[i].dim[j];
                    }
                }
                // generate_random_centroids(_ThreadArgs.k, _ThreadArgs.dataSize, _ThreadArgs.dim, _ThreadArgs.centroids_b, _ThreadArgs.points);
                // generate_random_centroids(_ThreadArgs.k, _ThreadArgs.dataSize, _ThreadArgs.dim, _ThreadArgs.centroids_m, _ThreadArgs.points);
    
                stage_3=1;
                break;
            }
            case 4:{
                top1();
                kmeansParallel_B();
                top2();
                temps = cpu_time();
                printf("\ntime block par = %ld.%03ldms\n", temps/1000, temps%1000);
                break;
            }
            case 5:{
                top1();
                kmeansParallel_M();
                top2();
                temps = cpu_time();
                printf("\ntime modulo par = %ld.%03ldms\n", temps/1000, temps%1000);
                break;
            }
            case 6:{
                top1();
                kmeansSequential();
                top2();
                temps = cpu_time();
                printf("\ntime seq = %ld.%03ldms\n", temps/1000, temps%1000);
                break;
            }
            case 7:{
                top1();
                kmeansParallel_B();
                top2();
                temps = cpu_time();
                printf("\ntime block par = %ld.%03ldms\n", temps/1000, temps%1000);



                top1();
                kmeansSequential();
                top2();
                temps2 = cpu_time();
                printf("\ntime seq = %ld.%03ldms\n", temps2/1000, temps2%1000);

                // char line[100];
                // snprintf(line, sizeof(line), "%d,%ld.%03ld,%ld.%03ld", Ncol1_row2, temps/1000, temps%1000 , temps2/1000, temps2%1000);

                // write_csv(line, path_1);
                break;
            }
            case 8:{
                top1();
                kmeansParallel_M();
                top2();
                temps = cpu_time();
                printf("\ntime modulo par = %ld.%03ldms\n", temps/1000, temps%1000);



                top1();
                kmeansSequential();
                top2();
                temps2 = cpu_time();
                printf("\ntime seq = %ld.%03ldms\n", temps2/1000, temps2%1000);

                // char line[100];
                // snprintf(line, sizeof(line), "%d,%ld.%03ld,%ld.%03ld", Ncol1_row2, temps2/1000, temps2%1000 , temps/1000, temps%1000);

                // write_csv(line, path_2);
                break;
            }
            case 9:{
                top1();
                printf("ddd");
                kmeansParallel_B();
                top2();
                temps2 = cpu_time();
                printf("\ntime block par = %ld.%03ldms\n", temps2/1000, temps2%1000);



                top1();
                kmeansParallel_M();
                top2();
                temps = cpu_time();
                printf("\ntime modulo par = %ld.%03ldms\n", temps/1000, temps%1000);

                // char line[100];
                // snprintf(line, sizeof(line), "%d,%ld.%03ld,%ld.%03ld,%d", Ncol1_row2, temps/1000, temps%1000 , temps2/1000, temps2%1000,Nthread);

                // write_csv(line, path);
                break;
            }
            case 10:{
                top1();
                kmeansParallel_B();
                top2();
                temps = cpu_time();
                printf("\ntime block par = %ld.%03ldms\n", temps/1000, temps%1000);


                top1();
                kmeansParallel_M();
                top2();
                temps = cpu_time();
                printf("\ntime modulo par = %ld.%03ldms\n", temps/1000, temps%1000);


                top1();
                kmeansSequential();
                top2();
                temps = cpu_time();
                printf("\ntime seq = %ld.%03ldms\n", temps/1000, temps%1000);
                break;
            }
            case 11:{
                int choose;
                do {
                    printf(" Clusters Printing ....\n");
                    printf(" 12 - Seq result\n");
                    printf(" 13 - Bloc result\n");
                    printf(" 14 - Modulo result\n");
                    choose = get_integer();
                } while(choose <12 || choose >14);
                switch (choose) {
                    case 12:
                        print_cluster(_ThreadArgs.points, _ThreadArgs.centroids, _ThreadArgs.assignments, _ThreadArgs.dataSize, _ThreadArgs.k, "seq", _ThreadArgs.dim);
                        break;
                    case 13:
                        print_cluster(_ThreadArgs.points, _ThreadArgs.centroids_b, _ThreadArgs.assignments_b, _ThreadArgs.dataSize, _ThreadArgs.k, "bloc", _ThreadArgs.dim);
                        break;
                    case 14:
                        print_cluster(_ThreadArgs.points, _ThreadArgs.centroids_m, _ThreadArgs.assignments_m, _ThreadArgs.dataSize, _ThreadArgs.k, "modulo", _ThreadArgs.dim);
                        break;
                }
                break;
            }
        }
    } while (ch != 0);
}
