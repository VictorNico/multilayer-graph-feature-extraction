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
    unsigned long temps1, temps2, temps3, temps4;
    const char* path = "outputs/results.csv";
    const char* line = "lines,bloc_time,modulo_time,map_reduce,seq_time,n_thread";


    if (fileExists(path)) {
        printf("%s exists\n",path);
    } else {
        write_csv(line, path);
    }

    // allocate matrix
    printf(" Memory allocation of data points (%d)\n", _ThreadArgs.dataSize);
    // Données d'entrée
    _ThreadArgs.points = generate_points(_ThreadArgs.dataSize);
    

    printf(" Memory allocation of centroids points (%d)\n", _ThreadArgs.dataSize);
    _ThreadArgs.centroids = generate_points(_ThreadArgs.k);
    
    _ThreadArgs.centroids_b = generate_points(_ThreadArgs.k);
    
    _ThreadArgs.centroids_m = generate_points(_ThreadArgs.k);

    _ThreadArgs.centroids_mr = generate_points(_ThreadArgs.k);
    
    printf(" Memory allocation of result matrix (%d)\n",_ThreadArgs.dataSize);
    _ThreadArgs.assignments = malloc(_ThreadArgs.dataSize * sizeof(int));
    memset(_ThreadArgs.assignments, 0, _ThreadArgs.dataSize * sizeof(int));
    _ThreadArgs.assignments_m = malloc(_ThreadArgs.dataSize * sizeof(int));
    memset(_ThreadArgs.assignments_m, 0, _ThreadArgs.dataSize * sizeof(int));
    _ThreadArgs.assignments_mr = malloc(_ThreadArgs.dataSize * sizeof(int));
    memset(_ThreadArgs.assignments_mr, 0, _ThreadArgs.dataSize * sizeof(int));
    _ThreadArgs.assignments_b = malloc(_ThreadArgs.dataSize * sizeof(int));
    memset(_ThreadArgs.assignments_b, 0, _ThreadArgs.dataSize * sizeof(int));

    // local cluster
    _clusters = malloc(_ThreadArgs.k * sizeof(Cluster));
    // initialSize = 2;

    for (int i = 0; i < _ThreadArgs.k; i++){
        _clusters[i].pointSize = 0;
        _clusters[i].points = malloc(_ThreadArgs.dataSize*sizeof(int));
        memset(_clusters[i].points, 0, _ThreadArgs.dataSize * sizeof(int));
    }
    
    generate_random_points(_ThreadArgs.dataSize, _ThreadArgs.dim, _ThreadArgs.points);

    generate_random_centroids(_ThreadArgs.k, _ThreadArgs.dataSize, _ThreadArgs.dim, _ThreadArgs.centroids, _ThreadArgs.points);
    
    for (int i = 0; i<_ThreadArgs.k; i++) {
        _ThreadArgs.centroids_b[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
        _ThreadArgs.centroids_m[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
        _ThreadArgs.centroids_mr[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
        for (int j = 0; j < _ThreadArgs.dim; j++) {
            _ThreadArgs.centroids_b[i].dim[j] = (double)_ThreadArgs.centroids[i].dim[j];
            _ThreadArgs.centroids_m[i].dim[j] = (double)_ThreadArgs.centroids[i].dim[j];
            _ThreadArgs.centroids_mr[i].dim[j] = (double)_ThreadArgs.centroids[i].dim[j];

        }
    }

    top1();
    kmeansParallel_M();
    top2();
    temps1 = cpu_time();
    printf("\ntime modulo par = %ld.%03ldms\n", temps1/1000, temps1%1000);
    
    top1();
    kmeansParallel_B();
    top2();
    temps2 = cpu_time();
    printf("\ntime block par = %ld.%03ldms\n", temps2/1000, temps2%1000);

    top1();
    kmeansParallel_MR();
    top2();
    temps3 = cpu_time();
    printf("\ntime map reduce par = %ld.%03ldms\n", temps3/1000, temps3%1000);

    top1();
    kmeansSequential();
    top2();
    temps4 = cpu_time();
    printf("\ntime seq = %ld.%03ldms\n", temps4/1000, temps4%1000);

    char line1[1000];
    snprintf(line1, sizeof(line1), "%d,%ld.%03ld,%ld.%03ld,%ld.%03ld,%ld.%03ld,%d",  _ThreadArgs.dataSize, temps1/1000, temps1%1000 , temps2/1000, temps2%1000, temps3/1000, temps3%1000, temps4/1000, temps4%1000, _ThreadArgs.k);

    write_csv(line1, path);

    // int flag = 1;
    // for (int i = 0; i<_ThreadArgs.k; i++) {
    //     for (int j = 0; j < _ThreadArgs.dim; j++) {
    //         if (_ThreadArgs.centroids[i].dim[j] != _ThreadArgs.centroids_m[i].dim[j])
    //             flag = 0;
    //     }
    // }
    // printf("s & m = %d\n",flag);
    // flag = 1;
    // for (int i = 0; i<_ThreadArgs.k; i++) {
    //     for (int j = 0; j < _ThreadArgs.dim; j++) {
    //         if (_ThreadArgs.centroids[i].dim[j] != _ThreadArgs.centroids_mr[i].dim[j])
    //             flag = 0;
    //     }
    // }
    // printf("s & mr = %d\n",flag);
    // flag = 1;
    // for (int i = 0; i<_ThreadArgs.k; i++) {
    //     for (int j = 0; j < _ThreadArgs.dim; j++) {
    //         if (_ThreadArgs.centroids_b[i].dim[j] != _ThreadArgs.centroids[i].dim[j])
    //             flag = 0;
    //     }
    // }
    // printf("s & b = %d\n",flag);


    if (_print == 1){
        print_cluster(_ThreadArgs.points, _ThreadArgs.centroids, _ThreadArgs.assignments, _ThreadArgs.dataSize, _ThreadArgs.k, "seq", _ThreadArgs.dim);

        print_cluster(_ThreadArgs.points, _ThreadArgs.centroids_b, _ThreadArgs.assignments_b, _ThreadArgs.dataSize, _ThreadArgs.k, "bloc", _ThreadArgs.dim);

        print_cluster(_ThreadArgs.points, _ThreadArgs.centroids_m, _ThreadArgs.assignments_m, _ThreadArgs.dataSize, _ThreadArgs.k, "modulo", _ThreadArgs.dim);

        print_cluster(_ThreadArgs.points, _ThreadArgs.centroids_mr, _ThreadArgs.assignments_mr, _ThreadArgs.dataSize, _ThreadArgs.k, "mapR", _ThreadArgs.dim);
    }   
    free_structure();
}
