#ifndef KMEANS_H
#define KMEANS_H


#include "file.h"
#include "time.h"

/** \struct POINT
 * \brief Storage structure for point
 * 
 * This structure store all usefull variable needs for a point
 */
typedef struct {
    double* dim;
} Point;

/** \struct ThreadArgs
 * \brief Storage structure
 * 
 * This structure store all usefull variable needs for theads matrix multiply operation
 */
typedef struct {
    Point* points; /** vector of data point */
    int dim; /** point embedding size */
    int* assignments_b; /** vector of assignments by bloc */
    int* assignments_m; /** vector of assignments by mod */
    int* assignments; /** vector of assignments */
    Point* centroids_b; /** vector of centroids by bloc */
    Point* centroids_m; /** vector of centroids by mod */
    Point* centroids; /** vector of centroids */
    int dataSize; /** size of data */
    int k; /** number of cluster */
    double tolerance; /** convergence threshold */
    int max_iter; /** max number of iteration */
    int NUM_THREADS; /** Number of thread to use */
    int convergence; /** 1 if no more centroids change and 0 else */    
} ThreadArgs;

// global instance
ThreadArgs _ThreadArgs;

void kmeansParallel_B();
void kmeansParallel_M();

void kmeansSequential();

void print_cluster(Point* points, Point* centroids, int* assignments, int dataSize, int k, const char* domaine, int dim);

int checkConvergence(Point* oldCentroids, Point* centroids, int numCentroids, double epsilon, int dim);

void updateCentroids(int* assignments, Point* centroids, Point* points, int dataSize, int k, int dim);

void* assignPointsToClustersByMod(void* arg);

void* assignPointsToClustersByBloc(void* arg);

int assignCluster(Point point, Point* centroids, int k, int dim);

double distance(Point p1, Point p2, int dim);

Point point_soustraction(Point p1, Point p2, int size);

double sum_of_power_2_of_minus_items(Point result, int size);

void free_structure();

Point*  generate_points(int size);

void generate_random_centroids(int k, int dataSize, int dim, Point* centroids, Point* points);

void generate_random_points(int dataSize, int dim, Point* points);

int compare(const void* a, const void* b);

void calculateQuartiles(double* vector, int size, double* quartiles);

void gnuplot(const char* currentDate, const char* domaine);

#endif