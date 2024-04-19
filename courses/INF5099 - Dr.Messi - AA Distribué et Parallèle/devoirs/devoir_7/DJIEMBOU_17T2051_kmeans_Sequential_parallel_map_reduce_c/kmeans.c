#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <assert.h>

#include "headers/kmeans.h"
// long plotIndex = 0;
// points data generator
void generate_random_points(int dataSize, int dim, Point* points) {
    for (int i = 0; i < dataSize; i++) {
        points[i].dim = malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            points[i].dim[j] = ((double)rand() / RAND_MAX) * (2 * dataSize) - dataSize;
        }
    }
}

// centroids data generator
void generate_random_centroids(int k, int dataSize, int dim, Point* centroids, Point* points) {
    int* usedIndices = (int*)malloc(dataSize * sizeof(int));
    for (int i = 0; i < dataSize; i++) {
        usedIndices[i] = 0;
    }
    for (int i = 0; i < k; i++) {
        int randomIndex;
        do {
            randomIndex = rand() % dataSize;
        } while (usedIndices[randomIndex]);
        usedIndices[randomIndex] = 1;

        centroids[i].dim = malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            centroids[i].dim[j] = points[randomIndex].dim[j];
        }
    }
    free(usedIndices);
}


// points vector allocation
Point*  generate_points(int size) {
    Point* points = malloc(size * sizeof(Point));
    return points;
}

// free points
void free_structure() {
    free(_ThreadArgs.points->dim);
    free(_ThreadArgs.points);
    free(_ThreadArgs.centroids->dim);
    free(_ThreadArgs.centroids);
    free(_ThreadArgs.assignments_b);
    free(_ThreadArgs.assignments_m);
    free(_ThreadArgs.assignments_mr);
    free(_ThreadArgs.assignments);
    free(_ThreadArgs.centroids_b->dim);
    free(_ThreadArgs.centroids_b);
    free(_ThreadArgs.centroids_m->dim);
    free(_ThreadArgs.centroids_m);
    free(_ThreadArgs.centroids_mr->dim);
    free(_ThreadArgs.centroids_mr);
}



// Fonction pour calculer la distance euclidienne entre deux points
double distance(Point p1, Point p2, int dim) {
    Point Dist = point_soustraction(p1, p2, dim);
    return sqrt(sum_of_power_2_of_minus_items(Dist,dim));
}

Point point_soustraction(Point p1, Point p2, int size) {
    Point result;
    result.dim = malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        result.dim[i] = p1.dim[i]-p2.dim[i];
    }
    return result;
}

double sum_of_power_2_of_minus_items(Point result, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum+= (result.dim[i] * result.dim[i]);
    }
    return sum;
}

// Fonction pour attribuer un point à un cluster
int assignCluster(Point point, Point* centroids, int k, int dim) {
    int cluster = 0;
    double minDistance = distance(point, centroids[0], dim);

    for (int i = 1; i < k; i++) {
        double d = distance(point, centroids[i], dim);
        if (d < minDistance) {
            minDistance = d;
            cluster = i;
        }
    }

    return cluster;
}

// compare two element
int compare(const void* a, const void* b) {
    double arg1 = *(const double*)a;
    double arg2 = *(const double*)b;
    
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

// computes Q1, Q2, Q3, QMax
void calculateQuartiles(double* vector, int size, double* quartiles) {
    qsort(vector, size, sizeof(double), compare);
    
    quartiles[0] = vector[size / 4];
    quartiles[1] = vector[size / 2];
    quartiles[2] = vector[3 * size / 4];
    quartiles[3] = vector[size - 1];
}

// Fonction exécutée par chaque thread par bloc
void* assignPointsToClustersByBloc(void* arg) {
    long threadId = (long)arg;

    // Calcul de la plage de données à traiter par le thread
    long start = (threadId * _ThreadArgs.dataSize) / _ThreadArgs.NUM_THREADS;
    long end = ((threadId + 1) * _ThreadArgs.dataSize) / _ThreadArgs.NUM_THREADS;
    int pointsPerCluster = _ThreadArgs.dataSize / _ThreadArgs.k;
    // Attribution des points aux clusters
    for (int i = start; i < end; i++) {
        // printf("%d->%ld\n",i,threadId);
        // exit(1);
        // int cluster 
        // if (iter == 0){
        //     _ThreadArgs.assignments_b[i] = abs((i/pointsPerCluster)-(_ThreadArgs.k-1));
        // }
        // else{
            _ThreadArgs.assignments_b[i] = assignCluster(_ThreadArgs.points[i], _ThreadArgs.centroids_b, _ThreadArgs.k, _ThreadArgs.dim);
        // printf("assignation par bloc du point %d = %d \n",i,_ThreadArgs.assignments_b[i]);
        // }
    }
    pthread_exit(NULL);
}

void* map(void* arg) {
    long threadId = (long)arg;

    // Calcul de la plage de données à traiter par le thread
    long start = (threadId * _ThreadArgs.dataSize) / _ThreadArgs.NUM_THREADS;
    long end = ((threadId + 1) * _ThreadArgs.dataSize) / _ThreadArgs.NUM_THREADS;
    int indice, prev;
    // local cluster
    Cluster* localCluster = malloc(_ThreadArgs.k * sizeof(Cluster));
    for (int i = 0; i < _ThreadArgs.k; i++){
        localCluster[i].pointSize = 0;
        localCluster[i].points = malloc(_ThreadArgs.dataSize*sizeof(int));
    }

    // Attribution des points aux clusters
    for (int i = start; i < end; i++) {
        // recuperer l'indice du cluster d'assignation
        _ThreadArgs.assignments_mr[i] = assignCluster(_ThreadArgs.points[i], _ThreadArgs.centroids_mr, _ThreadArgs.k, _ThreadArgs.dim);
        
        // printf("iter %d assignation par map reduce du point %d = %d \n", iter,i,_ThreadArgs.assignments_mr[i]);
        // sauvegarder le dernier indice occupé
        prev = localCluster[_ThreadArgs.assignments_mr[i]].pointSize;
        // incrementer le nombre de points dans le cluster
        localCluster[_ThreadArgs.assignments_mr[i]].pointSize += 1;
        // reallouer le vecteur de 
        // if(localCluster[_ThreadArgs.assignments_mr[i]].pointSize > initialSize){
        //     localCluster[_ThreadArgs.assignments_mr[i]].points = realloc(localCluster[_ThreadArgs.assignments_mr[i]].points,localCluster[_ThreadArgs.assignments_mr[i]].pointSize*sizeof(Point));
        // }
        // ajouter le nouveau point
        localCluster[_ThreadArgs.assignments_mr[i]].points[prev] = i;
    }

    
    // melange de données des clusters
    for(int i=0; i<_ThreadArgs.k; i++)
    {

        // indice du dernier element dans le cluster global
        // printf("g%d:%d, l%d:%d",i,_clusters[i].pointSize,i,localCluster[i].pointSize);
        pthread_mutex_lock (&mutex_variable);
            prev = _clusters[i].pointSize;
            // incrementer le nombre de points dans le cluster global
            _clusters[i].pointSize += localCluster[i].pointSize;
        pthread_mutex_unlock (&mutex_variable);

        // // reallouer le vecteur de points
        // if(_clusters[i].pointSize > initialSize){
            
        //     _clusters[i].points = realloc(_clusters[i].points,_clusters[i].pointSize*sizeof(Point));
            
        // }
        // ajouter les nouveaux points
        
        for(int j=0; j<localCluster[i].pointSize; j++){
            _clusters[i].points[prev+j] = localCluster[i].points[j];
        }
    }   
    

    pthread_exit((void*) 0);
}

// void* map(void* arg) {
//     long threadId = (long)arg;

//     // Calcul de la plage de données à traiter par le thread
//     // long start = (threadId * _ThreadArgs.dataSize) / _ThreadArgs.NUM_THREADS;
//     // long end = ((threadId + 1) * _ThreadArgs.dataSize) / _ThreadArgs.NUM_THREADS;
//     int  prev;
//     // local cluster
//     Cluster* localCluster = malloc(_ThreadArgs.k * sizeof(Cluster));
//     for (int i = 0; i < _ThreadArgs.k; i++){
//         localCluster[i].pointSize = 0;
//         localCluster[i].points = malloc(_ThreadArgs.dataSize*sizeof(int));
//     }

//     int pointsPerCluster = _ThreadArgs.dataSize / _ThreadArgs.k;
//     // Attribution des points aux clusters
//     for (int i = threadId; i < _ThreadArgs.dataSize; i+=_ThreadArgs.NUM_THREADS) {
//         // recuperer l'indice du cluster d'assignation
//         // if (iter == 0){
//         //     _ThreadArgs.assignments_mr[i] = abs((i/pointsPerCluster)-(_ThreadArgs.k-1));
//         // }
//         // else{
//             _ThreadArgs.assignments_mr[i] = assignCluster(_ThreadArgs.points[i], _ThreadArgs.centroids_mr, _ThreadArgs.k, _ThreadArgs.dim);
//         // }
//         printf("iter %d assignation par map reduce du point %d = %d \n", iter,i,_ThreadArgs.assignments_mr[i]);
//         prev = localCluster[_ThreadArgs.assignments_mr[i]].pointSize;
//         localCluster[_ThreadArgs.assignments_mr[i]].pointSize ++;
//         // ajouter le nouveau point
//         localCluster[_ThreadArgs.assignments_mr[i]].points[prev] = i;
//     } 
//     // melange de données des clusters
//     for(int i=0; i<_ThreadArgs.k; i++)
//     {

//         // indice du dernier element dans le cluster global
//         // printf("g%d:%d, l%d:%d",i,_clusters[i].pointSize,i,localCluster[i].pointSize);
//         pthread_mutex_lock (&mutex_variable);
//         prev = _clusters[i].pointSize;
//         // incrementer le nombre de points dans le cluster global
//         _clusters[i].pointSize += localCluster[i].pointSize;
//         pthread_mutex_unlock (&mutex_variable);
//         // ajouter les nouveaux points
//         for(int j=0; j<localCluster[i].pointSize; j++){
//             _clusters[i].points[prev+j] = localCluster[i].points[j];
//         }
//         free(localCluster[i].points);
//     } 
//     free(localCluster);  

//     pthread_exit((void*) 0);
// }

// Fonction exécutée par chaque thread par bloc
void* assignPointsToClustersByMod(void* arg) {
    long threadId = (long)arg;

    // Calcul de la plage de données à traiter par le thread
    // long start = (threadId * _ThreadArgs.dataSize) / _ThreadArgs.NUM_THREADS;
    long start = threadId;
    int pointsPerCluster = _ThreadArgs.dataSize / _ThreadArgs.k;
    // Attribution des points aux clusters
    for (int i = start; i < _ThreadArgs.dataSize; i+= _ThreadArgs.NUM_THREADS) {
        // int cluster 
        // if (iter == 0){
        //     _ThreadArgs.assignments_m[i] = abs((i/pointsPerCluster)-(_ThreadArgs.k-1));
        // }
        // else{
        _ThreadArgs.assignments_m[i] = assignCluster(_ThreadArgs.points[i],_ThreadArgs.centroids_m, _ThreadArgs.k, _ThreadArgs.dim);
        
        // }
        // printf("assignation par module du point %d = %d \n",i,_ThreadArgs.assignments_m[i]);
    }
    pthread_exit(NULL);
}

// Fonction pour recalculer les centres de cluster
void* reduce(void* arg){
    // long begin, end;
    long thread_id;
    thread_id = (long)arg;

    // begin = (thread_id * _ThreadArgs.k) / _ThreadArgs.NUM_THREADS;

    // if(thread_id == (_ThreadArgs.NUM_THREADS - 1))
    //    end = _ThreadArgs.k;
    // else
    //    end = ((thread_id + 1) * _ThreadArgs.k) / _ThreadArgs.NUM_THREADS;

    
    

    // calcul des nouveaux centroids
    for (int i = thread_id; i < _ThreadArgs.k; i+=_ThreadArgs.NUM_THREADS) {
        if (_clusters[i].pointSize > 0) {
            // Point* sum = malloc(sizeof(Point));
            // sum->dim = malloc(_ThreadArgs.dim * sizeof(double));  // Allocate memory for sum[i].dim
    
            // // reinitialiser le centroid i
            // // if (_ThreadArgs.centroids_mr[i].dim != NULL)
            // //     free(_ThreadArgs.centroids_mr[i].dim);

            // // _ThreadArgs.centroids_mr[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
            // // initialiser le point de aggregation
            // // pthread_mutex_lock (&mutex_variable);
            // memset(sum->dim, 0.0, _ThreadArgs.dim * sizeof(double)); // Initialize sum.dim
            // free(_ThreadArgs.centroids_mr[i].dim);
            // _ThreadArgs.centroids_mr[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
            memset(_ThreadArgs.centroids_mr[i].dim, 0.0, _ThreadArgs.dim * sizeof(double));
            // pthread_mutex_unlock (&mutex_variable);
            // somme des points du cluster
            for (int j = 0; j < _clusters[i].pointSize; j++) {
                for (int k = 0; k < _ThreadArgs.dim; k++) {
                    _ThreadArgs.centroids_mr[i].dim[k] += _ThreadArgs.points[_clusters[i].points[j]].dim[k];
                    
                }
                // printf(" cluster  %d -> point %d \n",i,_clusters[i].points[j]);
            }
            for (int k = 0; k < _ThreadArgs.dim; k++) {
                _ThreadArgs.centroids_mr[i].dim[k] = _ThreadArgs.centroids_mr[i].dim[k] / _clusters[i].pointSize;
                // printf("mr new du centroids i %d = %f \n",i,_ThreadArgs.centroids_mr[i].dim[k]);
            }
            // for (int k = 0; k < _ThreadArgs.dim; k++) {
            //     _ThreadArgs.centroids_mr[i].dim[k] += _ThreadArgs.centroids_mr[i].dim[k] / _clusters[i].pointSize;
            // }
            // calculer le nouveau centroide
            // for (int j = 0; j < _ThreadArgs.dim; j++) {
            //     _ThreadArgs.centroids_mr[i].dim[j] = sum->dim[j] / _clusters[i].pointSize;
            // }
            // free(sum->dim);
            // free(sum);
        }
        
    }
    // exit(0);
    //
    pthread_exit(NULL);
}

void updateCentroids(int* assignments, Point* centroids, Point* points, int dataSize, int k, int dim) {
    int* clusterCounts = malloc(k * sizeof(int));
    memset(clusterCounts, 0, k * sizeof(int));

    Point* sum = malloc(k * sizeof(Point));

    // init sum vector
    for (int i = 0; i < k; i++) {
        
        sum[i].dim = malloc(dim * sizeof(double));
        memset(sum[i].dim, 0.0, dim * sizeof(double)); // Initialize sum[i].dim

    }
    // calculate count of points in cluter and sum of cluster
    for (int i = 0; i < dataSize; i++) {
        int cluster = assignments[i];
        clusterCounts[cluster]++;
        for (int j = 0; j < dim; j++) {
            sum[cluster].dim[j] += points[i].dim[j];
        }
    }
    // calculate new centroid
    for (int i = 0; i < k; i++) {
        if (clusterCounts[i] > 0) {
            // reset centroids
            free(centroids[i].dim);
            centroids[i].dim = malloc(dim * sizeof(double));
            // compute new point
            for (int j = 0; j < dim; j++) {
                centroids[i].dim[j] = sum[i].dim[j] / clusterCounts[i];
                // printf("new du centroids i %d = %f \n",i,centroids[i].dim[j]);
            }
        }
    }
    // // get the point which is near of the new centroids
    // for (int i = 0; i < k; i++) {
    //     int index = assignCluster(sum[i],points, dataSize);
    //     centroids[i].x = points[index].x;
    //     centroids[i].y = points[index].y;
    // }
    free(sum->dim);
    free(sum);
    free(clusterCounts);
}

// Fonction pour vérifier la convergence des centroïdes
int checkConvergence(Point* oldCentroids, Point* centroids, int numCentroids, double tolerance, int dim) {
    for (int i = 0; i < numCentroids; i++) {
        double d = distance(oldCentroids[i], centroids[i], dim);
        if (d > tolerance) {
            return 0; // Non convergé
        }
    }
    return 1; // Convergé
}

// clusters printing
void print_cluster(Point* points, Point* centroids, int* assignments, int dataSize, int k, const char* domaine, int dim) {
    const char* path = "outputs/clusters";
    const char* path1 = "outputs/centroids";
    const char* currentDate = getCurrentDate();
    char f[100];
    snprintf(f, sizeof(f), "%s_%s_%d_%d_%d_%s.csv", path, domaine,
        _ThreadArgs.NUM_THREADS,
        _ThreadArgs.dataSize,
        _ThreadArgs.dim,currentDate);
    const char* line = "X,Y,cluster";
    const char* line1 = "X,Y,color";
    if (fileExists(f)) {
        printf("%s exists\n",f);
    } else {
        write_csv(line, f);
    }
    char f1[100];
    snprintf(f1, sizeof(f1), "%s_%s_%d_%d_%d_%s.csv", path1, domaine,
        _ThreadArgs.NUM_THREADS,
        _ThreadArgs.dataSize,
        _ThreadArgs.dim,currentDate);
    if (fileExists(f1)) {
        printf("%s exists\n",f1);
    } else {
        write_csv(line1, f1);
    }
    double* quartiles = malloc(4 * sizeof(double));
    for (int i = 0; i < dataSize; i++) {
        char line[100];
        calculateQuartiles(points[i].dim, dim, quartiles);
        snprintf(line, sizeof(line), "%lf,%lf,%d", quartiles[0], quartiles[3], assignments[i]);
        // printf("Point (%.1f, %.1f) assigned to cluster %d\n", quartiles[0], quartiles[3], assignments[i]);
        
        
        write_csv(line, f);
    }
    for (int i = 0; i < k; i++) {
        char line[100];
        calculateQuartiles(centroids[i].dim, dim, quartiles);
        snprintf(line, sizeof(line), "%lf,%lf,%d", quartiles[0], quartiles[3],i);
        // printf("Point (%.1f, %.1f) assigned to cluster %d\n", quartiles[0], quartiles[3], assignments[i]);
        write_csv(line, f1);
    }

    gnuplot(currentDate, domaine);


    free(quartiles);
}

void gnuplot(const char* currentDate, const char* domaine){
    // generate the gnuplot context
    char line[10000];
    snprintf(
        line, 
        sizeof(line), 
        "set datafile separator \",\"\n"
        "# set terminal pngcairo size 800,600\n"
        "set terminal png\n"
        "set output \"outputs/cluster_%s_%d_%d_%d_%s.png\"\n\n"

        "set xlabel \"X\"\n"
        "set ylabel \"Y\"\n"
        "set title \"Bubble Chart\"\n"
        "set key off\n"

        "# Define color palette\n"
        "set palette defined (0 \"red\", 1 \"blue\", 2 \"green\", 3 \"orange\", 4 \"purple\", 5 \"cyan\", 6 \"magenta\", 7 \"yellow\", 8 \"gray\", 9 \"brown\")\n\n"

        "# Define bubble size\n"
        "centroid_size = 2.0\n"

        "# Plot the data points\n"
        "plot \"outputs/clusters_%s_%d_%d_%d_%s.csv\" using 1:2:3 with points linecolor variable pointtype 7,  \\\n\t"
            "\"outputs/centroids_%s_%d_%d_%d_%s.csv\" using 1:2:3 with points linecolor variable pointtype 7 pointsize centroid_size", 
        domaine , 
        _ThreadArgs.NUM_THREADS,
        _ThreadArgs.dataSize,
        _ThreadArgs.dim,
        currentDate, 
        domaine,
        _ThreadArgs.NUM_THREADS,
        _ThreadArgs.dataSize,
        _ThreadArgs.dim, 
        currentDate, 
        domaine,
        _ThreadArgs.NUM_THREADS,
        _ThreadArgs.dataSize,
        _ThreadArgs.dim,
        currentDate
        );

    // define the name
    char file[100];
    snprintf(file, sizeof(file), "cluster_%s.gp", currentDate);
    write_csv(line, file);

    // define the relate commande
    char command[100];
    snprintf(command, sizeof(command), "gnuplot cluster_%s.gp", currentDate);

    // execute the context
    printf("%d is the status of gnuplot call\n", system(command));

    // plotIndex++;

}

// sequential kmeans
void kmeansSequential() {
    // Initialisation des centres de cluster
    // _ThreadArgs.centroids = generate_points(_ThreadArgs.k);
    // generate_random_centroids(_ThreadArgs.k, _ThreadArgs.dataSize, _ThreadArgs.dim, _ThreadArgs.centroids, _ThreadArgs.points);
    // Attribution initiale des points aux clusters
    _ThreadArgs.assignments = malloc(_ThreadArgs.dataSize * sizeof(int));

    // Algorithme K-means séquentiel
    iter = 0;
    Point* oldCentroids = malloc(_ThreadArgs.k * sizeof(Point));
    for (int i = 0; i < _ThreadArgs.k; i++){
        oldCentroids[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
    }
    int pointsPerCluster = _ThreadArgs.dataSize / _ThreadArgs.k;
    _ThreadArgs.convergence = 0;
    // while (iter < _ThreadArgs.max_iter) {
    while (iter < _ThreadArgs.max_iter && !_ThreadArgs.convergence) {
    // while (!_ThreadArgs.convergence) {

        for (int i = 0; i < _ThreadArgs.k; i++){
            // free(oldCentroids[i].dim);
            // oldCentroids[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
            for (int j = 0; j < _ThreadArgs.dim; j++) {
                oldCentroids[i].dim[j] = _ThreadArgs.centroids[i].dim[j] ;
            }
            
        }
        // Attribution des points aux clusters
        _ThreadArgs.convergence = 1;
        for (int i = 0; i < _ThreadArgs.dataSize; i++) {
            // int cluster
            // if (iter == 0){
            //     _ThreadArgs.assignments[i] = abs((i/pointsPerCluster)-(_ThreadArgs.k-1));
            // }
            // else{
            _ThreadArgs.assignments[i] = assignCluster(_ThreadArgs.points[i], _ThreadArgs.centroids, _ThreadArgs.k, _ThreadArgs.dim);
            // };
        }
        // printf("%d<---",iter);

        // Recalcul des centres de cluster
        updateCentroids(_ThreadArgs.assignments, _ThreadArgs.centroids, _ThreadArgs.points, _ThreadArgs.dataSize, _ThreadArgs.k, _ThreadArgs.dim);
        _ThreadArgs.convergence = checkConvergence(oldCentroids, _ThreadArgs.centroids, _ThreadArgs.k, _ThreadArgs.tolerance, _ThreadArgs.dim);
        iter++;
    }
    printf("iter:%d",iter);
    for (int i = 0; i < _ThreadArgs.k; i++) {
        if (oldCentroids[i].dim != NULL)
            free(oldCentroids[i].dim);
    }
    if (oldCentroids != NULL)
        free(oldCentroids);
}

// parallel kmeans by bloc
void kmeansParallel_B() {
    // Initialisation des centres de cluster
    // printf("lll..");
    
    // _ThreadArgs.centroids_b = generate_points(_ThreadArgs.k);
    // generate_random_centroids(_ThreadArgs.k, _ThreadArgs.dataSize, _ThreadArgs.dim, _ThreadArgs.centroids_b, _ThreadArgs.points);
    // Attribution initiale des points aux clusters
    // printf("lll..");
    // _ThreadArgs.assignments_b = malloc(_ThreadArgs.dataSize * sizeof(int));

    // Algorithme K-means séquentiel
    iter = 0;
    Point* oldCentroids = malloc(_ThreadArgs.k * sizeof(Point));

    // printf("lll..");
    for (int i = 0; i < _ThreadArgs.k; i++){
        oldCentroids[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
    }

    _ThreadArgs.convergence = 0;
    // printf("lll..");

    pthread_t threads[_ThreadArgs.NUM_THREADS];

    // while (iter < _ThreadArgs.max_iter) {
    while (iter < _ThreadArgs.max_iter  && !_ThreadArgs.convergence) {
    // while (!_ThreadArgs.convergence) {

        for (int i = 0; i < _ThreadArgs.k; i++){
            // printf("lll..");
            // free(oldCentroids[i].dim);
            // oldCentroids[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
            for (int j = 0; j < _ThreadArgs.dim; j++) {
                oldCentroids[i].dim[j] = _ThreadArgs.centroids_b[i].dim[j] ;
            }
            
        }
        _ThreadArgs.convergence = 1;
        // Création des threads
        // printf("lll..");
        // Attribution des points aux clusters
        // printf("lll..");
        // long i;
        for (long i = 0; i < _ThreadArgs.NUM_THREADS; i++) {
            // printf("%ldll..",i);

            pthread_create(&threads[i], NULL, assignPointsToClustersByBloc, (void*)i);
        }
        for (long i = 0; i < _ThreadArgs.NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
        // Recalcul des centres de cluster
        updateCentroids(_ThreadArgs.assignments_b, _ThreadArgs.centroids_b, _ThreadArgs.points, _ThreadArgs.dataSize, _ThreadArgs.k, _ThreadArgs.dim);
        
        _ThreadArgs.convergence = checkConvergence(oldCentroids, _ThreadArgs.centroids_b, _ThreadArgs.k, _ThreadArgs.tolerance, _ThreadArgs.dim);
        // for (int i = 0; i<_ThreadArgs.k; i++) {
        //     for (int j = 0; j < _ThreadArgs.dim; j++) {
        //         printf("b[%d][%d]:%f iter:%d\n",i,j,_ThreadArgs.centroids_b[i].dim[j],iter);
        //         printf("----\n");
        //     }
        // }
        iter++;
    }
    printf("iter:%d",iter);
    for (int i = 0; i < _ThreadArgs.k; i++) {
        if (oldCentroids[i].dim != NULL)
            free(oldCentroids[i].dim);
    }
    if (oldCentroids != NULL)
        free(oldCentroids);
}

// parallel kmeans by modulo
void kmeansParallel_M() {
    // Initialisation des centres de cluster
    // _ThreadArgs.centroids_m = generate_points(_ThreadArgs.k);
    // generate_random_centroids(_ThreadArgs.k, _ThreadArgs.dataSize, _ThreadArgs.dim, _ThreadArgs.centroids_m, _ThreadArgs.points);
    // Attribution initiale des points aux clusters
    // _ThreadArgs.assignments_m = malloc(_ThreadArgs.dataSize * sizeof(int));

    // Algorithme K-means séquentiel
    iter = 0;
    Point* oldCentroids = malloc(_ThreadArgs.k * sizeof(Point));
    // printf("lll..");
    for (int i = 0; i < _ThreadArgs.k; i++){
        oldCentroids[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
    }
    _ThreadArgs.convergence = 0;

    // Création des threads
    pthread_t threads[_ThreadArgs.NUM_THREADS];
    // Algorithme K-means séquentiel
    // while (iter < _ThreadArgs.max_iter) {
    while (iter < _ThreadArgs.max_iter && !_ThreadArgs.convergence) {
    // while (!_ThreadArgs.convergence) {

        for (int i = 0; i < _ThreadArgs.k; i++){
            // printf("lll..");
            // free(oldCentroids[i].dim);
            // oldCentroids[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
            for (int j = 0; j < _ThreadArgs.dim; j++) {
                oldCentroids[i].dim[j] = _ThreadArgs.centroids_m[i].dim[j] ;
            }
            
        }

        _ThreadArgs.convergence = 1;
        // Attribution des points aux clusters
        for (long i = 0; i < _ThreadArgs.NUM_THREADS; i++) {
            pthread_create(&threads[i], NULL, assignPointsToClustersByMod, (void*)i);
        }
        for (long i = 0; i < _ThreadArgs.NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }

        // exit(0);
        // Recalcul des centres de cluster
        updateCentroids(_ThreadArgs.assignments_m, _ThreadArgs.centroids_m, _ThreadArgs.points, _ThreadArgs.dataSize, _ThreadArgs.k, _ThreadArgs.dim);
        // exit(0);
        // exit(0);
        _ThreadArgs.convergence = checkConvergence(oldCentroids, _ThreadArgs.centroids_m, _ThreadArgs.k, _ThreadArgs.tolerance, _ThreadArgs.dim);
        
        // for (int i = 0; i<_ThreadArgs.k; i++) {
        //     for (int j = 0; j < _ThreadArgs.dim; j++) {
        //         printf("m[%d][%d]:%f iter:%d\n",i,j,_ThreadArgs.centroids_m[i].dim[j],iter);
        //         printf("----\n");
        //     }
        // }
        iter++;
    }
    printf("iter:%d",iter);
    for (int i = 0; i < _ThreadArgs.k; i++) {
        if (oldCentroids[i].dim != NULL)
            free(oldCentroids[i].dim);
    }
    if (oldCentroids != NULL)
        free(oldCentroids);
}

// parallel kmeans by MapReduce
void kmeansParallel_MR() {
    // Initialisation des centres de cluster
    // printf("lll..");
    
    // _ThreadArgs.centroids_b = generate_points(_ThreadArgs.k);
    // generate_random_centroids(_ThreadArgs.k, _ThreadArgs.dataSize, _ThreadArgs.dim, _ThreadArgs.centroids_b, _ThreadArgs.points);
    // Attribution initiale des points aux clusters
    // printf("lll..");
    // _ThreadArgs.assignments_mr = malloc(_ThreadArgs.dataSize * sizeof(int));

    // Algorithme K-means séquentiel
    iter = 0;
    Point* oldCentroids = malloc(_ThreadArgs.k * sizeof(Point));
    // printf("lll..");
    for (int i = 0; i < _ThreadArgs.k; i++){
        oldCentroids[i].dim = malloc(_ThreadArgs.dim * sizeof(double));
    }
    _ThreadArgs.convergence = 0;
    // printf("lll..");
    int rc;

    pthread_t threads[_ThreadArgs.NUM_THREADS];
    pthread_mutex_init(&mutex_variable, NULL);
    // while (iter < _ThreadArgs.max_iter) {
    while (iter < _ThreadArgs.max_iter  && !_ThreadArgs.convergence) {
    // while (!_ThreadArgs.convergence) {

        for (int i = 0; i < _ThreadArgs.k; i++){
            for (int j = 0; j < _ThreadArgs.dim; j++) {
                oldCentroids[i].dim[j] = _ThreadArgs.centroids_mr[i].dim[j] ;
            }
            _clusters[i].pointSize = 0;
        }
        _ThreadArgs.convergence = 1;
        // Création des threads
        
        // Attribution des points aux clusters
        // long i;
        // Pour chaque moins associer un cluster de plus proche distance
        // printf("0");
        for (long i = 0; i < _ThreadArgs.NUM_THREADS; i++) {
            pthread_create(&threads[i], NULL, map, (void*)i);
        }
        for (long i = 0; i < _ThreadArgs.NUM_THREADS; i++) {
            rc = pthread_join(threads[i], NULL);
            // printf("map iter %d est %d\n",iter,rc);
        }

        // pour chaque cluster, mixer les resultats et trier
        // aucun traitement necessaire
        // Recalcul des centres de cluster
        // printf("1");
        for (long i = 0; i < _ThreadArgs.NUM_THREADS; i++) {
            pthread_create(&threads[i], NULL, reduce, (void*)i);
        }
        // printf("----");
        for (long i = 0; i < _ThreadArgs.NUM_THREADS; i++) {
            rc = pthread_join(threads[i], NULL);
            // printf("reduce iter %d est %d\n",iter,rc);
        }
        // exit(0);
        _ThreadArgs.convergence = checkConvergence(oldCentroids, _ThreadArgs.centroids_mr, _ThreadArgs.k, _ThreadArgs.tolerance, _ThreadArgs.dim);
        // for (int i = 0; i<_ThreadArgs.k; i++) {
        //     for (int j = 0; j < _ThreadArgs.dim; j++) {
        //         printf("mr[%d][%d]:%f iter:%d\n",i,j,_ThreadArgs.centroids_mr[i].dim[j],iter);
        //         printf("----\n");
        //     }
        // }
        iter++;
        // printf("--");
    }
    pthread_mutex_destroy(&mutex_variable);  // Destroy the mutex
    printf("iter:%d",iter);
    for (int i = 0; i < _ThreadArgs.k; i++) {
        if (oldCentroids[i].dim != NULL)
            free(oldCentroids[i].dim);
    }
    if (oldCentroids != NULL)
        free(oldCentroids);
}