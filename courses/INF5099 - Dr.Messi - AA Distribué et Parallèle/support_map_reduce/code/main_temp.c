#include "hello_world_map_reduce.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>



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




data_t data;
pthread_mutex_t mutex_sum;


float* creer_vecteur(int n)
{
	int i;
	float* tab = (float*)malloc(sizeof(float)*n);
	
	for(i=0;i<n;i++)
	{
		tab[i] = 1.0+ 0.5 * (rand()%100);
	}

	return tab;	
}

float produit_scalaire(float* x, float* y, int n)
{
	int i;
	float somme = 0.0;
	for(i = 0; i < n; i++)
		somme = somme + x[i]*y[i];

	return somme;
}

void *BusyWork(void *t)
{
	int i;
	long tid;
	double result=0.0;
	tid = (long)t;
	printf("Thread %ld starting...\n",tid);
	for (i=0; i<1000000; i++)
	{
		//result = result + sin(i) * tan(i);
		result = result + i*i + (4*i);
	}
	printf("Thread %ld done. Result = %e\n",tid, result);
	pthread_exit((void*) t);
}

void *PrintHello(void *threadid)
{
	long tid;
	tid = (long)threadid;
	printf("Hello World! It's me, thread #%ld!\n", tid);
	pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
	pthread_t thread[NUM_THREADS];
	pthread_attr_t attr;
	int rc, n;
	float somme, somme_parallele;
	long t;
	void *status;

	srand(time(NULL));

	float* x, *y;
	
	//printf("Entrez la taille des vecteurs\n");
	//scanf("%d",&n);

	if(argc < 2)
	{
		printf("Nombre d'argument insuffisant");
		exit(-1);
	}
	else
		n =  atoi(argv[1]);

	x= creer_vecteur(n);
	y= creer_vecteur(n);

	data.x = x;
	data.y = y;
	data.sum = 0.0;
	data.length= n;

	pthread_mutex_init(&mutex_sum, NULL);
	

	unsigned long temps = cpu_time();
	printf("\ntime seq = %ld.%03ldms\n", temps/1000, temps%1000);
	
	/* Initialize and set thread detached attribute */
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

top1();
	for(t=0; t<NUM_THREADS; t++) {
		//printf("Main: creating thread %ld\n", t);
		rc = pthread_create(&thread[t], &attr, map, (void *)t);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	/* Free attribute and wait for the other threads */
	//pthread_attr_destroy(&attr);
	for(t=0; t<NUM_THREADS; t++) {
		rc = pthread_join(thread[t], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
		//printf("Main: completed join with thread %ld having a statusof %ld\n",t,(long)status);
	}

top2();
unsigned long temps_par = cpu_time();
	printf("\ntime par = %ld.%03ldms\n", temps_par/1000, temps_par%1000);


// code qui trie.

top1();
	for(t=0; t<NUM_THREADS; t++) {
		//printf("Main: creating thread %ld\n", t);
		rc = pthread_create(&thread[t], &attr, reduce, (void *)t);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	/* Free attribute and wait for the other threads */
	//pthread_attr_destroy(&attr);
	for(t=0; t<NUM_THREADS; t++) {
		rc = pthread_join(thread[t], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
		//printf("Main: completed join with thread %ld having a statusof %ld\n",t,(long)status);
	}

top2();
unsigned long temps_par = cpu_time();
	printf("\ntime par = %ld.%03ldms\n", temps_par/1000, temps_par%1000);
	

}
