#include "hello_world_map_reduce.h"

#include <string.h>
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

int main(int argc, char *argv[])
{
	pthread_t thread[NB_THREADS];
	pthread_attr_t attr;
	int rc;
	long t;
	void *status;
	char*file_name = malloc(sizeof(char)*50);

	srand(time(NULL));
	if(argc < 2)
	{
		printf("Nombre d'argument insuffisant");
		exit(-1);
	}
	else
	{
		//n =  atoi(argv[1]);
		strcpy(file_name, argv[1]);
	}

	nb_lines = count_lines(file_name);
	tab_lines = read_file(file_name,nb_lines);

	nb_words=0;
	tab_words = (char**)malloc(sizeof(char*));

	nb_d_words=0;
	tab_d_words = (char**)malloc(sizeof(char*));
	



	//print_tab_string(tab_lines, nb_lines);


	top1();

	pthread_mutex_init(&mutex_variable, NULL);
	

	
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(t=0; t<NB_THREADS; t++){
		//printf("Main: creating thread %ld\n", t);
		rc = pthread_create(&thread[t], &attr, map, (void *)t);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	//pthread_attr_destroy(&attr);
	for(t=0; t<NB_THREADS; t++){
		rc = pthread_join(thread[t], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
		//printf("Main: completed join with thread %ld having a statusof %ld\n",t,(long)status);
	}

// code qui trie.

	//print_tab_string(tab_words, nb_words);

	quicksort_string(tab_words, 0, nb_words-1);
	//string_insertion_sort(tab_words,nb_words);


	//char** tmp = (char**)malloc(sizeof(char*)*nb_words);
	//triFusion(0, nb_words-1,tab_words, tmp);

	//print_tab_string(tab_words, nb_words);

	extract_distinct_words();

	//print_tab_string(tab_d_words, nb_d_words);

	tab_occ_words = (int*)malloc(sizeof(int)*nb_d_words);

	for(t=0; t<NB_THREADS; t++) {
		//printf("Main: creating thread %ld\n", t);
		rc = pthread_create(&thread[t], &attr, reduce, (void *)t);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	//pthread_attr_destroy(&attr);
	for(t=0; t<NB_THREADS; t++){
		rc = pthread_join(thread[t], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
		//printf("Main: completed join with thread %ld having a statusof %ld\n",t,(long)status);
	}
	
	printf("Le nombre de mots du fichier est: %d\n", nb_words);
	print_word_occurrence(tab_d_words, nb_d_words, tab_occ_words);

	top2();

	unsigned long temps = cpu_time();
	printf("\ntime seq = %ld.%03ldms\n", temps/1000, temps%1000);


	const char* path = "results.csv";
	const char* line_1 = "tri,mr_time";


	if (fileExists(path)) {
		printf("%s exists\n",path);
	} else {
		write_csv(line_1, path);
	}
	
	char line[100];
	snprintf(line, sizeof(line), "1,%ld.%03ld", temps/1000, temps%1000);

	write_csv(line, path);
}

