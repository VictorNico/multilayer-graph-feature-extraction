#include<pthread.h>
#include<stdlib.h>
#include<stdio.h>
#include <math.h>
#define NUM_THREADS 5


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
	int rc;
	long t;
	void *status;

	/* Initialize and set thread detached attribute */
	//pthread_attr_init(&attr);
	//pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(t=0; t<NUM_THREADS; t++) {
		printf("Main: creating thread %ld\n", t);
		//rc = pthread_create(&thread[t], NULL, PrintHello, (void *)t);
		rc = pthread_create(&thread[t], NULL, BusyWork, (void *)t);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	/* Free attribute and wait for the other threads */
	//pthread_attr_destroy(&attr);
	/*for(t=0; t<NUM_THREADS; t++) {
		rc = pthread_join(thread[t], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
		printf("Main: completed join with thread %ld having a statusof %ld\n",t,(long)status);
	}*/
}
