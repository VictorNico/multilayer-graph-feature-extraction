#include "headers/kmeans.h"
#include "headers/gui.h"
// #include "headers/timer.h"


int main(int argc, char *argv[])
{
	if(argc != 8){
        printf("Invalid number of hyperparameters\nThe call must have the following args order\nDataDimension DataExample ClustersNumber Threshold ThreadsNumber genImgFlag\nPlease fill on correct informations and try it again");
        exit(-1);
    }
    
    
    int arg1 = atoi(argv[1]);
    int arg2 = atoi(argv[2]);
    int arg3 = atoi(argv[3]);  
    int arg4 = atoi(argv[4]); 
    double arg5 = atoi(argv[5]); 
    int arg6 = atoi(argv[6]);  
    int arg7 = atoi(argv[7]);  

    if (arg1 <= 1){
    	printf(" invalid inputs! Set the points dimension greater than 2 \n");
        exit(-1);
    }
	if (arg2 < 1){
		printf(" invalid inputs! Set length of data points greater than 1\n");
		exit(-1);
	}
	if (arg3 < 2){
		printf("invalid inputs! Set number of targeted clusters greater than 1\n");
		exit(-1);
	}
	if (arg4 < 1){
		printf(" invalid! Set Maximum iteration threshold greater than 0\n");
		exit(-1);
	}  
	if (arg5 < 0.0){
		printf(" invalid inputs! Set convergence tolerance greater than 0.0\n");
		exit(-1);
	}
	if (arg6 < 2){
		printf(" invalid inputs! set number of threads greater than 1\n");
		exit(-1);
	}  
	if (arg7 < 0 ||  arg7 > 1){
		printf(" invalid inputs! set show plots between 1 and 0\n");
		exit(-1);
	}             

	_print = arg7;
    _ThreadArgs.dim = arg1;
                
    _ThreadArgs.dataSize = arg2;
    _ThreadArgs.k = arg3;
    _ThreadArgs.max_iter = arg4;
    _ThreadArgs.tolerance = arg5;
    _ThreadArgs.NUM_THREADS = arg6;

	gui();
	return 0;
}
