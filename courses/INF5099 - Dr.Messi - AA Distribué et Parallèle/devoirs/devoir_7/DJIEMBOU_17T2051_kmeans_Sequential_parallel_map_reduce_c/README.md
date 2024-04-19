# KMEANS - PARALLELISATION

# Tutorial

* In the project structure, you have 
	* [x] headers : contains all prototypes and global variables declarations
	* [x] outputs : contains results of executions
	* [x] file.c, gui.c, kmeans.c : code declaration max
	* [x] Makefile: a makefile to make easy the compilation and treatments
	* [x] genImg: bash file allowing generate 
	* [x] rmOutputs: bash file allowing remove generated outputs of program


__clear binary__
```bash
make clean
```

__compile a new binary__
```bash
make cli
```

__execute the binary__
```bash
make run_cli
```

__generate Images__
```bash
make genImg
```

__remove assets__
```bash
make clearOutputs
```

the cli run command is the following: 
```bash
./kmeans_cli DataDimension DataExample ClustersNumber MaxIter Threshold ThreadsNumber genImgFlag
```
where:
	+ DataDimension : the number of attibut representing the a point 
	+ DataExample : the number of points
	+ ClustersNumber : the number of desired cluster
	+ MaxIter : the max iteration value
	+ Threshold : convergence threshold which is different of 0.0
	+ ThreadsNumber : number of threads to launch
	+ genImgFlag : 0 if we want to generate files which aims in image clustering generation