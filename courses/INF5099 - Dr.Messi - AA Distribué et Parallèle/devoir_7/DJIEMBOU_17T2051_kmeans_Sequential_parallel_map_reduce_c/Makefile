kmeans:clustering.c gui.c kmeans.c file.c time.c
	gcc -o kmeans_c clustering.c gui.c kmeans.c file.c time.c -g -O0 -lm -lpthread
# 	gcc -o produit_matriciel_c clustering.c gui.c kmeans.c -g -O0 -lm -lpthread -lncurses
# 	gcc --std=c++11 -o produit_matriciel clustering.c -g -O0 -lm -lpthread -fopenmp 

run_kmeans:kmeans_c
	./kmeans_c

# make sure to give x permission to gp files
compare_bloc_vs_modulo:comparaison_bloc_vs_modulo.gp
	gnuplot "comparaison_bloc_vs_modulo.gp"

compare_bloc_vs_modulo_vs_nthreads:comparaison_bloc_vs_modulo_vs_nthreads.gp
	gnuplot "comparaison_bloc_vs_modulo_vs_nthreads.gp"

compare_seq_vs_modulo:comparaison_seq_vs_modulo.gp
	gnuplot "comparaison_seq_vs_modulo.gp"

compare_bloc_vs_sec:comparaison_bloc_vs_seq.gp
	gnuplot "comparaison_bloc_vs_seq.gp"

cluster:cluster.gp
	gnuplot "cluster.gp"

clean:
	rm  kmeans_c

