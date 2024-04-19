gui:clustering_gui.c gui.c kmeans.c file.c time.c
	gcc -o kmeans_gui clustering_gui.c gui.c kmeans.c file.c time.c -g -O0 -lm -lpthread

cli:clustering_cli.c cli.c kmeans.c file.c time.c
	gcc -o kmeans_cli clustering_cli.c cli.c kmeans.c file.c time.c -g -O0 -lm -lpthread

run_gui:kmeans_gui
	./kmeans_gui

run_cli:kmeans_cli
	./kmeans_cli 250 2000 3 144 0.001 2 0

# make sure to give x permission to gp files
genImg:genImg
	chmod +x genImg & ./genImg

clearOutputs:rmOutputs
	chmod +x rmOutputs & ./rmOutputs

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
	rm  kmeans_*

