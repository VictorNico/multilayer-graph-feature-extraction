produit_matriciel:produit_matriciel.cpp gui.cpp matrix.cpp file.cpp time.cpp
	g++ --std=c++20 -o produit_matriciel_cpp produit_matriciel.cpp gui.cpp matrix.cpp time.cpp file.cpp -g -O0 -lm -lpthread
# 	g++ --std=c++11 -o produit_matriciel produit_matriciel.cpp -g -O0 -lm -lpthread -fopenmp 

run_produit_matriciel:produit_matriciel
	./produit_matriciel_cpp

# make sure to give x permission to gp files
compare_bloc_vs_modulo:comparaison_bloc_vs_modulo.gp
	gnuplot "comparaison_bloc_vs_modulo.gp"

compare_bloc_vs_modulo_vs_nthreads:comparaison_bloc_vs_modulo_vs_nthreads.gp
	gnuplot "comparaison_bloc_vs_modulo_vs_nthreads.gp"

compare_seq_vs_modulo:comparaison_seq_vs_modulo.gp
	gnuplot "comparaison_seq_vs_modulo.gp"

compare_bloc_vs_seq:comparaison_bloc_vs_seq.gp
	gnuplot "comparaison_bloc_vs_seq.gp"

clean:
	rm  produit_matriciel_cpp

