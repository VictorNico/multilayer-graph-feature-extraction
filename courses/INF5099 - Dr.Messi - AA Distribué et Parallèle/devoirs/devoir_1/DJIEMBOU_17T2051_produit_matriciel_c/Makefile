produit_matriciel:produit_matriciel.c gui.c matrix.c
	gcc -o produit_matriciel_c produit_matriciel.c gui.c matrix.c -g -O0 -lm -lpthread
# 	gcc -o produit_matriciel_c produit_matriciel.c gui.c matrix.c -g -O0 -lm -lpthread -lncurses
# 	gcc --std=c++11 -o produit_matriciel produit_matriciel.c -g -O0 -lm -lpthread -fopenmp 

run_produit_matriciel:produit_matriciel
	./produit_matriciel_c

clean:
	rm  produit_matriciel_c

