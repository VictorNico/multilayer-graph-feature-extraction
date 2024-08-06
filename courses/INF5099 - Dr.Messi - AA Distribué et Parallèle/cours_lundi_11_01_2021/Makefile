produit_scalaire:produit_scalaire.c
	gcc -o produit_scalaire produit_scalaire.c -g -O0 -lm -lpthread -fopenmp 

run_produit_scalaire:produit_scalaire
	./produit_scalaire 10000 2
run_valgrind:produit_scalaire
	valgrind  --tool=memcheck --leak-check=yes --leak-resolution=low --show-reachable=yes  ./produit_scalaire 1000
clean:
	rm  produit_scalaire

profile:
	gcc -Wall -pg produit_scalaire.c -lm -lpthread -o produit_scalaire
