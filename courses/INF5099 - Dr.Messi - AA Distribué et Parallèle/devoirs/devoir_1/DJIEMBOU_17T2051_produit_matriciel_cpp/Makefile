produit_matriciel:produit_matriciel.cpp gui.cpp matrix.cpp
	g++ --std=c++11 -o produit_matriciel_cpp produit_matriciel.cpp gui.cpp matrix.cpp -g -O0 -lm -lpthread
# 	g++ --std=c++11 -o produit_matriciel produit_matriciel.cpp -g -O0 -lm -lpthread -fopenmp 

run_produit_matriciel:produit_matriciel
	./produit_matriciel_cpp

clean:
	rm  produit_matriciel_cpp

