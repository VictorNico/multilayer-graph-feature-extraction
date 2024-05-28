Dtree:test.cpp files.cpp metrics.cpp decisionTree.cpp
	g++ --std=c++20 -o DTree main.cpp files.cpp metrics.cpp decisionTree.cpp -g -O0 -lm -lpthread
# 	g++ --std=c++11 -o produit_matriciel produit_matriciel.cpp -g -O0 -lm -lpthread -fopenmp

run_DTree:DTree
	./DTree "data/german.csv" "," 1 "Class"

run_DTree_full:DTree
	./DTree "data/train.csv" "," 1 "Class" "data/test.csv" 1 '-1' 2 0.00000000000000000001 20

clean:
	rm  DTree

