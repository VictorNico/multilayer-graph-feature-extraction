Dtree:test.cpp files.cpp metrics.cpp decisionTree.cpp
	g++ --std=c++20 -o DTree main.cpp files.cpp metrics.cpp decisionTree.cpp -g -O0 -lm -lpthread
# 	g++ --std=c++11 -o produit_matriciel produit_matriciel.cpp -g -O0 -lm -lpthread -fopenmp
# <input_file> <delimiter> <is_header>  <class_name>  [<test_file> <is_classification> <max_depth> <min_samples_split> <min_information_gain> <max_categories>]
run_on_german:DTree
	./DTree "data/german.csv" "," 1 "Class"

run_on_german_full:DTree
	./DTree "data/train1.csv" "," 1 "Class" "data/test1.csv" 1 '-1' 2 0.00000000000000000001 20

run_on_credit_risk:DTree
	 ./DTree "data/credit_risk_dataset.csv" "," 1 "loan_status"

run_on_credit_risk_full:DTree
	./DTree "data/train.csv" "," 1 "loan_status" "data/test.csv" 1 '-1' 2 0.00000000000000000001 20

clean:
	rm  DTree

