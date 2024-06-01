run_on_credit_risk_full:utils/decision_tree.py utils/file.py utils/metrics.py
	source ../../../../scripting/.env/bin/activate && python3.9 main.py --input_file "data/train.csv" --delimiter ","  --is_header --class_name "loan_status" --test_file "data/test.csv" --is_classification --max_depth '-1' --min_samples_split 2 --min_information_gain 0.00000000000000000001 --max_categories 20

run_on_german_full:utils/decision_tree.py utils/file.py utils/metrics.py
	source ../../../../scripting/.env/bin/activate && python3.9 main.py --input_file "data/train1.csv" --delimiter ","  --is_header --class_name "Class" --test_file "data/test1.csv" --is_classification --max_depth '-1' --min_samples_split 2 --min_information_gain 0.00000000000000000001 --max_categories 20
