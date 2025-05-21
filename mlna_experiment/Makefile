# run our framework on credit risk dataset
mlna_on_credit_risk_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh credit_risk mlna_on_credit_risk_dataset.py
# stop our framework on credit risk dataset
stop-mlna_on_credit_risk_dataset:
	sudo chmod +x stop.sh && sudo ./stop.sh credit_risk mlna_on_credit_risk_dataset.py

# run our framework on Loan Default Data
mlna_on_ldd:
	sudo chmod +x launch.sh && sudo ./launch.sh ldd mlna_on_mlna_on_Loan_Default_Dataset.py
# stop our framework on Loan Default Data
stop-mlna_on_ldd:
	sudo chmod +x stop.sh && sudo ./stop.sh ldd mlna_on_mlna_on_Loan_Default_Dataset.py

# run our framework on Loan Data for Dummy Bank
mlna_on_ld4db:
	sudo chmod +x launch.sh && sudo ./launch.sh ld4db mlna_on_mlna_on_Loan_Data_for_Dummy_Bank.py
# stop our framework on Loan Data for Dummy Bank
stop-mlna_on_ld4db:
	sudo chmod +x stop.sh && sudo ./stop.sh ld4db mlna_on_mlna_on_Loan_Data_for_Dummy_Bank.py

# run our framework on AUSTRALIAN dataset
mlna_on_australian_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh australian mlna_on_australian_dataset.py
# stop our framework on AUSTRALIAN dataset
stop-mlna_on_australian_dataset:
	sudo chmod +x stop.sh && sudo ./stop.sh australian mlna_on_australian_dataset.py

# run our framework on GERMAN dataset
mlna_on_german_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh ger mlna_on_german_dataset.py
# stop our framework on GERMAN dataset
stop-mlna_on_german_dataset:
	sudo chmod +x stop.sh && sudo ./stop.sh ger mlna_on_german_dataset.py

# run some execution in parallel
# --jobs 4: can run 4 process in parallel
# CUDA_VISIBLE_DEVICES=0: ALLOW TO USE GPU
parallel:
	parallel --eta --jobs 2 "CUDA_VISIBLE_DEVICES=0 python3.9 {}" ::: mlna_on_aer_kaggle_dataset.py mlna_on_thomas_dataset.py mlna_on_credit_risk_dataset.py mlna_on_afb_dataset.py mlna_on_australian_dataset.py mlna_on_german_dataset.py mlna_on_japan_dataset.py mlna_on_taiwan_dataset.py

# run parallel execution by default
run-all:mlna_on_german_dataset mlna_on_ldd mlna_on_ld4db
	# mlna_on_credit_risk_dataset mlna_on_afb_dataset mlna_on_german_dataset
	# mlna_on_japan_dataset mlna_on_aer_kaggle_dataset mlna_on_thomas_dataset

stop-all: stop-mlna_on_german_dataset stop-mlna_on_ldd stop-mlna_on_ld4db
	# stop-mlna_on_credit_risk_dataset stop-mlna_on_afb_dataset stop-mlna_on_german_dataset
	# stop-mlna_on_japan_dataset stop-mlna_on_aer_kaggle_dataset stop-mlna_on_thomas_dataset
