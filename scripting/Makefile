# run our framework on credit risk dataset
mlna_on_credit_risk_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh credit_risk mlna_on_credit_risk_dataset.py
# stop our framework on credit risk dataset
stop-mlna_on_credit_risk_dataset:
	sudo chmod +x stop.sh && sudo ./stop.sh credit_risk mlna_on_credit_risk_dataset.py

# run our framework on AUSTRALIAN dataset
mlna_on_australian_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh australian mlna_on_australian_dataset.py
# stop our framework on AUSTRALIAN dataset
stop-mlna_on_australian_dataset:
	sudo chmod +x stop.sh && sudo ./stop.sh australian mlna_on_australian_dataset.py

# run our framework on GERMAN dataset
mlna_on_german_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh german mlna_on_german_dataset.py
# stop our framework on GERMAN dataset
stop-mlna_on_german_dataset:
	sudo chmod +x stop.sh && sudo ./stop.sh german mlna_on_german_dataset.py

# run our framework on JAPAN dataset
mlna_on_japan_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh japan mlna_on_japan_dataset.py
# stop our framework on JAPAN dataset
stop-mlna_on_japan_dataset:
	sudo chmod +x stop.sh && sudo ./stop.sh japan mlna_on_japan_dataset.py

# run our framework on TAIWAN dataset
mlna_on_taiwan_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh taiwan mlna_on_taiwan_dataset.py
# stop our framework on TAIWAN dataset
stop-mlna_on_taiwan_dataset:
	sudo chmod +x stop.sh && sudo ./stop.sh taiwan mlna_on_taiwan_dataset.py

# run our framework on AFB dataset
mlna_on_afb_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh afb mlna_on_afb_dataset.py
# stop our framework on AFB dataset
stop-mlna_on_afb_dataset:
	sudo chmod +x stop.sh && sudo ./stop.sh afb mlna_on_afb_dataset.py

# run our framework on AER dataset
mlna_on_aer_kaggle_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh aer mlna_on_aer_kaggle_dataset.py
# stop our framework on AER dataset
stop-mlna_on_aer_kaggle_dataset:
	sudo chmod +x stop.sh && sudo ./stop.sh aer mlna_on_aer_kaggle_dataset.py

# run our framework on Thomas dataset
mlna_on_thomas_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh thomas mlna_on_thomas_dataset.py
# stop our framework on Thomas dataset
stop-mlna_on_thomas_dataset:
	sudo chmod +x launch.sh && sudo ./launch.sh thomas mlna_on_thomas_dataset.py

# run some execution in parallel
# --jobs 4: can run 4 process in parallel
# CUDA_VISIBLE_DEVICES=0: ALLOW TO USE GPU
parallel:
	parallel --eta --jobs 2 "CUDA_VISIBLE_DEVICES=0 python3.9 {}" ::: mlna_on_aer_kaggle_dataset.py mlna_on_thomas_dataset.py mlna_on_credit_risk_dataset.py mlna_on_afb_dataset.py mlna_on_australian_dataset.py mlna_on_german_dataset.py mlna_on_japan_dataset.py mlna_on_taiwan_dataset.py

# run parallel execution by default
run-all: mlna_on_credit_risk_dataset mlna_on_afb_dataset mlna_on_german_dataset mlna_on_japan_dataset mlna_on_aer_kaggle_dataset mlna_on_thomas_dataset

stop-all: stop-mlna_on_credit_risk_dataset stop-mlna_on_afb_dataset stop-mlna_on_german_dataset stop-mlna_on_japan_dataset stop-mlna_on_aer_kaggle_dataset stop-mlna_on_thomas_dataset