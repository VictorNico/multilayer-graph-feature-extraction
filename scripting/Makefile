# run our framework on credit risk dataset
mlna_on_credit_risk_dataset:
	python3.8 mlna_on_credit_risk_dataset.py

# run our framework on AUSTRALIAN dataset
mlna_on_australian_dataset:
	python3.8 mlna_on_australian_dataset.py

# run our framework on GERMAN dataset
mlna_on_german_dataset:
	python3.8 mlna_on_german_dataset.py

# run our framework on JAPAN dataset
mlna_on_japan_dataset:
	python3.8 mlna_on_japan_dataset.py

# run our framework on TAIWAN dataset
mlna_on_taiwan_dataset:
	python3.8 mlna_on_taiwan_dataset.py

# run our framework on AFB dataset
mlna_on_afb_dataset:
	python3.8 mlna_on_afb_dataset.py

# run our framework on AER dataset
mlna_on_aer_kaggle_dataset:
	python3.8 mlna_on_aer_kaggle_dataset.py

# run our framework on Thomas dataset
mlna_on_thomas_dataset:
	python3.8 mlna_on_thomas_dataset.py

# run some execution in parallel
# --jobs 4: can run 4 process in parallel
# CUDA_VISIBLE_DEVICES=0: ALLOW TO USE GPU
parallel:
	parallel --eta --jobs 2 "CUDA_VISIBLE_DEVICES=0 python3.8 {}" ::: mlna_on_aer_kaggle_dataset.py mlna_on_thomas_dataset.py mlna_on_credit_risk_dataset.py mlna_on_afb_dataset.py mlna_on_australian_dataset.py mlna_on_german_dataset.py mlna_on_japan_dataset.py mlna_on_taiwan_dataset.py

# run parallel execution by default
all: parallel
