# run our framework on credit risk dataset
mlna_on_credit_risk_dataset:
	python3.8 mlna_on_credit_risk_dataset.py

# run our framework on AFB dataset
mlna_on_afb_dataset:
	python3.8 mlna_on_afb_dataset.py

# run some execution in parallel
# --jobs 4: can run 4 process in parallel
# CUDA_VISIBLE_DEVICES=0: ALLOW TO USE GPU
parallel:
	parallel --eta --jobs 2 "CUDA_VISIBLE_DEVICES=0 python3.8 {}" ::: mlna_on_credit_risk_dataset.py mlna_on_afb_dataset.py

# run parallel execution by default
all: parallel
