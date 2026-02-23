# MLNA — Multi-Layer Network Analysis
> **Graph-based ML** · Automatic Categorical Attributes Selection and Class-Based Personalized Multilayer Graph Feature Engineering for Supervised Machine Learning

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Environment variables (.env)](#1-environment-variables-env)
  - [Bash environment (env.sh)](#2-bash-environment-envsh)
  - [Dataset configuration (config.ini)](#3-dataset-configuration-configini)
- [Usage](#usage)
  - [Quick start — all datasets](#quick-start--all-datasets)
  - [Single dataset](#single-dataset)
  - [Individual pipeline scripts](#individual-pipeline-scripts)
  - [Monitoring and logs](#monitoring-and-logs)
  - [Stopping experiments](#stopping-experiments)
- [Pipeline Architecture](#pipeline-architecture)
- [Graph Construction Modes](#graph-construction-modes)
- [Descriptors](#descriptors)
- [Datasets](#datasets)
- [Results Layout](#results-layout)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Overview

MLNA is a research pipeline that transforms tabular classification datasets into **multilayer graphs**, extracts **PageRank-based descriptors** (global and personalized), trains 8 ML classifiers on classic + graph-enriched features, and generates **LaTeX/PDF statistical comparison reports**.

**Key features:**
- Three graph modes: MLNA-1 (monolayer), MLNA-K (combinatorial pairs), MLNA-TOP-K (top selected variables)
- PageRank descriptors: global (GLO) and personalized (PER), with/without class nodes
- 8 classifiers trained with and without SMOTE; cost-sensitive evaluation per alpha value
- Parallel execution via detached `screen` sessions (one per alpha)
- Automated LaTeX report generation with cross-dataset statistical analysis
- Optional email notifications for long-running experiments

---

## Project Structure

```
M2_thesis/
├── mlna_experiment/                    # Main experiment directory
│   ├── scripts/                        # Pipeline scripts (run as Python modules)
│   │   ├── 01_data_preprocessing.py    # Load raw CSV, EDA, preprocess, stratified sample
│   │   ├── 02_data_split.py            # Train/test split satisfying a performance threshold
│   │   ├── 03_graph_construction.py    # Build multilayer graphs; extract PageRank descriptors
│   │   ├── 04_model_training.py        # Train 8 classifiers; evaluate with/without SMOTE
│   │   ├── 05_report_generation.py     # Aggregate results; produce LaTeX/PDF reports
│   │   └── cpu_limitation_usage.py     # CPU resource management
│   ├── modules/                        # Core functionality modules
│   │   ├── graph.py                    # Multilayer graph construction (NetworkX), PageRank
│   │   ├── modeling.py                 # Classifier training, SMOTE, evaluation, SHAP
│   │   ├── preprocessing.py            # Data cleaning, encoding, combination generation
│   │   ├── statistical.py              # Cross-dataset analysis; result aggregation
│   │   ├── report.py                   # Report generation helpers
│   │   ├── eda.py                      # Exploratory data analysis
│   │   ├── file.py                     # File I/O utilities
│   │   ├── mailing.py                  # Email notifications (SMTP/Gmail)
│   │   └── env.py                      # .env loader
│   ├── configs/                        # Dataset-specific configurations
│   │   └── <DatasetName>/
│   │       └── config.ini
│   ├── data/                           # Input datasets
│   │   └── raw/<DatasetName>/          # Raw CSV per dataset
│   ├── results/                        # Experiment outputs (per dataset / alpha / fold)
│   ├── reports/                        # Generated LaTeX and PDF reports
│   ├── logs/                           # Execution logs
│   ├── .env_mlna/                      # Python virtual environment
│   ├── requirements.txt                # Python dependencies
│   ├── example.env                     # Template for .env
│   ├── env.sh                          # Bash environment variables (alphas, cwd)
│   ├── launch.sh                       # Main pipeline launcher (per dataset)
│   ├── env_setup.sh                    # One-time venv setup
│   ├── latex_install.sh                # LaTeX installation helper
│   ├── kill_screens.sh                 # Kill detached screen sessions
│   ├── stop.sh                         # Emergency stop
│   ├── Makefile                        # Convenience targets for all datasets
│   └── README.md                       # Internal quick-start reference
├── scripting/                          # Auxiliary scripts
├── BRAINSTORMING.md                    # Architecture analysis and optimization notes
├── CHANGELOG.md                        # Version history
├── CLAUDE.md                           # Claude Code project guide
├── .gitignore
├── LICENSE
└── README.md                           # This file
```

---

## Prerequisites

**Required:**
- Python 3.9 or higher
- pip
- GNU Screen (for background parallel execution)
- GNU Make
- Bash shell (Linux / macOS)
- GNU Parallel (Linux / macOS)

**Optional:**
- LaTeX distribution for PDF report generation
  - TeX Live (Linux): `sudo apt-get install texlive-full`
  - MacTeX (macOS): `brew install --cask mactex`
- Gmail account with an App Password (for email notifications)

**Recommended system resources:**
- 8 GB RAM minimum (16 GB+ for large datasets)
- Multi-core CPU — parallelism is controlled via `MAX_CORE` in `.env`
- 10 GB+ free disk space

---

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:VictorNico/multilayer-graph-feature-extraction.git
cd multilayer-graph-feature-extraction
```

### 2. Install system dependencies

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install -y screen make python3 python3-venv python3-pip
```

**macOS:**
```bash
brew install screen make python3
```

### 3. Install LaTeX (optional — required for PDF reports)

```bash
cd mlna_experiment
chmod +x latex_install.sh && ./latex_install.sh
```

### 4. Set up the Python virtual environment

```bash
cd mlna_experiment
chmod +x env_setup.sh && ./env_setup.sh
```

This creates `.env_mlna/` and installs all dependencies from `requirements.txt`. The environment is activated at the end of the script.

**Manual alternative:**
```bash
python3 -m venv .env_mlna
source .env_mlna/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

---

## Configuration

All commands below are run from `mlna_experiment/`.

### 1. Environment variables (`.env`)

Copy the template and fill in your values:

```bash
cp example.env .env
```

`.env` template:

```dotenv
# ==============================================
# EMAIL CONFIGURATION
# ==============================================

# Gmail credentials
GMAIL_USER=your.address@gmail.com
GMAIL_APP_PASSWORD=your_16_char_app_password

# SMTP settings (defaults to Gmail)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Cost-sensitive alpha values (used by launch.sh)
ALPHAS="0.20, 0.50, 0.80"

# ==============================================
# EMAIL RECIPIENTS
# ==============================================

# Primary recipients (comma-separated)
EMAIL_RECIPIENTS=Name1 <email1@example.com>,Name2 <email2@example.com>

# CC recipients (optional)
EMAIL_CC=Name3 <email3@example.com>

# ==============================================
# COMPUTE RESOURCES
# ==============================================

MAX_CORE=5          # Maximum CPU cores per process
SIZE_DIVIDER=2      # Memory management: graph size divisor
```

> To enable email notifications, generate a Gmail App Password at
> [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
> (requires 2-step verification).

---

### 2. Bash environment (`env.sh`)

```bash
cwd="$(pwd)"
alphas=(0.20 0.50 0.80)
graphWithClass=("True" "False")
```

Edit `alphas` here to change which alpha values are tested in parallel.

---

### 3. Dataset configuration (`config.ini`)

Each dataset requires `configs/<DatasetName>/config.ini`. Template based on the actual configs:

```ini
[GENERAL]
verbose = true
processed_dir = data/processed/
split_dir = data/splits/
results_dir = results/
report_dir = reports/
target_columns_type = cat

[DATA]
raw_path = data/raw/<DatasetName>/all.csv
domain = <3-LETTER-CODE>          # e.g. ADU, NUR, BAN
target = class                    # target column name in the CSV
dataset_delimiter = ,
index_col = None
na_values = ""
size = <total_row_count>

[PREPROCESSING]
to_remove =                       # comma-separated columns to drop (leave blank if none)
portion = <fraction>              # stratified fraction to keep, e.g. 0.077 for ~1 000 rows
encoding = utf-8

[SPLIT]
dataset_delimiter = ,
test_size = 0.2
random_state = 42
max_perf = 0.95                   # baseline accuracy threshold for split acceptance
index_col = 0

[GRAPH]
layers = 1
ohe_columns = <n_categorical>     # number of one-hot-encoded (categorical) features

[TRAINING]
cost = False
financialOption = {'amount': '<amount_col>', 'rate': '<rate_col>', 'duration': '<duration_col>'}
duration_divider = 12
rate_divider = 100

[REPORT]
result_path = results/evaluation_results.csv
send_email = false
recipient = user@example.com
shap_top = 10
```

**Adult dataset example (`configs/Adult/config.ini`):**

```ini
[GENERAL]
verbose = true
processed_dir = data/processed/
split_dir = data/splits/
results_dir = results/
report_dir = reports/
target_columns_type = cat

[DATA]
raw_path = data/raw/Adult/all.csv
domain = ADU
target = class
dataset_delimiter = ,
index_col = None
na_values = ""
size = 48842

[PREPROCESSING]
to_remove =
portion = 0.0204741921
encoding = utf-8

[SPLIT]
dataset_delimiter = ,
test_size = 0.2
random_state = 42
max_perf = 0.95
index_col = 0

[GRAPH]
layers = 1
ohe_columns = 7

[TRAINING]
cost = False
financialOption = {'amount': 'loan_amnt', 'rate': 'loan_int_rate', 'duration': 'cb_person_cred_hist_length'}
duration_divider = 12
rate_divider = 100

[REPORT]
result_path = results/evaluation_results.csv
send_email = false
recipient = user@example.com
shap_top = 10
```

---

## Usage

All commands run from `mlna_experiment/`. Activate the venv first:

```bash
source .env_mlna/bin/activate
```

### Quick start — all datasets

```bash
make run-all
```

Runs the full pipeline sequentially for all datasets (Adult, BankMarketing, CarEvaluation, CreditRiskDataset, Diabetes, GermanCredit, LoanDataforDummyBank, LoanDefaultDataset, Mushroom, Nursery, StudentPerformance).

To change the pipeline mode before running, edit `STEP` in the `Makefile`:
- `STEP=2` — MLNA framework mode (default)
- `STEP=3` — Random combinatorial search mode

### Single dataset

**Via Makefile:**

| Dataset | Target |
|---------|--------|
| Adult | `make mlna_on_adu` |
| BankMarketing | `make mlna_on_ban` |
| CarEvaluation | `make mlna_on_car` |
| CreditRiskDataset | `make mlna_on_crd` |
| Diabetes | `make mlna_on_dia` |
| GermanCredit | `make mlna_on_ger` |
| LoanDataforDummyBank | `make mlna_on_ld4` |
| LoanDefaultDataset | `make mlna_on_ldd` |
| Mushroom | `make mlna_on_mus` |
| Nursery | `make mlna_on_nur` |
| StudentPerformance | `make mlna_on_stu` |

**Via launch script:**

```bash
./launch.sh <DatasetName> <STEP>
# STEP=2 : framework mode (default)
# STEP=3 : random combination search

# Examples
./launch.sh Adult 2
./launch.sh Nursery 3
```

### Individual pipeline scripts

Scripts are always called **as Python modules** from `mlna_experiment/`:

```bash
# Step 1 — Data preprocessing
# Args: --cwd (required), --dataset_folder (required)
python3 -m scripts.01_data_preprocessing \
    --cwd=$(pwd) \
    --dataset_folder=Adult

# Step 2 — Train/test split
# Args: --cwd (required), --dataset_folder (required)
python3 -m scripts.02_data_split \
    --cwd=$(pwd) \
    --dataset_folder=Adult

# Step 3 — Graph construction
# Args: --cwd (required), --dataset_folder (required), --alpha (required), --turn (required)
#       --graph_with_class (optional), --metric (optional)
python3 -m scripts.03_graph_construction \
    --cwd=$(pwd) \
    --dataset_folder=Adult \
    --alpha=0.50 \
    --turn=1 \
    --graph_with_class

# Step 4 — Model training
# Args: --cwd (required), --dataset_folder (required), --alpha (required), --turn (required)
#       --graph_with_class (optional), --baseline (optional), --metric (optional)
python3 -m scripts.04_model_training \
    --cwd=$(pwd) \
    --dataset_folder=Adult \
    --alpha=0.50 \
    --turn=1 \
    --graph_with_class

# Step 5 — Report generation
# Args: --cwd (required), --dataset_folder (required), --metric (optional)
python3 -m scripts.05_report_generation \
    --cwd=$(pwd) \
    --dataset_folder=Adult \
    --metric=""
```

**Flag reference:**

| Flag | Scripts | Type | Description |
|------|---------|------|-------------|
| `--cwd` | 01 02 03 04 05 | required | Working directory (use `$(pwd)`) |
| `--dataset_folder` | 01 02 03 04 05 | required | Dataset folder name (e.g. `Adult`) |
| `--alpha` | 03 04 | required | PageRank damping factor: `0.20`, `0.50`, `0.80` |
| `--turn` | 03 04 | required | `1` = MLNA-1 · `2` = MLNA-TOP-K · `3` = MLNA-K |
| `--graph_with_class` | 03 04 | flag | Include target class nodes in the graph (CX descriptors) |
| `--baseline` | 04 | flag | Train on raw features only (no graph descriptors) |
| `--metric` | 03 04 05 | optional | Filter by metric: `"accuracy"`, `"f1-score"`, or `""` for all |

### Monitoring and logs

```bash
# List recent logs
make show-logs

# Follow the most recent log in real time
make tail-latest

# Summary of today's runs
make daily-summary

# List active screen sessions
screen -ls

# Attach to a session
screen -r <session_name>
# Detach: Ctrl+A then D
```

### Stopping experiments

```bash
# Stop a specific dataset
make stop-mlna_on_adu      # Adult
make stop-mlna_on_nur      # Nursery
# etc.

# Stop all datasets
make stop-all

# Clean logs older than 7 days
make clean-old-logs
```

---

## Pipeline Architecture

```
01_data_preprocessing
        ↓
02_data_split
        ↓
03_graph_construction   ← builds multilayer graph + extracts PageRank descriptors
        ↓                  (run in parallel per alpha via detached screen sessions)
04_model_training       ← trains 8 classifiers on classic + graph-enriched features
        ↓
05_report_generation    ← aggregates all results; produces LaTeX/PDF reports
```

A file `model_turn_2_completed.dtvni` is written at the end of step 4. Step 5 checks for this flag before aggregating results.

| Script | Input | Output |
|--------|-------|--------|
| `01_data_preprocessing.py` | `data/raw/<Dataset>/all.csv` | `data/processed/<Dataset>/` |
| `02_data_split.py` | Preprocessed data | `data/splits/<Dataset>/` |
| `03_graph_construction.py` | Splits + config | `results/<Dataset>/<alpha>/.../` |
| `04_model_training.py` | Splits + descriptors | `results/<Dataset>/<alpha>/.../` |
| `05_report_generation.py` | All results | `reports/` |

---

## Graph Construction Modes

| Mode | Flag `--turn` | Description |
|------|--------------|-------------|
| **MLNA-1** | `1` | One monolayer per variable (each categorical column becomes a layer) |
| **MLNA-TOP-K** | `2` | Top-K variables selected from MLNA-1 results form combined layers |
| **MLNA-K** | `3` | Combinatorial — k=2 variables per layer (all pairs) |

The `--graph_with_class` flag adds the target class as an extra graph layer, producing **CX** (borrower + class) vs **MX** (borrower only) descriptor variants.

---

## Descriptors

For each graph layer configuration, six PageRank descriptor sets are extracted:

| Code | Type | Description |
|------|------|-------------|
| `MX_GLO` | Global | PageRank on graph **without** class nodes — global personalization |
| `MX_PER` | Personalized | PageRank on graph **without** class nodes — per-instance personalization |
| `CX_GLO` | Global | PageRank on graph **with** class nodes — global personalization |
| `CX_PER` | Personalized | PageRank on graph **with** class nodes — per-instance personalization |
| `CY` | Class-only | PageRank restricted to class-layer nodes |
| `CXY` | All | Combined (CX + CY) |

Combined feature sets assembled from the above:
- **GAP** = GLO + PER merged
- **BOT** = MX + CX merged (both class configurations)

---

## Datasets

| Dataset | Domain | Raw size | Config code |
|---------|--------|---------|-------------|
| Adult | Census income | 48 842 | `ADU` |
| Audiology | Healthcare | 192 | `AUD` |
| BankMarketing | Marketing campaign | 4 520 | `BAN` |
| CarEvaluation | Automotive | 1 727 | `CAR` |
| CreditRiskDataset | Credit risk | 32 580 | `CRD` |
| Diabetes | Healthcare | 101 765 | `DIA` |
| GermanCredit | Credit risk | 1 000 | `GER` |
| LoanDataforDummyBank | Banking | 13 648 | `LD4` |
| LoanDefaultDataset | Banking | 12 242 | `LDD` |
| Mushroom | Biology | 8 123 | `MUS` |
| Nursery | Social | 12 957 | `NUR` |
| StudentPerformance | Education | 648 | `STU` |

Data sources: UCI Machine Learning Repository, Kaggle.

**Adding a new dataset:**

1. Place the CSV at `data/raw/<DatasetName>/all.csv`
2. Create `configs/<DatasetName>/config.ini` (use the template above)
3. Add a Makefile target following the existing pattern (optional)
4. Run: `./launch.sh <DatasetName> 2`

---

## Results Layout

```
results/
└── <DatasetName>/
    └── <alpha>/                          # e.g. 0.20, 0.50, 0.80
        ├── cat/                          # target_columns_type
        │   ├── withClass/
        │   │   └── mlna_1/
        │   │       └── <variable>/
        │   │           ├── global/
        │   │           │   └── withClass/
        │   │           │       └── evaluation/
        │   │           │           └── *_metric_*.pkl
        │   │           ├── personalized/
        │   │           └── mixed/
        │   └── withoutClass/
        │       └── ...
        └── model_turn_2_completed.dtvni  # completion flag for step 5
```

**Metrics evaluated per model:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Cost-sensitive score: `FPR × alpha + FNR × (1 − alpha)`
- Confusion matrix
- SHAP feature importances (top-N configurable via `shap_top` in config)

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `venv not found` | `rm -rf .env_mlna && ./env_setup.sh` |
| `Permission denied` on scripts | `chmod +x launch.sh env_setup.sh kill_screens.sh stop.sh` |
| Dead screen sessions blocking | `screen -wipe` |
| `Out of memory` errors | Reduce `MAX_CORE` or increase `SIZE_DIVIDER` in `.env` |
| LaTeX compilation fails | `./latex_install.sh` |
| `ImportError` after update | `pip install -r requirements.txt --force-reinstall` |
| `TypeError: 'NoneType' object is not subscriptable` (step 4) | Ensure `config_df2` is loaded — see BRAINSTORMING.md §4 |

**Log locations:**
- Pipeline logs: `logs/*.log`
- Per-dataset logs: named `<Dataset>_<YYYYMMDD_HHMMSS>.log`

[//]: # (---)

[//]: # ()
[//]: # (## Citation)

[//]: # ()
[//]: # (If you use this code in your research, please cite:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@mastersthesis{MLNA2025,)

[//]: # (  title   = {Prediction of Cost-Sensitive Classification Using Descriptors)

[//]: # (             Extracted from Multilayer Graphs},)

[//]: # (  author  = {Djiembouti Encth Nico Victor},)

[//]: # (  year    = {2025},)

[//]: # (  school  = {University of Yaoundé I})

[//]: # (})

[//]: # (```)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Last updated: February 2026 — version tracked in [CHANGELOG.md](CHANGELOG.md)*