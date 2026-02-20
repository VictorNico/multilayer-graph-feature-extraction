# M2 THESIS SOURCE CODE
> PREDICTION OF COST-SENSITIVE BANK CREDIT RISK USING DESCRIPTORS EXTRACTED FROM GRAPHS

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Datasets](#datasets)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a machine learning framework for predicting cost-sensitive bank credit risk using graph-based descriptors. The pipeline includes data preprocessing, graph construction, model training, and comprehensive reporting with statistical analysis.

**Key Features:**
- Automated end-to-end ML pipeline for credit risk prediction
- Graph-based feature extraction from tabular data
- Cost-sensitive learning with configurable alpha parameters
- Support for multiple benchmark datasets
- Parallel processing with CPU core management
- Automated LaTeX report generation
- Email notifications for long-running experiments

## Project Structure

```
M2_Thesis/
├── mlna_experiment/                    # Main experiment directory
│   ├── scripts/                        # Pipeline execution scripts
│   │   ├── 01_data_preprocessing.py   # Data loading, EDA, preprocessing
│   │   ├── 02_data_split.py           # Train/test split with class balance
│   │   ├── 03_graph_construction.py   # Graph creation and descriptor extraction
│   │   ├── 04_model_training.py       # Model training and evaluation
│   │   ├── 05_report_generation.py    # Results analysis and LaTeX report
│   │   └── cpu_limitation_usage.py    # CPU resource management
│   ├── modules/                        # Core functionality modules
│   │   ├── preprocessing.py           # Data preprocessing utilities
│   │   ├── graph.py                   # Graph construction algorithms
│   │   ├── modeling.py                # ML model implementations
│   │   ├── statistical.py             # Statistical analysis tools
│   │   ├── report.py                  # Report generation
│   │   ├── mailing.py                 # Email notification system
│   │   ├── eda.py                     # Exploratory data analysis
│   │   ├── file.py                    # File I/O operations
│   │   └── env.py                     # Environment configuration
│   ├── configs/                        # Dataset-specific configurations
│   │   └── {dataset_name}/
│   │       └── config.ini             # Configuration file per dataset
│   ├── data/                          # Input datasets
│   │   └── {dataset_name}/            # Dataset directory
│   ├── results/                       # Experiment results
│   ├── reports/                       # Generated reports
│   ├── logs/                          # Execution logs
│   ├── .env_mlna/                     # Python virtual environment
│   ├── requirements.txt               # Python dependencies
│   ├── launch.sh                      # Main pipeline launcher
│   ├── env_setup.sh                   # Environment setup script
│   ├── env.sh                         # Bash environment variables
│   ├── Makefile                       # Convenient execution commands
│   ├── kill_screens.sh                # Terminate detached screen sessions
│   ├── latex_install.sh               # LaTeX installation (optional)
│   ├── stop.sh                        # Stop running experiments
│   └── README.md                      # Detailed documentation
├── scripting/                         # Additional scripts
├── .gitignore                         # Git ignore rules
├── LICENSE                            # MIT License
└── README.md                          # This file

```

## Prerequisites

**Required:**
- Python 3.8 or higher
- pip (Python package manager)
- GNU Screen (for background processing)
- GNU Parallel (for parallel execution)
- Bash shell (Linux/macOS)

**Optional:**
- LaTeX distribution (for PDF report generation)
  - TeX Live (Linux)
  - MacTeX (macOS)
- Gmail account with App Password (for email notifications)

**System Requirements:**
- Minimum 8GB RAM (16GB+ recommended)
- Multi-core CPU (configurable via MAX_CORE in .env)
- 10GB+ free disk space

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:VictorNico/multilayer-graph-feature-extraction.git
cd multilayer-graph-feature-extraction/mlna_experiment
```

### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y screen parallel python3 python3-venv python3-pip
```

**macOS:**
```bash
brew install screen parallel python3
```

### 3. Install LaTeX (Optional but Recommended)

For automated PDF report generation:

```bash
chmod +x latex_install.sh
./latex_install.sh
```

This installs the necessary LaTeX packages for compiling reports.

### 4. Set Up Python Environment

Run the automated setup script:

```bash
chmod +x env_setup.sh
./env_setup.sh
```

This will:
- Create a virtual environment at `.env_mlna/`
- Install all Python dependencies from `requirements.txt`
- Activate the environment

**Manual Installation (Alternative):**
```bash
python3 -m venv .env_mlna
source .env_mlna/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

## Configuration

### 1. Environment Variables

Create a `.env` file from the example template:

```bash
cp example.env .env
```

Edit `.env` with your configuration:

```bash
# Email Configuration (Optional)
GMAIL_USER=your.email@gmail.com
GMAIL_APP_PASSWORD=your_app_password
EMAIL_RECIPIENTS=Recipient1 <email1@example.com>,Recipient2 <email2@example.com>
EMAIL_CC=CC1 <cc1@example.com>

# SMTP Settings
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Experiment Parameters
ALPHAS="0.20, 0.50, 0.80"  # Cost-sensitive alpha values

# Resource Management
MAX_CORE=5                   # Maximum CPU cores to use
SIZE_DIVIDER=2               # Memory management parameter
```

**To enable email notifications:**
1. Enable 2-factor authentication on your Gmail account
2. Generate an App Password: [Google Account Settings](https://myaccount.google.com/apppasswords)
3. Add credentials to `.env`

### 2. Bash Environment

The `env.sh` file contains bash-specific variables:

```bash
cwd="$(pwd)"
alphas=(0.20 0.50 0.80)      # Alpha values for cost-sensitive learning
graphWithClass=("True" "False")
```

### 3. Dataset Configuration

Each dataset requires a `config.ini` file in `configs/{dataset_name}/config.ini`:

```ini
[DATASET]
name = Adult
file_path = data/Adult/adult.csv
target_column = income
positive_class = >50K

[PREPROCESSING]
handle_missing = drop
encoding = onehot
scaling = standard

[GRAPH]
similarity_metric = euclidean
k_neighbors = 5
```

## Usage

### Quick Start

**Run all datasets in parallel:**

```bash
cd mlna_experiment
make run-all
```

This launches experiments for all configured datasets in detached screen sessions.

### Running Individual Datasets

**Using Makefile targets:**

```bash
# Adult dataset
make mlna_on_adu

# Bank Marketing dataset
make mlna_on_ban

# German Credit dataset
make mlna_on_ger

# See all available targets
make help
```

**Using launch script directly:**

```bash
./launch.sh <dataset_folder_name> <turn_parameter>
```

Example:
```bash
./launch.sh Adult 2
```

**Parameters:**
- `dataset_folder_name`: Name of folder in `data/` and `configs/`
- `turn_parameter`: Number of iterations (default: 2 for framework, 3 for random search)

### Monitoring Experiments

**List running screen sessions:**
```bash
screen -ls
```

**Attach to a running session:**
```bash
screen -r <session_name>
```

**Detach from session:**
Press `Ctrl+A`, then `D`

**View logs:**
```bash
# Real-time log monitoring
make tail-adu  # For Adult dataset

# Or manually
tail -f logs/Adult/Adu_a20_*.log
```

### Stopping Experiments

**Stop all experiments for a dataset:**
```bash
make stop-mlna_on_adu  # For Adult
```

**Kill all screen sessions:**
```bash
./kill_screens.sh Adult
```

**Emergency stop all:**
```bash
chmod +x stop.sh
./stop.sh
```

## Pipeline Steps

The pipeline consists of 5 main stages:

### Step 1: Data Preprocessing
**Script:** `01_data_preprocessing.py`

- Load raw dataset
- Exploratory Data Analysis (EDA)
- Handle missing values
- Encode categorical features
- Remove rare classes (< 10 samples)
- Save preprocessed data

**Output:** `data/{dataset}/preprocessed_data.csv`

### Step 2: Train/Test Split
**Script:** `02_data_split.py`

- Stratified split maintaining class distribution
- Ensure minority class representation
- Validate split quality
- Save train/test indices

**Output:** `data/{dataset}/train_indices.csv`, `test_indices.csv`

### Step 3: Graph Construction
**Script:** `03_graph_construction.py`

- Construct k-NN graphs from tabular data
- Extract graph-based descriptors:
  - Degree centrality
  - Betweenness centrality
  - Clustering coefficient
  - PageRank
  - Community detection features
- Create variants: with/without class labels
- Run in parallel for different alpha values

**Output:** `results/{dataset}/alpha_{value}/graph_descriptors.csv`

### Step 4: Model Training
**Script:** `04_model_training.py`

- Train baseline models (no graph features)
- Train models with graph descriptors
- Models used:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
- Cost-sensitive learning with configurable alpha
- Cross-validation and hyperparameter tuning
- Performance evaluation

**Output:** `results/{dataset}/alpha_{value}/model_results.pkl`

### Step 5: Report Generation
**Script:** `05_report_generation.py`

- Aggregate results across experiments
- Statistical significance testing
- Generate LaTeX tables and figures
- Compile PDF report
- Send email notification (if configured)

**Output:** `reports/{dataset}_report.pdf`

## Datasets

The framework supports multiple benchmark datasets:

| Dataset | Domain | Samples | Features | Class Balance |
|---------|--------|---------|----------|---------------|
| Adult | Census Income | 48,842 | 14 | Imbalanced |
| BankMarketing | Marketing | 45,211 | 16 | Imbalanced |
| GermanCredit | Credit Risk | 1,000 | 20 | Imbalanced |
| CreditRiskDataset | Credit Risk | 32,581 | 11 | Imbalanced |
| Diabetes | Healthcare | 768 | 8 | Imbalanced |
| StudentPerformance | Education | 649 | 30 | Balanced |
| Mushroom | Biology | 8,124 | 22 | Balanced |
| Nursery | Social | 12,960 | 8 | Imbalanced |
| CarEvaluation | Automotive | 1,728 | 6 | Imbalanced |

**Adding a New Dataset:**

1. Place data file in `data/{dataset_name}/`
2. Create config file in `configs/{dataset_name}/config.ini`
3. Add Makefile target (optional)
4. Run: `./launch.sh {dataset_name} 2`

## Results

Results are organized by dataset and alpha value:

```
results/
└── {dataset_name}/
    ├── baseline/
    │   └── model_results.pkl
    ├── alpha_0.20/
    │   ├── graph_descriptors.csv
    │   ├── model_results.pkl
    │   └── performance_metrics.json
    ├── alpha_0.50/
    └── alpha_0.80/
```

**Key Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Cost-sensitive metrics (FPR × alpha + FNR × (1-alpha))
- Confusion matrices
- Feature importance rankings

## Troubleshooting

### Common Issues

**1. Virtual environment not found:**
```bash
rm -rf .env_mlna
./env_setup.sh
```

**2. Screen session won't start:**
```bash
# Clean up dead screens
screen -wipe
```

**3. Permission denied:**
```bash
chmod +x launch.sh env_setup.sh kill_screens.sh
```

**4. Out of memory:**
- Reduce `MAX_CORE` in `.env`
- Increase `SIZE_DIVIDER` in `.env`

**5. LaTeX compilation fails:**
```bash
# Reinstall LaTeX packages
./latex_install.sh
```

**6. Import errors:**
```bash
source .env_mlna/bin/activate
pip install -r requirements.txt --force-reinstall
```

### Debug Mode

Enable verbose logging:
```bash
export DEBUG=1
./launch.sh Adult 2
```

### Logs Location

- Main logs: `logs/{dataset}/*.log`
- Screen logs: `logs/{dataset}/{screen_name}_*.log`
- Error logs: Check stderr in log files

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{thesis2025,
  title={Prediction of Cost-Sensitive Bank Credit Risk Using Descriptors Extracted from Graphs},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [your.email@example.com]

## Acknowledgments

- Dataset sources: UCI Machine Learning Repository, Kaggle
- Graph algorithms: NetworkX library
- ML frameworks: scikit-learn, XGBoost
- Report generation: LaTeX, Python-LaTeX integration

---

**Last Updated:** January 2026
**Version:** 1.0.0
