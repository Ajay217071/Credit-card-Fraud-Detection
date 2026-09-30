# Credit Card Fraud Detection with Machine Learning

![Credit Card Fraud Detection](thumbnail.png)


An end-to-end machine-learning project for identifying potentially fraudulent credit-card transactions using **XGBoost** and **SMOTE**.

## Project Overview

Credit-card fraud detection is a highly imbalanced binary-classification problem. In this dataset, fraudulent transactions are a small minority compared with legitimate transactions, so accuracy alone can give a misleading impression of model performance.

This project explores the data, analyses missing values and feature relationships, prepares categorical and numerical variables, addresses class imbalance with SMOTE, and trains an XGBoost classifier.

## Dataset

The supplied dataset contains:

- **590,528 transactions**
- **201 columns** before preprocessing
- Target variable: `isFraud`
- Approximately **96.50% legitimate** transactions
- Approximately **3.50% fraudulent** transactions

The raw dataset is not included in this repository because of its size. See [`data/README.md`](data/README.md) for setup instructions.

## Workflow

```text
Raw transaction data
        |
        v
Exploratory Data Analysis
        |
        v
Missing-value analysis
        |
        v
Correlation analysis
        |
        v
Categorical encoding
        |
        v
Train / test split
        |
        v
StandardScaler
        |
        v
SMOTE oversampling
        |
        v
XGBoost classifier
        |
        v
Classification report + confusion matrix
```

## Machine-Learning Approach

### 1. Exploratory Data Analysis

The notebook examines:

- Dataset dimensions
- Data types
- Summary statistics
- Missing values
- Categorical cardinality
- Fraud-class distribution
- Correlations between numerical variables

### 2. Missing Values

Numerical variables are handled using median imputation, while categorical variables are handled using their most frequent value.

### 3. Encoding

The project contains categorical variables such as `ProductCD`, `card4`, `card6`, `M6`, and `P_emaildomain`.

Low-cardinality categorical variables are one-hot encoded. `P_emaildomain` is label encoded in the original modelling approach.

### 4. Class Imbalance

Fraudulent transactions represent only about 3.5% of observations. The project therefore uses **SMOTE (Synthetic Minority Over-sampling Technique)** inside the training pipeline.

Importantly, SMOTE is applied only to the training data through the imbalanced-learn pipeline; the test set remains untouched for evaluation.

### 5. Model

The classifier is **XGBoost**, configured with `logloss` as the evaluation metric and a fixed random seed of 42.

## Original Notebook Benchmark

The original submitted notebook produced the following classification report:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Legitimate | 0.98 | 0.99 | 0.99 |
| Fraud | 0.69 | 0.52 | 0.59 |
| **Overall accuracy** | | | **0.97** |

The most important limitation is the **0.52 recall for the fraud class**. The model therefore misses a substantial proportion of fraudulent transactions in the original evaluation.

See [`reports/model_results.md`](reports/model_results.md) for the complete benchmark.

## Repository Structure

```text
credit-card-fraud-detection/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
│
├── reports/
│   └── model_results.md
│
├── src/
│   └── README.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd credit-card-fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the Parquet file here:

```text
data/ieee_fraud_detection.parquet
```

### 5. Run the notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/credit_card_fraud_detection.ipynb
```

## Key Skills Demonstrated

- Python
- Pandas
- NumPy
- Exploratory Data Analysis
- Missing-value analysis
- Feature preprocessing
- Categorical encoding
- Correlation analysis
- Imbalanced-learn / SMOTE
- XGBoost
- Scikit-learn pipelines
- Classification metrics
- Confusion-matrix analysis
- Data visualisation with Matplotlib and Seaborn

## Limitations and Future Improvements

The project is a portfolio-level machine-learning implementation rather than a production fraud-detection system. Potential improvements include:

- Hyperparameter tuning for XGBoost
- Precision-Recall AUC evaluation
- Probability-threshold optimisation
- Cross-validation
- Feature-importance and SHAP analysis
- Comparison with alternative models such as LightGBM, CatBoost and Random Forest
- More rigorous leakage and temporal-validation checks
- Model calibration and monitoring

## Disclaimer

This project is for educational and portfolio purposes. It should not be used directly to make financial decisions or deploy a production fraud-detection system without additional validation, security controls, monitoring, and domain-specific review.
