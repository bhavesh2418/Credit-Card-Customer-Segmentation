# Credit Card Customer Segmentation

An end-to-end unsupervised Machine Learning project that segments credit card customers into distinct behavioural groups using clustering — enabling targeted marketing and personalised financial products.

---

## Problem Statement

A bank has 9,000 active credit card holders with 18 behavioural features collected over 6 months. The goal is to identify natural customer segments based on spending habits, payment behaviour, and credit usage — without any predefined labels.

---

## Dataset

**Source:** [Kaggle — Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
**Size:** ~9,000 customers | 18 features | 6-month snapshot

| Feature | Description |
|---|---|
| BALANCE | Amount left for purchases |
| PURCHASES | Total purchase amount |
| ONEOFF_PURCHASES | Maximum single transaction |
| INSTALLMENTS_PURCHASES | Total installment purchases |
| CASH_ADVANCE | Total cash advances taken |
| PURCHASES_FREQUENCY | How often purchases are made (0-1) |
| CASH_ADVANCE_FREQUENCY | How often cash advances are taken (0-1) |
| CREDIT_LIMIT | Customer credit limit |
| PAYMENTS | Total payments made |
| MINIMUM_PAYMENTS | Minimum payments made |
| PRC_FULL_PAYMENT | % of full payments made |
| TENURE | Duration of credit card service (months) |

---

## Project Structure

```
Credit-Card-Customer-Segmentation/
│
├── data/
│   ├── raw/                         # Original dataset (gitignored)
│   └── processed/                   # Scaled & cleaned data (gitignored)
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb # Imputation, scaling, new features
│   └── 03_Model_Clustering.ipynb    # KMeans, Elbow, Silhouette, PCA
│
├── src/
│   ├── config.py                    # Paths, constants, cluster params
│   ├── data_loader.py               # Load & validate raw data
│   ├── preprocessing.py             # Clean, impute, engineer, scale
│   ├── model.py                     # KMeans, optimal K, PCA
│   └── evaluate.py                  # Cluster visualisation plots
│
├── models/                          # Saved models: kmeans.pkl, pca.pkl, scaler.pkl
├── reports/
│   ├── cluster_summary.csv          # Mean feature values per cluster
│   └── figures/                     # Auto-generated plots
│
├── scripts/
│   ├── download_data.py             # Kaggle API download
│   └── github_push.py               # Git commit & push helper
│
├── main.py                          # Full pipeline runner
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Workflow

```
Raw Data -> 01_EDA.ipynb -> 02_Feature_Engineering.ipynb -> 03_Model_Clustering.ipynb -> main.py
```

---

## Setup & Usage

```bash
git clone https://github.com/bhavesh2418/Credit-Card-Customer-Segmentation.git
cd Credit-Card-Customer-Segmentation
pip install -r requirements.txt
cp .env.example .env   # fill in Kaggle + GitHub credentials
python scripts/download_data.py
python main.py
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | KMeans, PCA, Silhouette |
| Matplotlib, Seaborn | Visualisation |
| Joblib | Model persistence |
| Jupyter Notebook | Interactive analysis |
| Kaggle API | Dataset download |

---

## Author

**Bhavesh** — Portfolio project demonstrating unsupervised ML and customer segmentation.
