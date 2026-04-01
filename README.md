# Credit Card Customer Segmentation

An end-to-end unsupervised Machine Learning project that segments 8,950 credit card customers into distinct behavioural groups using KMeans clustering — enabling targeted marketing strategies and personalised financial products.

---

## Problem Statement

A bank has 8,950 active credit card holders with 18 behavioural features collected over 6 months. There are no predefined labels. The goal is to discover natural customer segments based on spending habits, payment behaviour, cash advance usage, and credit utilisation — so each group can be targeted differently.

---

## Dataset

**Source:** [Kaggle — Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
**Size:** 8,950 customers | 18 features | 6-month behavioural snapshot

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
| PRC_FULL_PAYMENT | % of months where full payment was made |
| TENURE | Duration of credit card service (months) |

---

## Project Structure

```
Credit-Card-Customer-Segmentation/
│
├── data/
│   ├── raw/                          # Original dataset (gitignored)
│   └── processed/                    # Scaled & cleaned data (gitignored)
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Distributions, correlations, missing values
│   ├── 02_Feature_Engineering.ipynb  # Imputation, 4 new ratio features, StandardScaler
│   └── 03_Model_Clustering.ipynb     # Elbow + Silhouette, KMeans K=4, PCA, profiles
│
├── src/
│   ├── config.py                     # Paths, constants, cluster params (K=4)
│   ├── data_loader.py                # Load & validate raw data
│   ├── preprocessing.py              # Impute, engineer features, scale
│   ├── model.py                      # KMeans, optimal K search, PCA
│   └── evaluate.py                   # Cluster visualisation plots
│
├── models/                           # kmeans.pkl, pca.pkl, scaler.pkl
├── reports/
│   ├── cluster_summary.csv           # Mean feature values per cluster
│   └── figures/                      # Auto-generated plots
│
├── scripts/
│   ├── download_data.py              # Kaggle API download
│   └── github_push.py                # Git commit & push helper
│
├── main.py                           # Full pipeline runner
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Workflow

```
Raw Data  ->  01_EDA.ipynb  ->  02_Feature_Engineering.ipynb  ->  03_Model_Clustering.ipynb  ->  main.py
```

Each notebook is self-contained and tells the story of one phase. `main.py` runs the full pipeline end-to-end in one command.

---

## Setup & Usage

```bash
git clone https://github.com/bhavesh2418/Credit-Card-Customer-Segmentation.git
cd Credit-Card-Customer-Segmentation
pip install -r requirements.txt
cp .env.example .env        # fill in Kaggle + GitHub credentials
python scripts/download_data.py
python main.py
```

---

## EDA Key Findings

- **Missing values:** `MINIMUM_PAYMENTS` (3.5%) and `CREDIT_LIMIT` (0.01%) — imputed with column median
- **Right-skewed monetary features:** Most customers have low spend; a small group drives the upper tail
- **Two distinct purchase types:** One-off vs installment buyers are clearly different customer types
- **Cash advance users:** ~50% of customers never use cash advances — strong segmentation signal
- **Full payment behaviour:** Most customers never pay in full — majority are revolving credit users
- **Tenure:** Dataset dominated by 12-month tenured customers

---

## Feature Engineering

| Feature | Formula | Business Meaning |
|---|---|---|
| `PURCHASES_TO_LIMIT_RATIO` | PURCHASES / CREDIT_LIMIT | Credit utilisation for purchases |
| `CASH_ADVANCE_RATIO` | CASH_ADVANCE / BALANCE | Reliance on cash vs balance |
| `PAYMENT_TO_MINIMUM_RATIO` | PAYMENTS / MINIMUM_PAYMENTS | Aggressiveness of debt repayment |
| `MONTHLY_AVG_PURCHASE` | PURCHASES / TENURE | Average monthly spending rate |

---

## Clustering Results

**Algorithm:** KMeans | **K:** 4 | **Silhouette Score:** 0.1944 | **Customers:** 8,950

| Cluster | Size | Label | Key Traits |
|---|---|---|---|
| 0 | 3,390 (37.9%) | **Low Activity** | Low balance, low purchases, low engagement |
| 1 | 1,232 (13.8%) | **Transactors** | High purchases, pay frequently, high credit limit |
| 2 | 267 (3.0%) | **Cash Advance Heavy** | Very high cash advance, high balance, high risk |
| 3 | 4,061 (45.4%) | **Revolvers** | Moderate balance, installment purchases, carry debt |

**K selection:** Elbow method showed diminishing returns after K=4; Silhouette score confirmed K=2 or K=4 as viable — K=4 chosen for richer business interpretability.

**PCA:** 2 principal components explain **44.4%** of total variance.

---

## Key Insights

1. **Cash Advance Heavy** (3%) is a small but high-risk segment — high balance, low payments, heavy cash usage
2. **Revolvers** (45%) represent the bank's largest segment and primary revenue driver through interest
3. **Transactors** (14%) are low-risk, high-spend customers — ideal candidates for premium rewards products
4. **Low Activity** (38%) customers represent a re-engagement opportunity
5. All monetary features are heavily right-skewed — StandardScaler was essential before clustering

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | KMeans, PCA, StandardScaler, Silhouette |
| Matplotlib, Seaborn | Visualisation |
| Joblib | Model persistence |
| Jupyter Notebook | Interactive analysis |
| Kaggle API | Dataset download |

---

## Author

**Bhavesh** — Portfolio project demonstrating unsupervised ML, customer segmentation, and a production-style end-to-end workflow.
