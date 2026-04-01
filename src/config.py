"""
config.py — Central configuration: paths, constants, model parameters.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"
FIGURES_DIR    = ROOT / "reports" / "figures"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
RAW_DATA_FILE   = DATA_RAW / "CC GENERAL.csv"
CLEAN_DATA_FILE = DATA_PROCESSED / "cc_clean.csv"

# ── Features ──────────────────────────────────────────────────────────────────
ID_COLUMN = "CUST_ID"

FEATURE_COLUMNS = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFFPURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE",
]

# ── Clustering Parameters ─────────────────────────────────────────────────────
RANDOM_STATE  = 42
N_CLUSTERS    = 4          # determined by Elbow + Silhouette analysis
PCA_COMPONENTS = 2         # for 2D visualisation
K_RANGE       = range(2, 11)  # range to evaluate for optimal K
