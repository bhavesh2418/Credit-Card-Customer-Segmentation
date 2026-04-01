"""
model.py — KMeans clustering: find optimal K, fit, save, assign labels.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from src.config import RANDOM_STATE, K_RANGE, N_CLUSTERS, MODELS_DIR, FIGURES_DIR


def find_optimal_k(df_scaled: pd.DataFrame) -> dict:
    """Run Elbow + Silhouette analysis to find optimal number of clusters."""
    inertias, silhouettes = [], []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(df_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(df_scaled, labels))
        print(f"  K={k} | Inertia={km.inertia_:,.0f} | Silhouette={silhouette_score(df_scaled, labels):.4f}")

    # Plot Elbow + Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(K_RANGE), inertias, marker='o', color='steelblue', linewidth=2)
    axes[0].set_title('Elbow Method — Optimal K', fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia (WCSS)')
    axes[0].axvline(N_CLUSTERS, color='tomato', linestyle='--', label=f'K={N_CLUSTERS}')
    axes[0].legend()

    axes[1].plot(list(K_RANGE), silhouettes, marker='s', color='seagreen', linewidth=2)
    axes[1].set_title('Silhouette Score — Optimal K', fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].axvline(N_CLUSTERS, color='tomato', linestyle='--', label=f'K={N_CLUSTERS}')
    axes[1].legend()

    plt.suptitle('Optimal K Selection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_optimal_k.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: reports/figures/01_optimal_k.png")

    best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    print(f"\nBest K by Silhouette: {best_k}")
    return {"inertias": inertias, "silhouettes": silhouettes, "best_k": best_k}


def train_kmeans(df_scaled: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> np.ndarray:
    """Fit KMeans with chosen K, save model, return labels."""
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(df_scaled)
    score  = silhouette_score(df_scaled, labels)

    joblib.dump(km, MODELS_DIR / "kmeans.pkl")
    print(f"KMeans (K={n_clusters}) | Silhouette Score = {score:.4f}")
    print(f"Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return labels


def pca_transform(df_scaled: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Reduce to 2D for visualisation."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    components = pca.fit_transform(df_scaled)
    joblib.dump(pca, MODELS_DIR / "pca.pkl")
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA: {n_components} components explain {explained:.1f}% of variance")
    df_pca = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
    return df_pca, explained
