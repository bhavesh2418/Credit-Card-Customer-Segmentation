"""
evaluate.py — Visualise clusters, profiles, and segment characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)


def plot_clusters_2d(df_pca: pd.DataFrame, labels: np.ndarray, explained: float):
    """Scatter plot of clusters in PCA 2D space."""
    palette = ['steelblue', 'tomato', 'seagreen', 'orange', 'purple', 'brown']
    plt.figure(figsize=(9, 6))
    for cluster in np.unique(labels):
        mask = labels == cluster
        plt.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'],
                    label=f'Cluster {cluster}', alpha=0.5, s=15,
                    color=palette[cluster % len(palette)])
    plt.title(f'Customer Segments — PCA 2D (explains {explained:.1f}% variance)',
              fontweight='bold')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_clusters_pca.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/figures/02_clusters_pca.png")


def plot_cluster_profiles(df_unscaled: pd.DataFrame, labels: np.ndarray):
    """Radar/bar chart of mean feature values per cluster."""
    df = df_unscaled.copy()
    df['Cluster'] = labels

    key_features = [
        'BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
        'PAYMENTS', 'PRC_FULL_PAYMENT', 'PURCHASES_TO_LIMIT_RATIO',
        'CASH_ADVANCE_RATIO', 'MONTHLY_AVG_PURCHASE'
    ]
    key_features = [f for f in key_features if f in df.columns]

    profile = df.groupby('Cluster')[key_features].mean()

    # Normalize for comparison
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap
    sns.heatmap(profile_norm.T, annot=True, fmt='.2f', cmap='YlOrRd',
                linewidths=0.5, ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title('Cluster Feature Profiles (Normalized)', fontweight='bold')
    axes[0].set_xlabel('Cluster')

    # Bar chart — raw means for top features
    profile[['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'PAYMENTS']].plot(
        kind='bar', ax=axes[1], edgecolor='white', alpha=0.85
    )
    axes[1].set_title('Key Metrics by Cluster (Raw Mean)', fontweight='bold')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Mean Value ($)')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(loc='upper right', fontsize=8)

    plt.suptitle('Customer Segment Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_cluster_profiles.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/figures/03_cluster_profiles.png")


def plot_cluster_distribution(labels: np.ndarray):
    """Bar chart of cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    pct = counts / counts.sum() * 100

    plt.figure(figsize=(7, 4))
    bars = plt.bar([f'Cluster {k}' for k in unique], counts,
                   color=['steelblue', 'tomato', 'seagreen', 'orange'][:len(unique)],
                   edgecolor='white', alpha=0.85)
    for bar, p in zip(bars, pct):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{p:.1f}%', ha='center', fontsize=10, fontweight='bold')
    plt.title('Cluster Size Distribution', fontweight='bold')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_cluster_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/figures/04_cluster_distribution.png")
