"""
main.py — Full pipeline: Load -> Preprocess -> Find K -> Cluster -> Evaluate
Run: python main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from src.data_loader   import load_raw_data
from src.preprocessing import preprocess
from src.model         import find_optimal_k, train_kmeans, pca_transform
from src.evaluate      import plot_clusters_2d, plot_cluster_profiles, plot_cluster_distribution
from src.config        import N_CLUSTERS, REPORTS_DIR


def main():
    print("=" * 57)
    print("  Credit Card Customer Segmentation Pipeline")
    print("=" * 57)

    # Step 1: Load
    print("\n[1/5] Loading data...")
    df_raw = load_raw_data()

    # Step 2: Preprocess
    print("\n[2/5] Preprocessing & feature engineering...")
    df_scaled, df_unscaled = preprocess(df_raw, save=True)

    # Step 3: Find optimal K
    print("\n[3/5] Finding optimal number of clusters (Elbow + Silhouette)...")
    k_results = find_optimal_k(df_scaled)

    # Step 4: Train KMeans
    print(f"\n[4/5] Training KMeans with K={N_CLUSTERS}...")
    labels = train_kmeans(df_scaled, n_clusters=N_CLUSTERS)

    # Add cluster labels to unscaled for profiling
    df_unscaled["Cluster"] = labels

    # Step 5: Evaluate
    print("\n[5/5] Generating cluster visualisations...")
    df_pca, explained = pca_transform(df_scaled)
    plot_clusters_2d(df_pca, labels, explained)
    plot_cluster_profiles(df_unscaled, labels)
    plot_cluster_distribution(labels)

    # Save cluster summary
    summary = df_unscaled.groupby("Cluster").mean().round(2)
    summary["Customer_Count"] = df_unscaled["Cluster"].value_counts().sort_index()
    summary.to_csv(REPORTS_DIR / "cluster_summary.csv")

    print("\nPipeline complete.")
    print("Plots    -> reports/figures/")
    print("Summary  -> reports/cluster_summary.csv")


if __name__ == "__main__":
    main()
