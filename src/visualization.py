"""Visualisasi evaluasi clustering: K vs metric, dan PCA 2D."""
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def plot_silhouette_by_k(metrics_df: pd.DataFrame, selected_k: Optional[int] = None, ax=None):
    """Plot Silhouette Score vs K (semakin tinggi semakin baik)."""
    ax = ax or plt.gca()
    ax.plot(metrics_df["K"], metrics_df["Silhouette"], marker="o")
    if selected_k is not None:
        ax.axvline(selected_k, color="red", linestyle="--", alpha=0.6, label=f"K terpilih = {selected_k}")
        ax.legend()
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score"); ax.set_title("Silhouette vs K (tinggi = baik)")
    return ax


def plot_dbi_by_k(metrics_df: pd.DataFrame, selected_k: Optional[int] = None, ax=None):
    """Plot Davies-Bouldin Index vs K (semakin rendah semakin baik)."""
    ax = ax or plt.gca()
    ax.plot(metrics_df["K"], metrics_df["DBI"], marker="o", color="orange")
    if selected_k is not None:
        ax.axvline(selected_k, color="red", linestyle="--", alpha=0.6, label=f"K terpilih = {selected_k}")
        ax.legend()
    ax.set_xlabel("K"); ax.set_ylabel("Davies-Bouldin Index"); ax.set_title("DBI vs K (rendah = baik)")
    return ax


def plot_chi_by_k(metrics_df: pd.DataFrame, selected_k: Optional[int] = None, ax=None):
    """Plot Calinski-Harabasz Index vs K (semakin tinggi semakin baik)."""
    ax = ax or plt.gca()
    ax.plot(metrics_df["K"], metrics_df["CHI"], marker="o", color="green")
    if selected_k is not None:
        ax.axvline(selected_k, color="red", linestyle="--", alpha=0.6, label=f"K terpilih = {selected_k}")
        ax.legend()
    ax.set_xlabel("K"); ax.set_ylabel("Calinski-Harabasz Index"); ax.set_title("CHI vs K (tinggi = baik)")
    return ax


def plot_clusters_pca(X: pd.DataFrame, labels, random_state: int = 42, ax=None):
    """Visualisasi cluster pada ruang PCA 2D (hanya untuk visualisasi, bukan fitur clustering).

    Args:
        X: Fitur yang sudah distandardisasi (dipakai saat K-Means final).
        labels: Label cluster hasil K-Means final.
        random_state: Seed PCA agar reproducible.

    Returns:
        (ax, explained_variance_ratio) agar persentase varians yang dijelaskan dapat dilaporkan.
    """
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X)
    ax = ax or plt.gca()
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=15)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Visualisasi Cluster (PCA 2D)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend1)
    return ax, pca.explained_variance_ratio_
