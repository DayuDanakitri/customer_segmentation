"""Evaluasi kualitas clustering dan Adaptive K Selection (multi-metric ranking)."""
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def calculate_clustering_metrics(X: pd.DataFrame, kmeans_results: Dict[int, dict]) -> pd.DataFrame:
    """Menghitung Silhouette, Davies-Bouldin, dan Calinski-Harabasz untuk tiap K.

    Interpretasi: Silhouette (tinggi = baik), DBI (rendah = baik), CHI (tinggi = baik).
    Istilah yang dipakai adalah "kualitas clustering" / "K yang paling sesuai", bukan
    "akurasi", karena clustering bersifat unsupervised.

    Args:
        X: Fitur yang sudah distandardisasi (sama dengan yang dipakai run_kmeans_for_multiple_k).
        kmeans_results: Output run_kmeans_for_multiple_k.

    Returns:
        DataFrame kolom [K, Silhouette, DBI, CHI], diurutkan menurut K.
    """
    rows: List[dict] = []
    for k, res in kmeans_results.items():
        labels = res["labels"]
        if len(set(labels)) < 2:
            raise ValueError(f"K={k}: hanya menghasilkan 1 cluster, metrik tidak terdefinisi.")
        rows.append({
            "K": k,
            "Silhouette": silhouette_score(X, labels),
            "DBI": davies_bouldin_score(X, labels),
            "CHI": calinski_harabasz_score(X, labels),
        })
    return pd.DataFrame(rows).sort_values("K").reset_index(drop=True)


def rank_k_candidates(
    metrics_df: pd.DataFrame,
    weights: Dict[str, float] = None,
) -> pd.DataFrame:
    """Meranking kandidat K berdasarkan tiga metric (metode ranking sederhana, bukan satu-satunya cara valid).

    Ranking: Silhouette descending (1 = terbaik), DBI ascending (1 = terbaik),
    CHI descending (1 = terbaik). Total Rank = jumlah rank (bisa diberi bobot via `weights`).

    Args:
        metrics_df: Output calculate_clustering_metrics.
        weights: Bobot per metric, mis. {"Silhouette": 1, "DBI": 1, "CHI": 1}. Default semua 1.

    Returns:
        DataFrame dengan kolom tambahan Rank_Silhouette, Rank_DBI, Rank_CHI, TotalRank,
        diurutkan dari TotalRank terkecil (paling disarankan).
    """
    weights = weights or {"Silhouette": 1.0, "DBI": 1.0, "CHI": 1.0}
    df = metrics_df.copy()
    df["Rank_Silhouette"] = df["Silhouette"].rank(ascending=False, method="min")
    df["Rank_DBI"] = df["DBI"].rank(ascending=True, method="min")
    df["Rank_CHI"] = df["CHI"].rank(ascending=False, method="min")
    df["TotalRank"] = (
        df["Rank_Silhouette"] * weights.get("Silhouette", 1.0)
        + df["Rank_DBI"] * weights.get("DBI", 1.0)
        + df["Rank_CHI"] * weights.get("CHI", 1.0)
    )
    return df.sort_values("TotalRank").reset_index(drop=True)


def select_optimal_k(ranked_df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """Memilih K dengan TotalRank terkecil (K "paling sesuai" berdasarkan multi-metric ranking).

    Jika ada lebih dari satu K dengan TotalRank sama, dipilih K terkecil (model lebih sederhana)
    dan hal ini dicatat secara eksplisit.

    Args:
        ranked_df: Output rank_k_candidates.

    Returns:
        (selected_k, ranked_df_terurut). ranked_df_terurut sama seperti input, sudah terurut.
    """
    best_rank = ranked_df["TotalRank"].min()
    tied = ranked_df[ranked_df["TotalRank"] == best_rank]
    if len(tied) > 1:
        print(f"[select_optimal_k] {len(tied)} kandidat K seri pada TotalRank={best_rank}: "
              f"{sorted(tied['K'].tolist())}. Dipilih K terkecil.")
    selected_k = int(tied["K"].min())
    return selected_k, ranked_df
