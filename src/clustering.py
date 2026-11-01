"""Standardisasi fitur dan K-Means dengan beberapa kandidat K."""
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def standardize_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    log_transform: bool = False,
    log_features: Sequence[str] = (),
) -> Tuple[pd.DataFrame, StandardScaler, pd.DataFrame]:
    """Menstandardisasi fitur dengan StandardScaler (opsional log1p sebelumnya).

    Args:
        df: DataFrame customer-level yang memuat feature_cols.
        feature_cols: Fitur untuk clustering, mis. [Recency, Frequency, Monetary, PredictiveCLV].
        log_transform: Jika True, terapkan log1p pada kolom di log_features sebelum scaling.
        log_features: Kolom yang di-log (harus bagian dari feature_cols; nilai harus >= 0).

    Returns:
        (X_scaled_df, scaler, X_pre_scaling_df). Index sama dengan df; scaler bisa disimpan.

    Raises:
        ValueError: Jika kolom hilang, ada NaN/inf, atau ada nilai negatif pada fitur yang di-log.
    """
    feature_cols = list(feature_cols)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak ditemukan: {missing}")
    X = df[feature_cols].astype(float).copy()
    if not np.isfinite(X.values).all():
        raise ValueError("Ada NaN/inf pada fitur clustering.")

    if log_transform:
        bad = [c for c in log_features if c not in feature_cols]
        if bad:
            raise ValueError(f"log_features harus bagian dari feature_cols: {bad}")
        for c in log_features:
            if (X[c] < 0).any():
                raise ValueError(f"Nilai negatif pada {c}; log1p tidak dapat diterapkan.")
            X[c] = np.log1p(X[c])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    return X_scaled, scaler, X


@dataclass
class KMeansResult:
    """Hasil satu run K-Means."""

    k: int
    model: KMeans
    labels: np.ndarray
    inertia: float


def run_kmeans(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
) -> KMeansResult:
    """Menjalankan K-Means untuk satu nilai K.

    Args:
        X: Matriks fitur ter-standardisasi (n_samples, n_features).
        k: Jumlah cluster.
        random_state: Seed agar reproducible.
        n_init: Jumlah inisialisasi centroid; hasil terbaik (inertia terendah) dipakai.
        max_iter: Iterasi maksimum per inisialisasi.

    Raises:
        ValueError: Jika k < 2 atau k >= jumlah sampel.
    """
    X = np.asarray(X, dtype=float)
    if k < 2 or k >= len(X):
        raise ValueError(f"K={k} tidak valid untuk {len(X)} sampel.")
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter)
    labels = model.fit_predict(X)
    return KMeansResult(k=k, model=model, labels=labels, inertia=float(model.inertia_))


def run_kmeans_for_multiple_k(
    X: np.ndarray,
    candidate_k: Sequence[int],
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
    verbose: bool = True,
) -> Dict[int, KMeansResult]:
    """Menjalankan K-Means untuk setiap kandidat K dan menyimpan label tiap K.

    Returns:
        Dict {k: KMeansResult}.
    """
    results: Dict[int, KMeansResult] = {}
    for k in candidate_k:
        results[k] = run_kmeans(X, k, random_state, n_init, max_iter)
        if verbose:
            sizes = np.bincount(results[k].labels)
            print(f"K={k:>2} | inertia={results[k].inertia:>10,.1f} | ukuran cluster min/max = {sizes.min():,}/{sizes.max():,}")
    return results


def run_kmeans(
    X: pd.DataFrame,
    k: int,
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
) -> "KMeans":
    """Menjalankan K-Means untuk satu nilai K.

    Args:
        X: Fitur yang sudah distandardisasi.
        k: Jumlah cluster.
        random_state: Seed agar hasil reproducible.
        n_init: Jumlah inisialisasi centroid berbeda yang dicoba (ambil terbaik).
        max_iter: Iterasi maksimum per inisialisasi.

    Returns:
        Model KMeans yang sudah di-fit.
    """
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter)
    model.fit(X)
    return model


def run_kmeans_for_multiple_k(
    X: pd.DataFrame,
    candidate_k: Sequence[int],
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
) -> dict:
    """Menjalankan K-Means untuk setiap kandidat K.

    Args:
        X: Fitur yang sudah distandardisasi.
        candidate_k: Daftar nilai K yang dicoba, mis. range(2, 11).
        random_state, n_init, max_iter: Parameter K-Means (lihat run_kmeans).

    Returns:
        Dict {k: {"model": KMeans, "labels": np.ndarray}}.

    Raises:
        ValueError: Jika ada k < 2 atau k >= jumlah sampel.
    """
    n = len(X)
    bad = [k for k in candidate_k if k < 2 or k >= n]
    if bad:
        raise ValueError(f"K tidak valid untuk {n} sampel: {bad}")
    results = {}
    for k in candidate_k:
        model = run_kmeans(X, k, random_state=random_state, n_init=n_init, max_iter=max_iter)
        results[k] = {"model": model, "labels": model.labels_}
        print(f"  K={k:<2d} selesai | inertia={model.inertia_:,.1f}")
    return results
