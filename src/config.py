"""Konfigurasi terpusat untuk pipeline customer segmentation."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    """Seluruh parameter yang dapat diubah tanpa menyentuh logika kode."""

    # --- Data ---
    data_path: Path = PROJECT_ROOT / "data" / "online_retail_II.xlsx"
    sheet_name: str = "Year 2010-2011"
    output_dir: Path = PROJECT_ROOT / "outputs"

    # --- Preprocessing ---
    # True = hapus StockCode non-produk (POST, M, C2, DOT, BANK CHARGES, PADS).
    # Default False agar sesuai brief; ubah untuk analisis sensitivitas.
    remove_non_product: bool = False

    # --- Reproducibility ---
    random_state: int = 42

    # --- Adaptive K selection ---
    # Bobot (Silhouette, DBI, CHI) pada total rank. (1, 1, 1) = ranking sama rata.
    metric_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Kandidat K dengan cluster terkecil < fraksi ini dikeluarkan dari pemilihan
    # (0.0 = tidak ada filter; mis. 0.01 = cluster terkecil minimal 1% customer).
    min_cluster_fraction: float = 0.0
    # Seed untuk uji stabilitas pemilihan K.
    stability_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    # --- K-Means ---
    candidate_k: Tuple[int, ...] = tuple(range(2, 11))
    n_init: int = 20
    max_iter: int = 300

    # --- Predictive CLV (dipakai pada Step 4) ---
    holdout_days: int = 90            # panjang holdout period (hari)
    clv_horizon_months: int = 3       # prediction horizon CLV
    discount_rate: float = 0.01       # discount rate bulanan
    bgnbd_penalizer: float = 0.05     # regularisasi BG/NBD (0.001-0.01 gagal konvergen pada data ini)
    gg_penalizer: float = 0.0         # regularisasi Gamma-Gamma (0.01 memberi q<1 pada data ini)

    # --- Standardisasi (Step 6) ---
    # log1p pada fitur yang sangat skewed sebelum StandardScaler.
    # Set log_transform=False untuk StandardScaler langsung (sesuai brief murni).
    log_transform: bool = True
    log_features: Tuple[str, ...] = ("Frequency", "Monetary", "PredictiveCLV")


CONFIG = Config()
