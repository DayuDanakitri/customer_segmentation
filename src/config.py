
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
    # True = hapus StockCode non-produk.
    remove_non_product: bool = False

    # --- Reproducibility ---
    random_state: int = 42

    # --- Adaptive K selection ---
    metric_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    min_cluster_fraction: float = 0.0
    stability_seeds: Tuple[int, ...] = (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    )

    # --- K-Means ---
    candidate_k: Tuple[int, ...] = tuple(range(2, 11))
    n_init: int = 20
    max_iter: int = 300

    # --- Predictive CLV ---
    # Definisi transaksi untuk BG/NBD dan Gamma-Gamma.
    clv_transaction_unit: str = "Invoice"

    # Panjang periode holdout.
    holdout_days: int = 90

    # Horizon prediksi CLV.
    clv_horizon_months: int = 3

    # Discount rate per bulan.
    discount_rate: float = 0.01

    # Penalizer BG/NBD.
    bgnbd_penalizer: float = 0.05

    # Penalizer Gamma-Gamma.
    gg_penalizer: float = 0.0

    # --- Standardisasi (Step 6) ---
    log_transform: bool = True
    log_features: Tuple[str, ...] = (
        "Frequency",
        "Monetary",
        "PredictiveCLV",
    )


CONFIG = Config()
