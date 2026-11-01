from typing import Dict, List

import pandas as pd


def create_cluster_profile(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    feature_cols: List[str] = None,
) -> pd.DataFrame:
    """Menghitung statistik deskriptif tiap cluster.

    Args:
        df: DataFrame customer-level dengan kolom cluster_col dan feature_cols.
        cluster_col: Nama kolom label cluster.
        feature_cols: Fitur yang diringkas. Default: Recency, Frequency, Monetary, PredictiveCLV.

    Returns:
        DataFrame satu baris per cluster: jumlah customer, persentase, rata-rata tiap fitur,
        dan total Monetary + PredictiveCLV.
    """
    feature_cols = feature_cols or ["Recency", "Frequency", "Monetary", "PredictiveCLV"]
    missing = [c for c in [cluster_col, *feature_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    n_total = len(df)
    agg = {f"Avg{c}": (c, "mean") for c in feature_cols}
    profile = df.groupby(cluster_col).agg(
        NumCustomers=(cluster_col, "size"),
        **agg,
        TotalMonetary=("Monetary", "sum"),
        TotalPredictiveCLV=("PredictiveCLV", "sum"),
    ).reset_index()
    profile["PctCustomers"] = profile["NumCustomers"] / n_total * 100
    return profile.sort_values(cluster_col).reset_index(drop=True)


def assign_customer_profile(
    profile_df: pd.DataFrame,
    recency_col: str = "AvgRecency",
    frequency_col: str = "AvgFrequency",
    monetary_col: str = "AvgMonetary",
    clv_col: str = "AvgPredictiveCLV",
) -> pd.DataFrame:
    """Memberi label segmen pada tiap cluster berdasarkan RANKING antar cluster (bukan
    threshold median per fitur).

    Kenapa ranking, bukan median: dengan K kecil (mis. K=3), median dari K nilai
    cluster-level SAMA DENGAN salah satu nilai cluster itu sendiri (cluster "tengah"),
    sehingga cluster tengah otomatis "lolos >= median" di semua fitur dan bisa
    tersamakan skornya dengan cluster terbaik. Pendekatan ranking tidak punya masalah ini.

    Metode (didokumentasikan secara eksplisit, dapat diubah):
        1. Tiap cluster diranking per fitur: Recency ascending (1=paling baru=terbaik),
           Frequency/Monetary/PredictiveCLV descending (1=tertinggi=terbaik).
        2. CompositeRank = jumlah keempat rank (lebih kecil = lebih bernilai). Metode
           ini konsisten dengan pendekatan multi-metric ranking yang dipakai untuk
           Adaptive K Selection (Step 9).
        3. Cluster dengan CompositeRank terendah -> "High Value / Loyal".
           Cluster dengan CompositeRank tertinggi -> "At Risk" jika rank Recency-nya
           juga terburuk (customer lama tidak membeli), atau "Low Value" jika tidak.
           Cluster lain (K > 3) -> "Potential".
        4. Untuk K=2: hanya ada "High Value / Loyal" dan satu kelompok terburuk
           (diberi "At Risk" atau "Low Value" sesuai rule poin 3).

    Args:
        profile_df: Output create_cluster_profile.
        recency_col, frequency_col, monetary_col, clv_col: Nama kolom rata-rata.

    Returns:
        profile_df dengan kolom tambahan Rank_Recency, Rank_Frequency, Rank_Monetary,
        Rank_PredictiveCLV, CompositeRank, ValueOrder (1=cluster paling bernilai), ProfileLabel.
    """
    df = profile_df.copy()
    n = len(df)

    df["Rank_Recency"] = df[recency_col].rank(method="min", ascending=True)
    df["Rank_Frequency"] = df[frequency_col].rank(method="min", ascending=False)
    df["Rank_Monetary"] = df[monetary_col].rank(method="min", ascending=False)
    df["Rank_PredictiveCLV"] = df[clv_col].rank(method="min", ascending=False)
    df["CompositeRank"] = (
        df["Rank_Recency"] + df["Rank_Frequency"] + df["Rank_Monetary"] + df["Rank_PredictiveCLV"]
    )
    df["ValueOrder"] = df["CompositeRank"].rank(method="first", ascending=True).astype(int)

    def label_row(r: pd.Series) -> str:
        if r["ValueOrder"] == 1:
            return "High Value / Loyal"
        if r["ValueOrder"] == n:
            return "At Risk" if r["Rank_Recency"] == n else "Low Value"
        return "Potential"

    df["ProfileLabel"] = df.apply(label_row, axis=1)
    return df.sort_values("ValueOrder").reset_index(drop=True)


MARKETING_RECOMMENDATIONS: Dict[str, List[str]] = {
    "High Value / Loyal": [
        "Program retensi dan loyalitas (poin, tier membership)",
        "Penawaran personalisasi berbasis riwayat pembelian",
        "Early access produk baru / preview eksklusif",
    ],
    "Potential": [
        "Cross-selling dan upselling produk terkait",
        "Kampanye engagement untuk meningkatkan frekuensi pembelian",
        "Bundling atau diskon bertingkat untuk mendorong transaksi berikutnya",
    ],
    "At Risk": [
        "Kampanye reaktivasi (email/SMS 'kami merindukanmu')",
        "Diskon personalisasi dengan batas waktu",
        "Survei singkat untuk memahami alasan penurunan aktivitas",
    ],
    "Low Value": [
        "Kampanye otomatis berbiaya rendah (newsletter, promo umum)",
        "Uji insentif kecil untuk transaksi pertama/kedua",
        "Tidak diprioritaskan untuk kampanye personalisasi berbiaya tinggi",
    ],
}


def generate_marketing_recommendation(labeled_profile_df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan rekomendasi pemasaran berdasarkan ProfileLabel tiap cluster.

    Rekomendasi diambil dari MARKETING_RECOMMENDATIONS (tetap, berbasis karakteristik
    segmen), bukan dibuat acak per cluster.

    Args:
        labeled_profile_df: Output assign_customer_profile.

    Returns:
        DataFrame dengan kolom tambahan MarketingRecommendation (list of str).
    """
    df = labeled_profile_df.copy()
    df["MarketingRecommendation"] = df["ProfileLabel"].map(
        lambda lbl: MARKETING_RECOMMENDATIONS.get(lbl, ["Belum ada rule untuk label ini"])
    )
    return df
