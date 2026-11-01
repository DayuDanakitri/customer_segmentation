"""Agregasi transaksi menjadi fitur RFM level pelanggan."""
from typing import Optional

import pandas as pd


def calculate_customer_rfm(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Menghitung Recency, Frequency, Monetary per CustomerID.

    Definisi:
        Recency   = (reference_date - LastPurchaseDate) dalam hari penuh.
        Frequency = jumlah InvoiceNo unik.
        Monetary  = sum(Revenue), Revenue = Quantity x UnitPrice.

    Args:
        df: Data transaksi bersih (kolom CustomerID, InvoiceNo, InvoiceDate, Revenue).
        reference_date: Tanggal acuan. Default: max(InvoiceDate) + 1 hari.

    Returns:
        DataFrame [CustomerID, LastPurchaseDate, Recency, Frequency, Monetary].

    Raises:
        ValueError: Jika kolom wajib tidak ada atau data kosong.
    """
    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "Revenue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom berikut diperlukan untuk RFM: {sorted(missing)}")
    if df.empty:
        raise ValueError("DataFrame kosong, RFM tidak dapat dihitung.")

    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            LastPurchaseDate=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )
    rfm["Recency"] = (reference_date - rfm["LastPurchaseDate"]).dt.days
    return rfm[["CustomerID", "LastPurchaseDate", "Recency", "Frequency", "Monetary"]]
