"""Loading, validasi kolom, dan pembersihan data transaksi Online Retail II."""
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Nama kolom asli -> nama kolom yang dinormalisasi
COLUMN_MAPPING: Dict[str, str] = {
    "Invoice": "InvoiceNo",
    "StockCode": "StockCode",
    "Description": "Description",
    "Quantity": "Quantity",
    "InvoiceDate": "InvoiceDate",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
    "Country": "Country",
}
REQUIRED_COLUMNS: List[str] = list(COLUMN_MAPPING.values())

# StockCode non-produk yang ditemukan pada sheet 2010-2011 (ongkir, biaya, manual, dll.)
NON_PRODUCT_CODES: List[str] = ["POST", "M", "C2", "DOT", "BANK CHARGES", "PADS"]


def load_data(path: Path, sheet_name: str) -> pd.DataFrame:
    """Membaca file Excel Online Retail II.

    Args:
        path: Lokasi file .xlsx.
        sheet_name: Nama sheet yang dibaca.

    Returns:
        DataFrame mentah (belum dibersihkan).

    Raises:
        FileNotFoundError: Jika file tidak ditemukan.
        ValueError: Jika sheet tidak ada di dalam file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}. Letakkan file di folder data/ "
            "atau ubah data_path di src/config.py."
        )
    available = pd.ExcelFile(path, engine="openpyxl").sheet_names
    if sheet_name not in available:
        raise ValueError(f"Sheet {sheet_name!r} tidak ada. Sheet tersedia: {available}")
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Menormalisasi nama kolom dan memastikan semua kolom wajib tersedia.

    Raises:
        ValueError: Jika ada kolom wajib yang tidak ditemukan.
    """
    df = df.rename(columns=COLUMN_MAPPING)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}. Kolom yang ada: {list(df.columns)}"
        )
    return df[REQUIRED_COLUMNS].copy()


def create_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan kolom Revenue = Quantity x UnitPrice."""
    out = df.copy()
    out["Revenue"] = out["Quantity"] * out["UnitPrice"]
    return out


def clean_data(
    df: pd.DataFrame,
    remove_non_product: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Membersihkan data transaksi dengan pencatatan setiap langkah.

    Urutan langkah (jumlah yang dihapus bersifat sekuensial):
        1. InvoiceDate tidak valid
        2. CustomerID kosong
        3. Baris duplikat identik
        4. Cancellation (InvoiceNo diawali "C")
        5. Quantity <= 0
        6. UnitPrice <= 0
        7. (opsional) StockCode non-produk

    Args:
        df: Output validate_columns().
        remove_non_product: Jika True, hapus StockCode non-produk.

    Returns:
        Tuple (df_clean, cleaning_log).
    """
    df = df.copy()
    log: List[dict] = []

    # InvoiceNo dan StockCode bertipe campuran (int/str) pada dataset asli
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    def record(step: str, before: int) -> None:
        log.append({
            "step": step,
            "rows_before": before,
            "rows_after": len(df),
            "rows_removed": before - len(df),
            "customers_after": int(df["CustomerID"].nunique()),
        })

    log.append({
        "step": "initial",
        "rows_before": len(df),
        "rows_after": len(df),
        "rows_removed": 0,
        "customers_after": int(df["CustomerID"].nunique()),
    })

    n = len(df); df = df[df["InvoiceDate"].notna()]; record("remove_invalid_invoice_date", n)
    n = len(df); df = df[df["CustomerID"].notna()]; record("remove_missing_customer_id", n)
    n = len(df); df = df.drop_duplicates(); record("remove_duplicate_rows", n)
    n = len(df); df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]; record("remove_cancellation", n)
    n = len(df); df = df[df["Quantity"] > 0]; record("remove_non_positive_quantity", n)
    n = len(df); df = df[df["UnitPrice"] > 0]; record("remove_non_positive_unit_price", n)

    if remove_non_product:
        n = len(df)
        df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]
        record("remove_non_product_codes", n)

    df["CustomerID"] = df["CustomerID"].astype(int)
    df = create_revenue(df).reset_index(drop=True)
    return df, pd.DataFrame(log)


def print_cleaning_summary(cleaning_log: pd.DataFrame) -> None:
    """Menampilkan ringkasan preprocessing (rows/customers before-after)."""
    first, last = cleaning_log.iloc[0], cleaning_log.iloc[-1]
    print("=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Rows before      : {int(first['rows_before']):,}")
    print(f"Rows after       : {int(last['rows_after']):,}")
    print(f"Customers before : {int(first['customers_after']):,}")
    print(f"Customers after  : {int(last['customers_after']):,}")
    print("-" * 60)
    for _, r in cleaning_log.iloc[1:].iterrows():
        print(f"{r['step']:<35} removed: {int(r['rows_removed']):>10,}")
    print("=" * 60)
