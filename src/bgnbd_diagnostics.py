
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from lifetimes import BetaGeoFitter, GammaGammaFitter


# ============================================================
# 1. TRANSACTION TABLE
# ============================================================

def make_transaction_table(
    data,
    unit="day",
    customer_col="CustomerID",
    invoice_col="InvoiceNo",
    date_col="InvoiceDate",
    quantity_col="Quantity",
    price_col="UnitPrice",
):
    """
    Membuat tabel transaksi berdasarkan dua definisi:

    - day     : satu transaksi per customer per hari
    - invoice : satu transaksi per invoice

    Monetary dihitung berdasarkan total pendapatan setiap transaksi.
    """

    df = data.copy()

    required = [
        customer_col,
        invoice_col,
        date_col,
        quantity_col,
        price_col,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    df[date_col] = pd.to_datetime(
        df[date_col], errors="coerce"
    )

    df[quantity_col] = pd.to_numeric(
        df[quantity_col], errors="coerce"
    )

    df[price_col] = pd.to_numeric(
        df[price_col], errors="coerce"
    )

    df = df.dropna(
        subset=[
            customer_col,
            invoice_col,
            date_col,
            quantity_col,
            price_col,
        ]
    )

    # Menggunakan transaksi dengan quantity dan price positif.
    df = df[
        (df[quantity_col] > 0) &
        (df[price_col] > 0)
    ].copy()

    df["LineRevenue"] = (
        df[quantity_col] * df[price_col]
    )

    if unit == "invoice":

        tx = (
            df.groupby(
                [customer_col, invoice_col],
                as_index=False
            )
            .agg(
                TransactionDate=(date_col, "min"),
                Monetary=("LineRevenue", "sum"),
            )
        )

    elif unit == "day":

        df["TransactionDate"] = (
            df[date_col].dt.normalize()
        )

        tx = (
            df.groupby(
                [customer_col, "TransactionDate"],
                as_index=False
            )
            .agg(
                Monetary=("LineRevenue", "sum")
            )
        )

    else:
        raise ValueError(
            "unit harus 'day' atau 'invoice'"
        )

    tx = tx.rename(
        columns={customer_col: "CustomerID"}
    )

    tx["TransactionDate"] = pd.to_datetime(
        tx["TransactionDate"]
    )

    tx = tx[
        tx["Monetary"] > 0
    ].copy()

    return tx.sort_values(
        ["CustomerID", "TransactionDate"]
    ).reset_index(drop=True)


# ============================================================
# 2. CUSTOMER SUMMARY
# ============================================================

def build_customer_summary(
    transactions,
    calibration_end
):
    """
    Membuat summary BG/NBD.

    Frequency = jumlah transaksi berulang (total transaksi - 1)
    Recency   = jarak transaksi pertama hingga terakhir
    T         = jarak transaksi pertama hingga akhir calibration

    Semua satuan waktu menggunakan hari.
    """

    calibration_end = pd.Timestamp(
        calibration_end
    )

    rows = []

    for customer_id, group in transactions.groupby(
        "CustomerID"
    ):

        group = group.sort_values(
            "TransactionDate"
        )

        dates = group["TransactionDate"]

        first_date = dates.iloc[0]
        last_date = dates.iloc[-1]

        total_transactions = len(group)

        frequency = max(
            total_transactions - 1,
            0
        )

        recency = (
            last_date - first_date
        ).total_seconds() / 86400

        T = (
            calibration_end - first_date
        ).total_seconds() / 86400

        monetary_value = group[
            "Monetary"
        ].mean()

        rows.append({
            "CustomerID": customer_id,
            "frequency": frequency,
            "recency": recency,
            "T": T,
            "monetary_value": monetary_value,
            "transaction_count": total_transactions,
            "first_purchase": first_date,
            "last_purchase": last_date,
        })

    summary = pd.DataFrame(rows)

    if summary.empty:
        raise ValueError(
            "Customer summary kosong."
        )

    # Validasi data.
    summary = summary[
        (summary["frequency"] >= 0) &
        (summary["recency"] >= 0) &
        (summary["T"] >= 0) &
        (summary["recency"] <= summary["T"]) &
        (summary["monetary_value"] > 0)
    ].copy()

    return summary.reset_index(drop=True)


# ============================================================
# 3. BG/NBD DIAGNOSTICS
# ============================================================

def diagnose_bgnbd(
    model,
    summary
):
    """
    Menghasilkan diagnostik parameter BG/NBD
    dan distribusi Probability Alive.
    """

    params = model.params_.to_dict()

    result = {
        "r": params.get("r", np.nan),
        "alpha": params.get("alpha", np.nan),
        "a": params.get("a", np.nan),
        "b": params.get("b", np.nan),
    }

    alive = model.conditional_probability_alive(
        summary["frequency"],
        summary["recency"],
        summary["T"]
    )

    alive = pd.Series(
        np.asarray(alive),
        index=summary.index,
        name="probability_alive"
    )

    alive = alive.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_alive = alive.dropna()

    result.update({
        "prob_alive_mean": valid_alive.mean(),
        "prob_alive_std": valid_alive.std(),
        "prob_alive_min": valid_alive.min(),
        "prob_alive_median": valid_alive.median(),
        "prob_alive_max": valid_alive.max(),
        "prob_alive_unique": valid_alive.nunique(),
        "prob_alive_all_one": bool(
            np.allclose(
                valid_alive.to_numpy(),
                1.0
            )
        ),
        "near_zero_parameters": [
            name
            for name in ["r", "alpha", "a", "b"]
            if pd.notna(result[name])
            and abs(result[name]) < 1e-6
        ],
    })

    return result, alive


# ============================================================
# 4. FIT BG/NBD
# ============================================================

def fit_bgnbd_experiment(
    summary,
    penalizer
):
    """
    Fit satu model BG/NBD.

    converged menunjukkan bahwa fit selesai tanpa exception.
    Warning tetap disimpan untuk pemeriksaan lebih lanjut.
    """

    model = BetaGeoFitter(
        penalizer_coef=penalizer
    )

    warning_messages = []
    error_message = None
    converged = False

    try:

        with warnings.catch_warnings(
            record=True
        ) as caught:

            warnings.simplefilter("always")

            model.fit(
                summary["frequency"],
                summary["recency"],
                summary["T"]
            )

            warning_messages = [
                str(w.message)
                for w in caught
            ]

        converged = True

    except Exception as exc:

        error_message = repr(exc)

    return {
        "model": model if converged else None,
        "converged": converged,
        "warnings": warning_messages,
        "error": error_message,
    }


# ============================================================
# 5. FIT GAMMA-GAMMA
# ============================================================

def fit_gamma_gamma(
    summary,
    penalizer=0.001
):
    """
    Gamma-Gamma hanya menggunakan pelanggan yang:

    - Memiliki repeat purchase (frequency > 0)
    - Memiliki monetary value positif
    """

    eligible = summary[
        (summary["frequency"] > 0) &
        (summary["monetary_value"] > 0)
    ].copy()

    if eligible.empty:
        raise ValueError(
            "Tidak ada pelanggan yang memenuhi syarat Gamma-Gamma."
        )

    if eligible["frequency"].nunique() < 2:
        raise ValueError(
            "Variasi frequency tidak cukup untuk Gamma-Gamma."
        )

    model = GammaGammaFitter(
        penalizer_coef=penalizer
    )

    model.fit(
        eligible["frequency"],
        eligible["monetary_value"]
    )

    return model, eligible


# ============================================================
# 6. EVALUATE PREDICTIONS
# ============================================================

def evaluate_predictions(
    bgf,
    summary,
    holdout_transactions,
    horizon_days,
    gamma_model=None,
):
    """
    Evaluasi BG/NBD dan Gamma-Gamma pada periode holdout.

    BG/NBD:
    - Prediksi pembelian dilakukan pada pelanggan dengan
      frequency > 0 dan recency > 0.
    - Pelanggan yang tidak memenuhi syarat tetap dipertahankan
      dalam hasil, tetapi prediksinya NaN.

    Gamma-Gamma:
    - Prediksi revenue hanya untuk pelanggan dengan
      frequency > 0 dan monetary value positif.

    Metrik pembelian dan revenue dihitung pada populasi
    yang memenuhi syarat masing-masing.
    """

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error
    )

    from scipy.stats import spearmanr

    if horizon_days <= 0:
        raise ValueError(
            "horizon_days harus lebih besar dari 0."
        )

    result = summary.copy()

    # --------------------------------------------------------
    # Actual purchases and revenue
    # --------------------------------------------------------

    actual = (
        holdout_transactions
        .groupby("CustomerID")
        .agg(
            actual_purchases=(
                "TransactionDate",
                "size"
            ),
            actual_revenue=(
                "Monetary",
                "sum"
            )
        )
    )

    result = result.merge(
        actual,
        left_on="CustomerID",
        right_index=True,
        how="left"
    )

    result[
        ["actual_purchases", "actual_revenue"]
    ] = result[
        ["actual_purchases", "actual_revenue"]
    ].fillna(0)

    # --------------------------------------------------------
    # BG/NBD purchase prediction
    # --------------------------------------------------------

    result["predicted_purchases"] = np.nan

    valid_mask = (
        (result["frequency"] > 0) &
        (result["recency"] > 0) &
        (result["T"] > 0)
    )

    if valid_mask.any():

        predictions = (
            bgf.conditional_expected_number_of_purchases_up_to_time(
                horizon_days,
                result.loc[valid_mask, "frequency"],
                result.loc[valid_mask, "recency"],
                result.loc[valid_mask, "T"]
            )
        )

        predictions = np.asarray(
            predictions,
            dtype=float
        )

        result.loc[
            valid_mask,
            "predicted_purchases"
        ] = predictions

    # Hanya nilai prediksi finite yang dapat dievaluasi.
    purchase_valid_mask = (
        result["predicted_purchases"].notna() &
        np.isfinite(result["predicted_purchases"]) &
        result["actual_purchases"].notna()
    )

    purchase_eval = result[
        purchase_valid_mask
    ].copy()

    if not purchase_eval.empty:

        purchase_mae = mean_absolute_error(
            purchase_eval["actual_purchases"],
            purchase_eval["predicted_purchases"]
        )

        purchase_rmse = np.sqrt(
            mean_squared_error(
                purchase_eval["actual_purchases"],
                purchase_eval["predicted_purchases"]
            )
        )

        purchase_spearman = spearmanr(
            purchase_eval["actual_purchases"],
            purchase_eval["predicted_purchases"]
        ).statistic

    else:

        purchase_mae = np.nan
        purchase_rmse = np.nan
        purchase_spearman = np.nan

    # --------------------------------------------------------
    # Gamma-Gamma revenue prediction
    # --------------------------------------------------------

    result["predicted_revenue"] = np.nan

    if gamma_model is not None:

        revenue_valid_mask = (
            (result["frequency"] > 0) &
            (result["monetary_value"] > 0) &
            result["predicted_purchases"].notna() &
            np.isfinite(result["predicted_purchases"])
        )

        if revenue_valid_mask.any():

            expected_value = (
                gamma_model.conditional_expected_average_profit(
                    result.loc[
                        revenue_valid_mask,
                        "frequency"
                    ],
                    result.loc[
                        revenue_valid_mask,
                        "monetary_value"
                    ]
                )
            )

            expected_value = np.asarray(
                expected_value,
                dtype=float
            )

            predicted_revenue = (
                result.loc[
                    revenue_valid_mask,
                    "predicted_purchases"
                ].to_numpy() *
                expected_value
            )

            result.loc[
                revenue_valid_mask,
                "predicted_revenue"
            ] = predicted_revenue

    # Revenue evaluation hanya untuk pelanggan dengan
    # prediksi revenue yang valid.
    revenue_valid_mask = (
        result["predicted_revenue"].notna() &
        np.isfinite(result["predicted_revenue"]) &
        result["actual_revenue"].notna()
    )

    revenue_eval = result[
        revenue_valid_mask
    ].copy()

    if (
        len(revenue_eval) > 1
        and revenue_eval["actual_revenue"].nunique() > 1
        and revenue_eval["predicted_revenue"].nunique() > 1
    ):

        revenue_spearman = spearmanr(
            revenue_eval["actual_revenue"],
            revenue_eval["predicted_revenue"]
        ).statistic

    else:

        revenue_spearman = np.nan

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {

        # Purchase metrics
        "purchase_evaluated_customers": len(
            purchase_eval
        ),

        "purchase_total_customers": len(result),

        "purchase_excluded_customers": (
            len(result) - len(purchase_eval)
        ),

        "predicted_purchases_total": (
            purchase_eval["predicted_purchases"].sum()
        ),

        "actual_purchases_total": (
            purchase_eval["actual_purchases"].sum()
        ),

        "purchase_MAE": purchase_mae,

        "purchase_RMSE": purchase_rmse,

        "purchase_Spearman": purchase_spearman,

        # Revenue metrics
        "revenue_evaluated_customers": len(
            revenue_eval
        ),

        "revenue_total_customers": len(result),

        "predicted_revenue_total": (
            revenue_eval["predicted_revenue"].sum()
        ),

        "actual_revenue_total": (
            revenue_eval["actual_revenue"].sum()
        ),

        "revenue_Spearman": revenue_spearman,
    }

    return result, metrics
