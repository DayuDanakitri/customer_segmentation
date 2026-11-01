import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import ConvergenceError
from scipy.stats import spearmanr


# ============================================================
# 1. CALIBRATION - HOLDOUT
# ============================================================

def create_calibration_holdout(
    df: pd.DataFrame,
    holdout_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Membagi data transaksi menjadi calibration dan holdout berdasarkan waktu."""

    required = {"InvoiceDate", "CustomerID", "InvoiceNo", "Revenue"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    if df.empty:
        raise ValueError("Data transaksi kosong.")

    if holdout_days <= 0:
        raise ValueError("holdout_days harus lebih besar dari 0.")

    last_day = df["InvoiceDate"].max().normalize()
    first_day = df["InvoiceDate"].min().normalize()

    if holdout_days >= (last_day - first_day).days:
        raise ValueError(
            f"holdout_days={holdout_days} tidak valid untuk rentang data."
        )

    cutoff = last_day - pd.Timedelta(days=holdout_days)
    boundary = cutoff + pd.Timedelta(days=1)

    calibration = df[df["InvoiceDate"] < boundary].copy()
    holdout = df[df["InvoiceDate"] >= boundary].copy()

    if calibration.empty or holdout.empty:
        raise ValueError("Calibration atau holdout kosong.")

    return calibration, holdout, cutoff


# ============================================================
# 2. PEMBENTUKAN RINGKASAN TRANSAKSI BERDASARKAN INVOICE
# ============================================================

def build_invoice_table(df: pd.DataFrame) -> pd.DataFrame:
    """Menggabungkan baris produk menjadi satu baris per invoice pelanggan."""

    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "Revenue"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    data = df.copy()

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )

    data = data.dropna(
        subset=["CustomerID", "InvoiceNo", "InvoiceDate", "Revenue"]
    )

    # Hanya transaksi dengan nilai revenue positif.
    data = data[
        (data["Revenue"] > 0)
        & (data["InvoiceNo"].astype(str).str.strip() != "")
    ].copy()

    # Satu invoice = satu transaksi.
    invoices = (
        data.groupby(["CustomerID", "InvoiceNo"], as_index=False)
        .agg(
            TransactionDate=("InvoiceDate", "min"),
            Monetary=("Revenue", "sum"),
        )
    )

    invoices = invoices.sort_values(
        ["CustomerID", "TransactionDate", "InvoiceNo"]
    ).reset_index(drop=True)

    if invoices.empty:
        raise ValueError("Tidak ada invoice valid setelah agregasi.")

    return invoices


def build_clv_summary(
    df: pd.DataFrame,
    observation_end: pd.Timestamp,
) -> pd.DataFrame:
    """Membuat ringkasan BG/NBD dan Gamma-Gamma berbasis invoice.

    Frequency:
        Jumlah invoice setelah invoice pertama.

    Recency:
        Selisih hari antara invoice pertama dan invoice terakhir.

    T:
        Selisih hari antara invoice pertama dan akhir periode observasi.

    Monetary:
        Rata-rata nilai invoice berulang, tidak termasuk invoice pertama.

    Pelanggan dengan satu invoice tetap dipertahankan:
        frequency = 0
        recency = 0
        monetary = 0
    """

    invoices = build_invoice_table(df)

    observation_end = pd.Timestamp(observation_end).normalize()
    observation_end_exclusive = observation_end + pd.Timedelta(days=1)

    if invoices["TransactionDate"].max() >= observation_end_exclusive:
        raise ValueError(
            "Ada transaksi di luar periode calibration: potensi data leakage."
        )

    rows = []

    for customer_id, group in invoices.groupby("CustomerID"):
        group = group.sort_values("TransactionDate")

        dates = group["TransactionDate"]
        monetary_values = group["Monetary"]

        transaction_count = len(group)
        first_date = dates.iloc[0]
        last_date = dates.iloc[-1]

        frequency = transaction_count - 1

        recency = (
            (last_date - first_date).total_seconds() / 86400
        )

        T = (
            (observation_end_exclusive - first_date).total_seconds() / 86400
        )

        # Monetary hanya dihitung dari invoice setelah transaksi pertama.
        if frequency > 0:
            monetary_value = float(monetary_values.iloc[1:].mean())
        else:
            monetary_value = 0.0

        rows.append({
            "CustomerID": customer_id,
            "lt_frequency": float(frequency),
            "lt_recency": float(recency),
            "lt_T": float(T),
            "lt_monetary": monetary_value,
            "lt_transaction_count": int(transaction_count),
        })

    summary = pd.DataFrame(rows).set_index("CustomerID")

    if summary.empty:
        raise ValueError("Ringkasan pelanggan kosong.")

    if (summary["lt_frequency"] < 0).any():
        raise ValueError("Frequency tidak boleh negatif.")

    if (summary["lt_recency"] < 0).any():
        raise ValueError("Recency tidak boleh negatif.")

    if (summary["lt_T"] <= 0).any():
        raise ValueError("T harus lebih besar dari 0.")

    if (summary["lt_recency"] > summary["lt_T"]).any():
        raise ValueError("Ditemukan recency yang lebih besar dari T.")

    return summary


# ============================================================
# 3. FITTING BG/NBD
# ============================================================

def _is_convergence_warning(message: str) -> bool:
    """Mengenali pesan warning yang mengindikasikan masalah optimisasi."""

    text = message.lower()

    keywords = (
        "failed to converge",
        "did not converge",
        "precision loss",
        "desired error not necessarily achieved",
        "maximum number of iterations",
        "maximum iterations",
        "optimization failed",
        "optimisation failed",
    )

    return any(keyword in text for keyword in keywords)


def fit_bgnbd(
    summary: pd.DataFrame,
    penalizer_coef: float = 0.05,
    fallback_penalizers: Tuple[float, ...] = (0.1, 0.2, 0.5),
) -> Tuple[BetaGeoFitter, float]:
    """Fit BG/NBD dengan pemeriksaan parameter dan warning konvergensi.

    Pelanggan frequency=0 dan recency=0 tetap disertakan.

    Catatan:
        lifetimes tidak menyediakan atribut status optimizer yang konsisten
        pada semua versinya. Oleh karena itu, fungsi memeriksa warning
        optimisasi, exception, serta validitas parameter hasil fitting.
    """

    required = ["lt_frequency", "lt_recency", "lt_T"]

    if summary[required].isna().any().any():
        raise ValueError("Input BG/NBD mengandung nilai kosong.")

    if (summary["lt_frequency"] < 0).any():
        raise ValueError("Frequency tidak boleh negatif.")

    if (summary["lt_recency"] < 0).any():
        raise ValueError("Recency tidak boleh negatif.")

    if (summary["lt_T"] <= 0).any():
        raise ValueError("T harus lebih besar dari 0.")

    if (summary["lt_recency"] > summary["lt_T"]).any():
        raise ValueError("Recency tidak boleh lebih besar dari T.")

    candidates = list(dict.fromkeys(
        (penalizer_coef, *fallback_penalizers)
    ))

    failures = []

    for pen in candidates:
        try:
            bgf = BetaGeoFitter(penalizer_coef=pen)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                bgf.fit(
                    summary["lt_frequency"],
                    summary["lt_recency"],
                    summary["lt_T"],
                )

            warning_messages = [
                str(w.message) for w in caught
            ]

            convergence_warnings = [
                msg for msg in warning_messages
                if _is_convergence_warning(msg)
            ]

            params = bgf.params_.astype(float)

            if not np.isfinite(params.to_numpy()).all():
                raise ValueError("Parameter BG/NBD tidak finite.")

            if convergence_warnings:
                failures.append(
                    f"penalizer={pen}: {convergence_warnings}"
                )
                continue

            if pen != penalizer_coef:
                print(
                    f"[BG/NBD] Penalizer awal {penalizer_coef} "
                    f"tidak memenuhi pemeriksaan; memakai {pen}."
                )

            if warning_messages:
                print(f"[BG/NBD] Warning pada penalizer {pen}:")
                for msg in warning_messages:
                    print(f"  - {msg}")

            print(f"[BG/NBD] Fitting selesai dengan penalizer={pen}.")
            return bgf, pen

        except (ConvergenceError, ValueError, FloatingPointError) as exc:
            failures.append(f"penalizer={pen}: {exc}")

    raise RuntimeError(
        "BG/NBD tidak lolos pemeriksaan pada seluruh penalizer.\n"
        + "\n".join(failures)
    )


def diagnose_bgnbd(
    bgf: BetaGeoFitter,
    tol: float = 0.01,
) -> Dict[str, object]:
    """Memeriksa indikasi degenerasi parameter dropout BG/NBD."""

    params = bgf.params_

    a = float(params["a"])
    b = float(params["b"])

    return {
        "params": {
            k: float(v) for k, v in params.items()
        },
        "dropout_degenerate": bool(min(a, b) < tol),
        "dropout_parameter_min": min(a, b),
    }


# ============================================================
# 4. FITTING GAMMA-GAMMA
# ============================================================

def fit_gamma_gamma(
    summary: pd.DataFrame,
    penalizer_coef: float = 0.0,
    fallback_penalizers: Tuple[float, ...] = (0.001, 0.01, 0.05),
) -> Tuple[GammaGammaFitter, float]:
    """Fit Gamma-Gamma pada pelanggan dengan transaksi berulang.

    Syarat:
        frequency > 0
        monetary > 0
        q > 1

    Monetary menggunakan rata-rata nilai invoice berulang.
    """

    repeat = summary[
        (summary["lt_frequency"] > 0)
        & (summary["lt_monetary"] > 0)
    ].copy()

    if len(repeat) < 50:
        raise ValueError(
            f"Customer repeat terlalu sedikit untuk Gamma-Gamma: {len(repeat)}"
        )

    corr = float(
        repeat[["lt_frequency", "lt_monetary"]].corr().iloc[0, 1]
    )

    if not np.isfinite(corr):
        raise ValueError("Korelasi frequency dan monetary tidak valid.")

    candidates = list(dict.fromkeys(
        (penalizer_coef, *fallback_penalizers)
    ))

    failures = []

    for pen in candidates:
        try:
            ggf = GammaGammaFitter(penalizer_coef=pen)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ggf.fit(
                    repeat["lt_frequency"],
                    repeat["lt_monetary"],
                )

            warning_messages = [
                str(w.message) for w in caught
            ]

            params = ggf.params_.astype(float)

            if not np.isfinite(params.to_numpy()).all():
                failures.append(f"penalizer={pen}: parameter tidak finite.")
                continue

            q = float(params["q"])

            if q <= 1.0:
                failures.append(
                    f"penalizer={pen}: q={q:.6f} tidak memenuhi q > 1."
                )
                continue

            convergence_warnings = [
                msg for msg in warning_messages
                if _is_convergence_warning(msg)
            ]

            if convergence_warnings:
                failures.append(
                    f"penalizer={pen}: {convergence_warnings}"
                )
                continue

            if pen != penalizer_coef:
                print(
                    f"[Gamma-Gamma] Penalizer awal {penalizer_coef} "
                    f"tidak memenuhi pemeriksaan; memakai {pen}."
                )

            if warning_messages:
                print(f"[Gamma-Gamma] Warning pada penalizer {pen}:")
                for msg in warning_messages:
                    print(f"  - {msg}")

            print(
                f"[Gamma-Gamma] Fitting selesai dengan penalizer={pen}, "
                f"q={q:.4f}."
            )

            return ggf, corr

        except (ConvergenceError, ValueError, FloatingPointError) as exc:
            failures.append(f"penalizer={pen}: {exc}")

    raise ValueError(
        "Gamma-Gamma tidak lolos pemeriksaan.\n"
        + "\n".join(failures)
    )


# ============================================================
# 5. PREDIKSI PEMBELIAN
# ============================================================

def _cumulative_expected_purchases(
    bgf: BetaGeoFitter,
    summary: pd.DataFrame,
    t_days: float,
) -> Tuple[pd.Series, pd.Series]:
    """Menghitung ekspektasi kumulatif pembelian BG/NBD.

    Pelanggan frequency=0 tetap dihitung.

    Prediksi yang tidak finite tidak langsung diganti dengan nilai populasi.
    Fungsi akan mencoba memeriksa hasilnya dan memberikan fallback hanya
    untuk nilai yang tidak valid.
    """

    if t_days < 0:
        raise ValueError("t_days tidak boleh negatif.")

    if t_days == 0:
        zero = pd.Series(0.0, index=summary.index)
        return zero, pd.Series(False, index=summary.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        cond = bgf.conditional_expected_number_of_purchases_up_to_time(
            t_days,
            summary["lt_frequency"],
            summary["lt_recency"],
            summary["lt_T"],
        )

    cond = pd.Series(
        np.asarray(cond, dtype=float),
        index=summary.index,
    )

    bad = (~np.isfinite(cond)) | (cond < 0)

    if bad.any():
        # Fallback populasi hanya untuk prediksi yang tidak valid.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            population_prediction = float(
                bgf.expected_number_of_purchases_up_to_time(t_days)
            )

        if not np.isfinite(population_prediction) or population_prediction < 0:
            raise ValueError(
                "Prediksi BG/NBD tidak valid dan fallback populasi gagal."
            )

        cond.loc[bad] = population_prediction

    return cond, bad


# ============================================================
# 6. PERHITUNGAN PREDICTIVE CLV
# ============================================================

def calculate_predictive_clv(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    summary: pd.DataFrame,
    horizon_months: int,
    discount_rate: float = 0.01,
) -> pd.DataFrame:
    """Menghitung Predictive CLV berdasarkan invoice.

    CLV = jumlah ekspektasi pendapatan bulanan yang telah didiskontokan.

    ExpectedPurchases:
        Ekspektasi jumlah invoice selama horizon prediksi.

    ExpectedAOV:
        Ekspektasi rata-rata nilai invoice.

    Pelanggan dengan frequency=0 tetap memperoleh estimasi AOV
    menggunakan conditional_expected_average_profit dari Gamma-Gamma.
    """

    if horizon_months <= 0:
        raise ValueError("horizon_months harus lebih besar dari 0.")

    if discount_rate < 0:
        raise ValueError("discount_rate tidak boleh negatif.")

    out = pd.DataFrame(index=summary.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        out["ProbAlive"] = bgf.conditional_probability_alive(
            summary["lt_frequency"],
            summary["lt_recency"],
            summary["lt_T"],
        )

        out["ExpectedAOV"] = ggf.conditional_expected_average_profit(
            summary["lt_frequency"],
            summary["lt_monetary"],
        )

    out["ProbAlive"] = np.asarray(out["ProbAlive"], dtype=float)
    out["ExpectedAOV"] = np.asarray(out["ExpectedAOV"], dtype=float)

    if not np.isfinite(out["ExpectedAOV"]).all():
        raise ValueError(
            "Gamma-Gamma menghasilkan ExpectedAOV tidak valid."
        )

    if (out["ExpectedAOV"] < 0).any():
        raise ValueError("ExpectedAOV tidak boleh negatif.")

    previous, imputed = _cumulative_expected_purchases(
        bgf, summary, 0
    )

    clv = pd.Series(0.0, index=summary.index)

    for month in range(1, horizon_months + 1):
        current, bad = _cumulative_expected_purchases(
            bgf,
            summary,
            30 * month,
        )

        incremental_purchases = current - previous

        # Perbedaan kecil akibat numerik tidak boleh menghasilkan CLV negatif.
        if (incremental_purchases < -1e-8).any():
            raise ValueError(
                "Ekspektasi pembelian kumulatif tidak monoton."
            )

        incremental_purchases = incremental_purchases.clip(lower=0)

        monthly_revenue = incremental_purchases * out["ExpectedAOV"]

        clv += monthly_revenue / ((1 + discount_rate) ** month)

        previous = current
        imputed = imputed | bad

    out["ExpectedPurchases"] = previous
    out["PredictiveCLV"] = clv
    out["ImputedPrediction"] = imputed

    return out[
        [
            "ProbAlive",
            "ExpectedPurchases",
            "ExpectedAOV",
            "PredictiveCLV",
            "ImputedPrediction",
        ]
    ]


# ============================================================
# 7. EVALUASI HOLDOUT
# ============================================================

def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    """Menghitung Spearman dengan penanganan nilai konstan."""

    valid = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(valid) < 2:
        return float("nan")

    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return float("nan")

    return float(spearmanr(valid["x"], valid["y"])[0])


def evaluate_clv_prediction(
    summary_cal: pd.DataFrame,
    pred: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Mengevaluasi prediksi jumlah invoice dan CLV terhadap holdout.

    Populasi evaluasi:
        Seluruh pelanggan yang sudah ada di calibration.

    Pelanggan yang baru muncul pada holdout tidak disertakan karena
    mereka belum memiliki informasi historis pada saat prediksi dibuat.

    ActualPurchases:
        Jumlah invoice holdout per pelanggan.

    ActualRevenue:
        Total revenue holdout per pelanggan.
    """

    invoices_holdout = build_invoice_table(holdout_df)

    actual = invoices_holdout.groupby("CustomerID").agg(
        ActualPurchases=("InvoiceNo", "nunique"),
        ActualRevenue=("Monetary", "sum"),
    )

    ev = pred.join(actual, how="left")
    ev = ev.join(summary_cal[["lt_frequency"]], how="left")

    ev[["ActualPurchases", "ActualRevenue"]] = (
        ev[["ActualPurchases", "ActualRevenue"]].fillna(0.0)
    )

    ev = ev.replace([np.inf, -np.inf], np.nan)

    required_predictions = [
        "ExpectedPurchases",
        "PredictiveCLV",
    ]

    invalid_prediction = ev[required_predictions].isna().any(axis=1)

    if invalid_prediction.any():
        raise ValueError(
            f"Ada {int(invalid_prediction.sum())} pelanggan "
            "dengan prediksi tidak valid. Periksa hasil fitting BG/NBD/Gamma-Gamma."
        )

    hist_monetary = (
        summary_cal["lt_frequency"] * summary_cal["lt_monetary"]
    )

    metrics = {
        "n_customers_evaluated": float(len(ev)),
        "n_active_in_holdout": float(
            (ev["ActualPurchases"] > 0).sum()
        ),
        "n_frequency_zero": float(
            (summary_cal["lt_frequency"] == 0).sum()
        ),
        "n_imputed_predictions": float(
            pred["ImputedPrediction"].sum()
        ),
        "purchases_spearman": _safe_spearman(
            ev["ExpectedPurchases"], ev["ActualPurchases"]
        ),
        "purchases_pearson": float(
            ev["ExpectedPurchases"].corr(ev["ActualPurchases"])
        ),
        "purchases_MAE": float(
            (ev["ExpectedPurchases"] - ev["ActualPurchases"]).abs().mean()
        ),
        "purchases_RMSE": float(
            np.sqrt(
                (
                    (ev["ExpectedPurchases"] - ev["ActualPurchases"]) ** 2
                ).mean()
            )
        ),
        "purchases_total_predicted": float(
            ev["ExpectedPurchases"].sum()
        ),
        "purchases_total_actual": float(
            ev["ActualPurchases"].sum()
        ),
        "clv_spearman_vs_revenue": _safe_spearman(
            ev["PredictiveCLV"], ev["ActualRevenue"]
        ),
        "clv_pearson_vs_revenue": float(
            ev["PredictiveCLV"].corr(ev["ActualRevenue"])
        ),
        "clv_total_predicted": float(
            ev["PredictiveCLV"].sum()
        ),
        "revenue_total_actual": float(
            ev["ActualRevenue"].sum()
        ),
        "baseline_spearman_frequency": _safe_spearman(
            ev["lt_frequency"], ev["ActualRevenue"]
        ),
        "baseline_spearman_hist_monetary": _safe_spearman(
            hist_monetary.loc[ev.index], ev["ActualRevenue"]
        ),
    }

    # Decile berdasarkan urutan CLV.
    number_of_deciles = min(10, len(ev))

    if number_of_deciles >= 2:
        ev["CLV_Decile"] = pd.qcut(
            ev["PredictiveCLV"].rank(method="first"),
            q=number_of_deciles,
            labels=False,
        ) + 1

        decile = ev.groupby("CLV_Decile", observed=True).agg(
            n=("PredictiveCLV", "size"),
            mean_predicted_clv=("PredictiveCLV", "mean"),
            mean_actual_revenue=("ActualRevenue", "mean"),
            mean_expected_purchases=("ExpectedPurchases", "mean"),
            mean_actual_purchases=("ActualPurchases", "mean"),
        ).reset_index()

    else:
        decile = pd.DataFrame()

    return metrics, decile


# ============================================================
# 8. EVALUASI MULTI-SPLIT
# ============================================================

def evaluate_clv_multi_split(
    df: pd.DataFrame,
    holdout_days_list: Tuple[int, ...],
    bgnbd_penalizer: float = 0.05,
    gg_penalizer: float = 0.0,
    clv_horizon_months: int = 3,
    discount_rate: float = 0.01,
) -> pd.DataFrame:
    """Mengevaluasi CLV pada beberapa periode holdout."""

    rows: List[dict] = []

    for holdout_days in holdout_days_list:
        try:
            calibration, holdout, cutoff = create_calibration_holdout(
                df, holdout_days
            )

            summary_cal = build_clv_summary(
                calibration, cutoff
            )

            bgf, used_bgnbd_penalizer = fit_bgnbd(
                summary_cal,
                bgnbd_penalizer,
            )

            ggf, corr = fit_gamma_gamma(
                summary_cal,
                gg_penalizer,
            )

            eval_months = max(
                1,
                int(round(holdout_days / 30)),
            )

            pred = calculate_predictive_clv(
                bgf,
                ggf,
                summary_cal,
                eval_months,
                discount_rate,
            )

            metrics, _ = evaluate_clv_prediction(
                summary_cal,
                pred,
                holdout,
            )

            diag = diagnose_bgnbd(bgf)

            rows.append({
                "holdout_days": holdout_days,
                "cutoff": cutoff.date(),
                "n_customers": int(
                    metrics["n_customers_evaluated"]
                ),
                "bgnbd_penalizer": used_bgnbd_penalizer,
                "gg_frequency_monetary_corr": corr,
                "dropout_degenerate": diag["dropout_degenerate"],
                "n_imputed_predictions": metrics["n_imputed_predictions"],
                "purchases_total_predicted": metrics[
                    "purchases_total_predicted"
                ],
                "purchases_total_actual": metrics[
                    "purchases_total_actual"
                ],
                "purchases_ratio": (
                    metrics["purchases_total_predicted"]
                    / metrics["purchases_total_actual"]
                    if metrics["purchases_total_actual"] > 0
                    else np.nan
                ),
                "purchases_spearman": metrics["purchases_spearman"],
                "purchases_MAE": metrics["purchases_MAE"],
                "purchases_RMSE": metrics["purchases_RMSE"],
                "clv_spearman_vs_revenue": metrics[
                    "clv_spearman_vs_revenue"
                ],
                "baseline_spearman_hist_monetary": metrics[
                    "baseline_spearman_hist_monetary"
                ],
            })

        except Exception as exc:
            rows.append({
                "holdout_days": holdout_days,
                "cutoff": None,
                "error": str(exc),
            })

    return pd.DataFrame(rows)


# ============================================================
# 9. PERBANDINGAN MODEL COUNTING
# ============================================================

def compare_counting_models(
    summary: pd.DataFrame,
    penalizer_coef: float = 0.05,
) -> pd.DataFrame:
    """Membandingkan BG/NBD, Pareto/NBD, dan Modified BG/NBD."""

    from lifetimes import ModifiedBetaGeoFitter, ParetoNBDFitter

    models = {
        "BG/NBD": BetaGeoFitter(penalizer_coef=penalizer_coef),
        "Pareto/NBD": ParetoNBDFitter(penalizer_coef=penalizer_coef),
        "Modified BG/NBD": ModifiedBetaGeoFitter(
            penalizer_coef=penalizer_coef
        ),
    }

    rows = []

    for name, model in models.items():
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                model.fit(
                    summary["lt_frequency"],
                    summary["lt_recency"],
                    summary["lt_T"],
                )

                pa = model.conditional_probability_alive(
                    summary["lt_frequency"],
                    summary["lt_recency"],
                    summary["lt_T"],
                )

            warning_messages = [
                str(w.message) for w in caught
            ]

            params = model.params_.astype(float)

            rows.append({
                "Model": name,
                "Params": {
                    k: float(v) for k, v in params.items()
                },
                "MeanProbAlive": float(np.mean(pa)),
                "StdProbAlive": float(np.std(pa)),
                "ConvergenceWarning": any(
                    _is_convergence_warning(msg)
                    for msg in warning_messages
                ),
                "Warnings": warning_messages,
            })

        except Exception as exc:
            rows.append({
                "Model": name,
                "Params": None,
                "MeanProbAlive": np.nan,
                "StdProbAlive": np.nan,
                "ConvergenceWarning": True,
                "Warnings": [str(exc)],
            })

    return pd.DataFrame(rows)
