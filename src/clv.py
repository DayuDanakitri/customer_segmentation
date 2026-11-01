"""Predictive Customer Lifetime Value: BG/NBD + Gamma-Gamma (library `lifetimes`).

Alur metodologis:
    1. create_calibration_holdout : bagi transaksi berdasarkan waktu.
    2. build_clv_summary          : ringkasan frequency/recency/T/monetary dari satu periode.
    3. fit_bgnbd, fit_gamma_gamma : fit HANYA pada data calibration (validasi).
    4. evaluate_clv_prediction    : bandingkan prediksi dengan holdout (data yang tidak dipakai fit).
    5. calculate_predictive_clv   : hitung CLV. Untuk fitur clustering, model di-fit ulang
                                    pada seluruh periode observasi dan memprediksi ke MASA DEPAN
                                    (di luar data), sehingga tidak ada informasi masa depan yang bocor.

Catatan definisi: `lifetimes` menghitung frequency dari jumlah HARI pembelian berbeda (freq="D"),
bukan jumlah invoice, dan monetary_value = rata-rata revenue per hari pembelian ULANGAN
(pembelian pertama tidak dihitung). Kolom hasilnya diberi prefix `lt_` agar tidak tertukar
dengan Recency/Frequency/Monetary pada RFM.
"""
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import ConvergenceError, summary_data_from_transaction_data
from scipy.stats import spearmanr


def create_calibration_holdout(
    df: pd.DataFrame,
    holdout_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Membagi transaksi menjadi calibration dan holdout berdasarkan waktu.

    Cutoff = tanggal (tanpa jam) terakhir di data dikurangi holdout_days.
    Calibration: InvoiceDate <  cutoff + 1 hari (yaitu s.d. akhir hari cutoff).
    Holdout    : InvoiceDate >= cutoff + 1 hari.

    Args:
        df: Data transaksi bersih.
        holdout_days: Panjang holdout period dalam hari.

    Returns:
        (calibration_df, holdout_df, cutoff_date).

    Raises:
        ValueError: Jika holdout_days tidak valid atau salah satu periode kosong.
    """
    last_day = df["InvoiceDate"].max().normalize()
    first_day = df["InvoiceDate"].min().normalize()
    if holdout_days <= 0 or holdout_days >= (last_day - first_day).days:
        raise ValueError(f"holdout_days={holdout_days} tidak valid untuk rentang data.")
    cutoff = last_day - pd.Timedelta(days=holdout_days)
    boundary = cutoff + pd.Timedelta(days=1)
    calibration = df[df["InvoiceDate"] < boundary].copy()
    holdout = df[df["InvoiceDate"] >= boundary].copy()
    if calibration.empty or holdout.empty:
        raise ValueError("Calibration atau holdout kosong; ubah holdout_days.")
    return calibration, holdout, cutoff


def build_clv_summary(df: pd.DataFrame, observation_end: pd.Timestamp) -> pd.DataFrame:
    """Membuat ringkasan input BG/NBD dan Gamma-Gamma per CustomerID.

    Args:
        df: Transaksi pada periode observasi (tidak boleh melewati observation_end).
        observation_end: Akhir periode observasi.

    Returns:
        DataFrame index CustomerID dengan kolom lt_frequency, lt_recency, lt_T,
        lt_monetary (dalam hari; monetary = rata-rata revenue per hari pembelian ulangan).
    """
    if df["InvoiceDate"].max() > observation_end + pd.Timedelta(days=1):
        raise ValueError("Ada transaksi setelah observation_end: potensi data leakage.")
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="Revenue",
        observation_period_end=observation_end,
        freq="D",
    )
    return summary.rename(columns={
        "frequency": "lt_frequency", "recency": "lt_recency",
        "T": "lt_T", "monetary_value": "lt_monetary",
    })


def fit_bgnbd(
    summary: pd.DataFrame,
    penalizer_coef: float = 0.05,
    fallback_penalizers: Tuple[float, ...] = (0.1, 0.2, 0.5),
) -> Tuple[BetaGeoFitter, float]:
    """Fit model BG/NBD (prediksi jumlah transaksi masa depan).

    Optimasi BG/NBD pada data retail sering gagal konvergen dengan penalizer kecil.
    Fungsi ini mencoba `penalizer_coef`, lalu `fallback_penalizers` berurutan sampai
    konvergen, dan melaporkan penalizer yang akhirnya dipakai (tidak diam-diam).

    Args:
        summary: Output build_clv_summary.
        penalizer_coef: Penalizer awal.
        fallback_penalizers: Penalizer cadangan bila belum konvergen.

    Returns:
        (bgf, penalizer_yang_dipakai).

    Raises:
        RuntimeError: Jika semua penalizer gagal konvergen.
    """
    tried = []
    for pen in (penalizer_coef, *fallback_penalizers):
        try:
            bgf = BetaGeoFitter(penalizer_coef=pen)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bgf.fit(summary["lt_frequency"], summary["lt_recency"], summary["lt_T"])
            if not np.isfinite(bgf.params_.values).all():
                raise ConvergenceError("parameter tidak finite")
            if pen != penalizer_coef:
                print(f"[fit_bgnbd] penalizer {penalizer_coef} gagal konvergen; memakai {pen}.")
            return bgf, pen
        except ConvergenceError:
            tried.append(pen)
    raise RuntimeError(f"BG/NBD gagal konvergen untuk penalizer {tried}.")


def diagnose_bgnbd(bgf: BetaGeoFitter, tol: float = 0.01) -> Dict[str, object]:
    """Memeriksa apakah komponen dropout BG/NBD terdegenerasi.

    Jika parameter a atau b mendekati 0, distribusi probabilitas dropout menjadi
    tidak informatif (P(alive) ~ 1 untuk semua customer) dan model efektif menjadi
    model pembelian Gamma-Poisson tanpa churn.

    Returns:
        Dict berisi parameter, flag `dropout_degenerate`, dan rata-rata E[dropout prob].
    """
    a, b = float(bgf.params_["a"]), float(bgf.params_["b"])
    return {
        "params": {k: round(float(v), 4) for k, v in bgf.params_.items()},
        "dropout_degenerate": bool(min(a, b) < tol),
    }


def fit_gamma_gamma(
    summary: pd.DataFrame,
    penalizer_coef: float = 0.0,
    fallback_penalizers: Tuple[float, ...] = (0.0,),
) -> Tuple[GammaGammaFitter, float]:
    """Fit model Gamma-Gamma (estimasi nilai rata-rata transaksi).

    Hanya memakai customer dengan lt_frequency > 0 dan lt_monetary > 0 (syarat model).
    Syarat validitas: parameter q > 1 (jika tidak, rata-rata nilai transaksi tidak
    terdefinisi dan CLV dapat bernilai negatif). Penalizer besar dapat merusak fit
    pada data dengan ekor berat, sehingga fungsi mencoba penalizer cadangan.

    Returns:
        (ggf, corr_frequency_monetary). Model mengasumsikan frequency dan monetary independen;
        korelasi dikembalikan untuk diperiksa.

    Raises:
        ValueError: Data repeat terlalu sedikit atau tidak ada penalizer yang memberi q > 1.
    """
    repeat = summary[(summary["lt_frequency"] > 0) & (summary["lt_monetary"] > 0)]
    if len(repeat) < 50:
        raise ValueError(f"Customer repeat terlalu sedikit untuk Gamma-Gamma: {len(repeat)}")
    corr = float(repeat[["lt_frequency", "lt_monetary"]].corr().iloc[0, 1])
    for pen in (penalizer_coef, *fallback_penalizers):
        ggf = GammaGammaFitter(penalizer_coef=pen)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ggf.fit(repeat["lt_frequency"], repeat["lt_monetary"])
        if float(ggf.params_["q"]) > 1.0:
            if pen != penalizer_coef:
                print(f"[fit_gamma_gamma] penalizer {penalizer_coef} memberi q<=1; memakai {pen}.")
            return ggf, corr
    raise ValueError("Gamma-Gamma tidak valid (q <= 1) untuk semua penalizer yang dicoba.")


def _cumulative_expected_purchases(
    bgf: BetaGeoFitter,
    summary: pd.DataFrame,
    t_days: float,
) -> Tuple[pd.Series, pd.Series]:
    """Ekspektasi kumulatif jumlah pembelian sampai t_days ke depan (kondisional).

    Jika hasil kondisional tidak finite (terjadi pada customer sangat baru saat parameter
    dropout BG/NBD terdegenerasi), diganti ekspektasi populasi tak-kondisional.

    Returns:
        (nilai_kumulatif, flag_imputasi).
    """
    if t_days <= 0:
        zero = pd.Series(0.0, index=summary.index)
        return zero, pd.Series(False, index=summary.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cond = bgf.conditional_expected_number_of_purchases_up_to_time(
            t_days, summary["lt_frequency"], summary["lt_recency"], summary["lt_T"])
        uncond = float(bgf.expected_number_of_purchases_up_to_time(t_days))
    cond = pd.Series(np.asarray(cond, dtype=float), index=summary.index)
    bad = ~np.isfinite(cond)
    cond[bad] = uncond
    return cond, bad


def calculate_predictive_clv(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    summary: pd.DataFrame,
    horizon_months: int,
    discount_rate: float = 0.01,
) -> pd.DataFrame:
    """Menghitung Predictive CLV untuk horizon tertentu (1 bulan = 30 hari).

    PredictiveCLV = sum_{i=1..H} [E(N(30i)) - E(N(30(i-1)))] x E[AOV] / (1 + d)^i
        E(N(.)) : ekspektasi jumlah pembelian dari BG/NBD (kondisional).
        E[AOV]  : ekspektasi nilai per transaksi dari Gamma-Gamma. Untuk customer tanpa
                  pembelian ulangan (lt_frequency = 0) hasilnya rata-rata populasi (shrinkage).
        d       : discount rate bulanan.

    Returns:
        DataFrame index CustomerID: ProbAlive, ExpectedPurchases (sampai horizon),
        ExpectedAOV, PredictiveCLV, ImputedPrediction (True bila ekspektasi populasi dipakai).
    """
    out = pd.DataFrame(index=summary.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["ProbAlive"] = bgf.conditional_probability_alive(
            summary["lt_frequency"], summary["lt_recency"], summary["lt_T"])
        aov = ggf.conditional_expected_average_profit(
            summary["lt_frequency"], summary["lt_monetary"])
    out["ExpectedAOV"] = np.asarray(aov, dtype=float)

    prev, imputed = _cumulative_expected_purchases(bgf, summary, 0)
    clv = pd.Series(0.0, index=summary.index)
    for i in range(1, horizon_months + 1):
        cur, bad = _cumulative_expected_purchases(bgf, summary, 30 * i)
        imputed = imputed | bad
        clv += (cur - prev) * out["ExpectedAOV"] / (1.0 + discount_rate) ** i
        prev = cur
    out["ExpectedPurchases"] = prev
    out["PredictiveCLV"] = clv
    out["ImputedPrediction"] = imputed
    return out[["ProbAlive", "ExpectedPurchases", "ExpectedAOV", "PredictiveCLV", "ImputedPrediction"]]


def evaluate_clv_prediction(
    summary_cal: pd.DataFrame,
    pred: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Mengevaluasi prediksi (dibuat dari calibration) terhadap kejadian nyata di holdout.

    Hanya customer yang ada di calibration yang dievaluasi (customer baru di holdout
    tidak mungkin diprediksi). Aktual di holdout: jumlah hari pembelian dan total revenue
    (0 jika tidak membeli).

    Args:
        summary_cal: Ringkasan calibration (lt_*).
        pred: Output calculate_predictive_clv (dihitung dari calibration, horizon = holdout).
        holdout_df: Transaksi holdout.

    Returns:
        (metrics, decile_table). Metrik mencakup Spearman/Pearson, MAE, RMSE, total
        prediksi vs aktual, serta pembanding naif (Spearman dari frequency dan
        monetary historis calibration).
    """
    h = holdout_df.copy()
    h["Day"] = h["InvoiceDate"].dt.normalize()
    actual = h.groupby("CustomerID").agg(
        ActualPurchases=("Day", "nunique"), ActualRevenue=("Revenue", "sum"))
    ev = pred.join(actual, how="left").join(summary_cal[["lt_frequency"]])
    ev[["ActualPurchases", "ActualRevenue"]] = ev[["ActualPurchases", "ActualRevenue"]].fillna(0.0)

    hist_monetary = (summary_cal["lt_frequency"] * summary_cal["lt_monetary"])
    metrics: Dict[str, float] = {
        "n_customers_evaluated": float(len(ev)),
        "n_active_in_holdout": float((ev["ActualPurchases"] > 0).sum()),
        "purchases_spearman": float(spearmanr(ev["ExpectedPurchases"], ev["ActualPurchases"])[0]),
        "purchases_pearson": float(ev["ExpectedPurchases"].corr(ev["ActualPurchases"])),
        "purchases_MAE": float((ev["ExpectedPurchases"] - ev["ActualPurchases"]).abs().mean()),
        "purchases_RMSE": float(np.sqrt(((ev["ExpectedPurchases"] - ev["ActualPurchases"]) ** 2).mean())),
        "purchases_total_predicted": float(ev["ExpectedPurchases"].sum()),
        "purchases_total_actual": float(ev["ActualPurchases"].sum()),
        "clv_spearman_vs_revenue": float(spearmanr(ev["PredictiveCLV"], ev["ActualRevenue"])[0]),
        "clv_pearson_vs_revenue": float(ev["PredictiveCLV"].corr(ev["ActualRevenue"])),
        "clv_total_predicted": float(ev["PredictiveCLV"].sum()),
        "revenue_total_actual": float(ev["ActualRevenue"].sum()),
        "baseline_spearman_frequency": float(spearmanr(ev["lt_frequency"], ev["ActualRevenue"])[0]),
        "baseline_spearman_hist_monetary": float(spearmanr(hist_monetary.loc[ev.index], ev["ActualRevenue"])[0]),
    }

    ev["CLV_Decile"] = pd.qcut(ev["PredictiveCLV"].rank(method="first"), 10,
                               labels=list(range(1, 11))).astype(int)
    decile = ev.groupby("CLV_Decile").agg(
        n=("PredictiveCLV", "size"),
        mean_predicted_clv=("PredictiveCLV", "mean"),
        mean_actual_revenue=("ActualRevenue", "mean"),
        mean_expected_purchases=("ExpectedPurchases", "mean"),
        mean_actual_purchases=("ActualPurchases", "mean"),
    ).reset_index()
    return metrics, decile

def evaluate_clv_multi_split(
    df: pd.DataFrame,
    holdout_days_list: Tuple[int, ...],
    bgnbd_penalizer: float = 0.05,
    gg_penalizer: float = 0.0,
    clv_horizon_months: int = 3,
    discount_rate: float = 0.01,
) -> pd.DataFrame:
    """Mengevaluasi CLV pada beberapa cutoff calibration/holdout sekaligus.

    Satu split tunggal bisa kebetulan jatuh pada periode yang tidak representatif
    (mis. memuat lonjakan musiman). Fungsi ini mengulang seluruh alur Step 4.2-4.3
    (fit pada calibration, prediksi, evaluasi terhadap holdout) untuk setiap nilai
    di `holdout_days_list`, sehingga stabilitas hasil dapat dinilai lewat rata-rata
    dan variasi antar split, bukan satu angka yang bisa menyesatkan.

    Args:
        df: Data transaksi bersih.
        holdout_days_list: Daftar panjang holdout (hari) yang diuji, mis. (30, 60, 90, 120).
        bgnbd_penalizer, gg_penalizer, clv_horizon_months, discount_rate: Sama seperti
            fungsi fit/predict lain; horizon evaluasi mengikuti panjang holdout tiap split.

    Returns:
        DataFrame satu baris per split: holdout_days, cutoff, n_customers, dropout_degenerate,
        purchases_total_predicted, purchases_total_actual, purchases_ratio,
        purchases_spearman, clv_spearman_vs_revenue, baseline_spearman_hist_monetary.
    """
    rows: List[dict] = []
    for hd in holdout_days_list:
        try:
            calibration, holdout, cutoff = create_calibration_holdout(df, hd)
            summary_cal = build_clv_summary(calibration, cutoff)
            bgf, _ = fit_bgnbd(summary_cal, bgnbd_penalizer)
            ggf, _ = fit_gamma_gamma(summary_cal, gg_penalizer)
            eval_months = max(1, int(round(hd / 30)))
            pred = calculate_predictive_clv(bgf, ggf, summary_cal, eval_months, discount_rate)
            metrics, _ = evaluate_clv_prediction(summary_cal, pred, holdout)
            diag = diagnose_bgnbd(bgf)
            rows.append({
                "holdout_days": hd,
                "cutoff": cutoff.date(),
                "n_customers": int(metrics["n_customers_evaluated"]),
                "dropout_degenerate": diag["dropout_degenerate"],
                "purchases_total_predicted": metrics["purchases_total_predicted"],
                "purchases_total_actual": metrics["purchases_total_actual"],
                "purchases_ratio": metrics["purchases_total_predicted"] / metrics["purchases_total_actual"],
                "purchases_spearman": metrics["purchases_spearman"],
                "clv_spearman_vs_revenue": metrics["clv_spearman_vs_revenue"],
                "baseline_spearman_hist_monetary": metrics["baseline_spearman_hist_monetary"],
            })
        except Exception as exc:  # noqa: BLE001 - dicatat, bukan menghentikan seluruh evaluasi
            rows.append({"holdout_days": hd, "cutoff": None, "error": str(exc)})
    return pd.DataFrame(rows)


def compare_counting_models(summary: pd.DataFrame, penalizer_coef: float = 0.05) -> pd.DataFrame:
    """Membandingkan BG/NBD, Pareto/NBD, dan Modified BG/NBD pada ringkasan yang sama.

    Dipakai untuk mendokumentasikan secara eksplisit apakah dropout yang terdegenerasi
    (parameter mendekati 0) adalah ciri data atau artefak pilihan model: jika ketiganya
    konvergen ke kesimpulan yang sama, itu bukti pilihan model bukan akar masalah.

    Args:
        summary: Output build_clv_summary (periode calibration).
        penalizer_coef: Penalizer yang sama dipakai untuk ketiga model agar adil dibandingkan.

    Returns:
        DataFrame: Model, parameter (dict), mean_prob_alive, std_prob_alive.
    """
    from lifetimes import ModifiedBetaGeoFitter, ParetoNBDFitter

    models = {
        "BG/NBD": BetaGeoFitter(penalizer_coef=penalizer_coef),
        "Pareto/NBD": ParetoNBDFitter(penalizer_coef=penalizer_coef),
        "Modified BG/NBD": ModifiedBetaGeoFitter(penalizer_coef=penalizer_coef),
    }
    rows: List[dict] = []
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(summary["lt_frequency"], summary["lt_recency"], summary["lt_T"])
            pa = model.conditional_probability_alive(
                summary["lt_frequency"], summary["lt_recency"], summary["lt_T"])
        rows.append({
            "Model": name,
            "Params": {k: round(float(v), 4) for k, v in model.params_.items()},
            "MeanProbAlive": float(np.mean(pa)),
            "StdProbAlive": float(np.std(pa)),
        })
    return pd.DataFrame(rows)

