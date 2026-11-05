"""
Week 2 - Data collection, cleaning and preprocessing for logistics analysis.

Runs the full pipeline on the DataCo Smart Supply Chain dataset (or any similar
orders CSV) and saves evidence for the report:
  outputs/figures/*.png   - console-style and chart images for the DOC file
  outputs/tables/*.csv    - before/after tables
  outputs/pipeline_log.txt
  data/processed/clean_orders.csv

Usage:
    python week2_pipeline.py                      # uses data/raw/DataCoSupplyChainDataset.csv
    python week2_pipeline.py --input path/to.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sns.set_theme(style="whitegrid")
BLUE, GREY = "#2563eb", "#9ca3af"

FIG = Path("outputs/figures")
TAB = Path("outputs/tables")
PROC = Path("data/processed")
SAMPLE = Path("data/sample")
for p in (FIG, TAB, PROC, SAMPLE):
    p.mkdir(parents=True, exist_ok=True)

LOG = []


def say(msg=""):
    """Print and keep a line for the log file."""
    print(msg)
    LOG.append(str(msg))


def console_png(name, lines):
    """Render text as a terminal-style image (used as output evidence in the report)."""
    lines = [str(l)[:130] for l in lines]
    width = min(max(len(l) for l in lines) * 0.088 + 0.8, 16)
    height = 0.2 * len(lines) + 0.7
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor("#111827")
    fig.text(0.012, 0.975, "\n".join(lines), family="monospace", fontsize=9,
             va="top", ha="left", color="#e5e7eb")
    fig.savefig(FIG / name, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def pick(df, names):
    return [c for c in names if c in df.columns]


def snapshot(df, num_cols):
    """Data-quality numbers used for the before/after table."""
    neg = 0
    for c in pick(df, ["days_for_shipping_real", "order_item_quantity", "sales"]):
        neg += int((pd.to_numeric(df[c], errors="coerce") < 0).sum())
    iqr_out = 0
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        iqr_out += int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(df.isna().sum().sum() / df.size * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_values": neg,
        "iqr_outlier_cells": iqr_out,
    }


# ---------------------------------------------------------------- STEP 1
def step1_load(path):
    say("STEP 1 - DATA COLLECTION / LOADING")
    if not Path(path).exists():
        sys.exit(f"File not found: {path}\nDownload the dataset (see README) and place it there.")
    df = pd.read_csv(path, encoding="latin-1")      # DataCo is not UTF-8
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
    say(f"Loaded {path}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    dtypes = df.dtypes.value_counts().to_string().split("\n")
    show = pick(df, ["order_id", "order_date_dateorders", "shipping_mode", "order_region",
                     "days_for_shipping_real", "days_for_shipment_scheduled", "sales",
                     "order_item_quantity", "late_delivery_risk"])
    head = df[show].head(5).to_string().split("\n")
    mem = df.memory_usage(deep=True).sum() / 1e6
    lines = [f">>> df.shape", f"{df.shape}", f">>> memory: {mem:.1f} MB", ">>> df.dtypes.value_counts()",
             *dtypes, ">>> df[key_columns].head()", *head]
    console_png("01_dataset_overview.png", lines)
    return df


# ---------------------------------------------------------------- STEP 2
def step2_audit(df, num_cols):
    say("\nSTEP 2 - DATA QUALITY AUDIT (BEFORE CLEANING)")
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss.rename("missing_pct").round(2).to_csv(TAB / "missing_before.csv")
    top = miss.head(12)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top.index[::-1], top.values[::-1],
            color=[BLUE if v > 50 else GREY for v in top.values[::-1]])
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missing data by column, before cleaning (top 12)")
    fig.tight_layout()
    fig.savefig(FIG / "02_missing_before.png", dpi=170)
    plt.close(fig)

    snap = snapshot(df, num_cols)
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    obj = df.select_dtypes(["object", "string"]).columns
    messy = {}
    for c in obj:
        raw_n = df[c].nunique()
        norm_n = df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower().nunique()
        if raw_n > norm_n:
            messy[c] = (raw_n, norm_n)
    lines = [">>> audit summary (before cleaning)"]
    lines += [f"{k:<20}{v:>12,}" for k, v in snap.items()]
    lines += [f"constant columns    : {const_cols if const_cols else 'none'}",
              f"columns with >50% NaN: {list(miss[miss > 50].index) or 'none'}",
              f"text columns with inconsistent spelling/spacing: {len(messy)}"]
    for c, (a, b) in list(messy.items())[:5]:
        lines.append(f"  {c}: {a} distinct values -> {b} after strip/lowercase")
    for l in lines:
        say(l)
    console_png("03_audit_before.png", lines)
    return snap


# ---------------------------------------------------------------- STEP 3
def step3_clean(df, na_drop_pct=90):
    say("\nSTEP 3 - CLEANING")
    log = []
    n0 = len(df)

    # 3a. drop columns that are mostly empty or not useful / sensitive
    miss = df.isna().mean() * 100
    empty_cols = list(miss[miss > na_drop_pct].index)
    sensitive = [c for c in df.columns
                 if any(k in c for k in ("email", "password", "fname", "lname", "street", "image"))]
    drop_cols = sorted(set(empty_cols + sensitive))
    df = df.drop(columns=drop_cols)
    log.append(f"3a dropped {len(drop_cols)} columns (>{na_drop_pct}% empty or PII/irrelevant): {drop_cols}")

    # 3b. duplicates
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log.append(f"3b removed {dups:,} duplicate rows")

    # 3c. parse dates
    for c in pick(df, ["order_date_dateorders", "shipping_date_dateorders"]):
        parsed = pd.to_datetime(df[c], format="%m/%d/%Y %H:%M", errors="coerce")
        if parsed.isna().mean() > 0.5:                      # format guess failed, fall back
            parsed = pd.to_datetime(df[c], errors="coerce")
        bad = int(parsed.isna().sum() - df[c].isna().sum())
        df[c] = parsed
        log.append(f"3c parsed {c} to datetime ({bad:,} unparseable values became NaT)")
    df = df.rename(columns={"order_date_dateorders": "order_date",
                            "shipping_date_dateorders": "shipping_date"})

    # 3d. impossible values
    before = len(df)
    for c, rule in (("days_for_shipping_real", lambda s: s >= 0),
                    ("order_item_quantity", lambda s: s > 0),
                    ("sales", lambda s: s > 0)):
        if c in df.columns:
            ok = rule(pd.to_numeric(df[c], errors="coerce")) | df[c].isna()
            df = df[ok]
    log.append(f"3d removed {before - len(df):,} rows with impossible values (negative days, zero quantity/sales)")

    # 3d2. identifiers (zip codes, ids) are labels, not quantities: keep them as text, never average them
    id_like = [c for c in df.columns if "zipcode" in c or c.endswith("_id") or c == "order_id"]
    for c in id_like:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype("string")
    log.append(f"3d2 treated {len(id_like)} identifier columns (zip codes, ids) as text")

    # 3e. standardise text
    for c in df.select_dtypes(["object", "string"]).columns:
        df[c] = (df[c].astype("string").str.strip()
                 .str.replace(r"\s+", " ", regex=True))
    cats = pick(df, ["shipping_mode", "customer_segment", "order_region", "category_name",
                     "market", "order_status", "delivery_status"])
    for c in cats:
        df[c] = df[c].str.title()
    log.append(f"3e trimmed whitespace and standardised case in {len(cats)} categorical columns")

    # 3f. missing values
    num = df.select_dtypes("number").columns
    effect = []
    for c in num:
        k = int(df[c].isna().sum())
        if k:
            m_before, med = df[c].mean(), df[c].median()
            df[f"{c}_was_missing"] = df[c].isna().astype(int)
            df[c] = df[c].fillna(med)
            effect.append({"column": c, "filled": k, "median_used": round(med, 3),
                           "mean_before": round(m_before, 3), "mean_after": round(df[c].mean(), 3)})
    for c in df.select_dtypes(["object", "string"]).columns:
        k = int(df[c].isna().sum())
        if k:
            df[c] = df[c].fillna("Unknown")
            effect.append({"column": c, "filled": k, "median_used": "Unknown",
                           "mean_before": "", "mean_after": ""})
    eff = pd.DataFrame(effect)
    eff.to_csv(TAB / "imputation_effect.csv", index=False)
    log.append(f"3f imputed {len(eff)} columns (numeric -> median, text -> 'Unknown'), flags added for numeric gaps")

    # 3g. engineered features
    if {"days_for_shipping_real", "days_for_shipment_scheduled"} <= set(df.columns):
        df["delay_days"] = df["days_for_shipping_real"] - df["days_for_shipment_scheduled"]
        df["is_late"] = (df["delay_days"] > 0).astype(int)
    if "order_date" in df:
        df["order_month"] = df["order_date"].dt.month
        df["order_dow"] = df["order_date"].dt.dayofweek
    log.append("3g created delay_days, is_late, order_month, order_dow")

    lines = [f">>> cleaning log ({n0:,} rows in -> {len(df):,} rows out)"] + log
    if len(eff):
        lines += ["", ">>> imputation effect (first 8)", *eff.head(8).to_string(index=False).split("\n")]
    for l in lines:
        say(l)
    console_png("04_cleaning_log.png", lines)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- STEP 4
def step4_outliers(df, cols):
    say("\nSTEP 4 - OUTLIER DETECTION AND TREATMENT")
    rows, before = [], df[cols].copy()
    for c in cols:
        s = df[c]
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_flag = (s < lo) | (s > hi)
        z = (s - s.mean()) / s.std(ddof=0)
        z_flag = z.abs() > 3
        df[f"{c}_is_outlier"] = iqr_flag.astype(int)         # keep a record of what was extreme
        df[c] = s.clip(lo, hi)                               # cap at the IQR fences: keep the row, tame the value
        rows.append({"column": c, "iqr_outliers": int(iqr_flag.sum()),
                     "iqr_pct": round(iqr_flag.mean() * 100, 2),
                     "zscore_outliers": int(z_flag.sum()),
                     "max_before": round(s.max(), 2), "max_after": round(df[c].max(), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "outliers.csv", index=False)
    lines = [">>> outlier summary (IQR 1.5x and |z|>3), then capped at the IQR fences",
             *out.to_string(index=False).split("\n")]
    for l in lines:
        say(l)
    console_png("05_outlier_table.png", lines)

    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 6), squeeze=False)
    for j, c in enumerate(cols):
        sns.boxplot(y=before[c], ax=axes[0][j], color=GREY, fliersize=2)
        axes[0][j].set_title(f"{c}\nbefore", fontsize=9)
        sns.boxplot(y=df[c], ax=axes[1][j], color=BLUE, fliersize=2)
        axes[1][j].set_title("after capping", fontsize=9)
        for r in (0, 1):
            axes[r][j].set_ylabel("")
    fig.suptitle("Outliers before and after treatment", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "06_outliers_before_after.png", dpi=170)
    plt.close(fig)
    return df, before


# ---------------------------------------------------------------- STEP 5
def step5_scale(df, cols):
    say("\nSTEP 5 - TRANSFORMATION AND NORMALISATION")
    raw = df[cols].copy()
    skew_before = raw.skew()
    work = df[cols].copy()
    logged = [c for c in cols if skew_before[c] > 1 and (work[c] >= 0).all()]
    for c in logged:
        work[c] = np.log1p(work[c])
    # NOTE: for modelling, fit scalers on the TRAINING split only to avoid leakage.
    z = pd.DataFrame(StandardScaler().fit_transform(work), columns=[f"{c}_z" for c in cols], index=df.index)
    mm = pd.DataFrame(MinMaxScaler().fit_transform(work), columns=[f"{c}_mm" for c in cols], index=df.index)
    df = pd.concat([df, z, mm], axis=1)

    summ = pd.DataFrame({
        "skew_raw": skew_before.round(2),
        "log1p_applied": [c in logged for c in cols],
        "skew_after_log": work.skew().round(2),
        "z_mean": z.mean().values.round(3),
        "z_std": z.std(ddof=0).values.round(3),
        "mm_min": mm.min().values.round(2),
        "mm_max": mm.max().values.round(2),
    }, index=cols)
    summ.to_csv(TAB / "scaling_summary.csv")
    lines = [">>> skewness, log transform and scaling check", *summ.to_string().split("\n")]
    for l in lines:
        say(l)
    console_png("07_scaling_summary.png", lines)

    c0 = cols[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    sns.histplot(raw[c0], ax=axes[0], color=GREY, bins=40)
    axes[0].set_title(f"{c0}: raw")
    sns.histplot(work[c0], ax=axes[1], color=BLUE, bins=40)
    axes[1].set_title("after log1p" if c0 in logged else "unchanged")
    sns.histplot(z[f"{c0}_z"], ax=axes[2], color=BLUE, bins=40)
    axes[2].set_title("after standardisation (mean 0, sd 1)")
    for a in axes:
        a.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG / "08_distribution_before_after.png", dpi=170)
    plt.close(fig)
    return df


# ---------------------------------------------------------------- STEP 6
def step6_encode(df):
    say("\nSTEP 6 - ENCODING CATEGORICAL VARIABLES")
    cat_cols = pick(df, ["shipping_mode", "customer_segment", "market", "order_region"])
    enc = pd.get_dummies(df[cat_cols], prefix=cat_cols, dtype=int)
    lines = [f">>> one-hot encoding of {cat_cols}",
             f"columns before: {len(cat_cols)}   columns after: {enc.shape[1]}",
             *enc.head(3).iloc[:, :6].to_string().split("\n")]
    for l in lines:
        say(l)
    console_png("09_encoding.png", lines)
    return pd.concat([df, enc], axis=1)


# ---------------------------------------------------------------- STEP 7
def step7_validate(df, before_snap, num_cols):
    say("\nSTEP 7 - VALIDATION AND SUMMARY")
    key_num = pick(df, num_cols)
    checks = {
        "no duplicate rows": int(df.duplicated().sum()) == 0,
        "no missing values in modelling columns": int(df[key_num].isna().sum().sum()) == 0,
        "no negative shipping days": bool((df.get("days_for_shipping_real", pd.Series([0])) >= 0).all()),
        "order_date parsed as datetime": str(df["order_date"].dtype).startswith("datetime") if "order_date" in df else True,
    }
    after = snapshot(df, num_cols)
    summary = pd.DataFrame({"before": before_snap, "after": after})
    summary.to_csv(TAB / "data_quality_summary.csv")
    lines = [">>> data quality summary", *summary.to_string().split("\n"), "", ">>> validation checks"]
    lines += [f"[{'PASS' if v else 'FAIL'}] {k}" for k, v in checks.items()]
    for l in lines:
        say(l)
    console_png("10_validation_summary.png", lines)
    if not all(checks.values()):
        say("WARNING: at least one validation check failed - inspect before using the data.")

    df.to_csv(PROC / "clean_orders.csv", index=False)
    df.sample(min(1000, len(df)), random_state=42).to_csv(SAMPLE / "clean_orders_sample.csv", index=False)
    say(f"\nSaved {PROC / 'clean_orders.csv'} ({len(df):,} rows) and a 1,000-row sample for GitHub.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/DataCoSupplyChainDataset.csv")
    args = ap.parse_args()

    df = step1_load(args.input)
    num_cols = pick(df, ["sales", "order_item_total", "order_item_product_price",
                         "days_for_shipping_real", "days_for_shipment_scheduled", "order_item_quantity"])
    before_snap = step2_audit(df, num_cols)
    df = step3_clean(df)
    out_cols = pick(df, ["sales", "order_item_total", "days_for_shipping_real"])
    df, _ = step4_outliers(df, out_cols)
    scale_cols = pick(df, ["sales", "order_item_total", "order_item_quantity",
                           "days_for_shipment_scheduled", "days_for_shipping_real"])
    df = step5_scale(df, scale_cols)
    df = step6_encode(df)
    step7_validate(df, before_snap, num_cols)

    Path("outputs/pipeline_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say("\nDone. Figures are in outputs/figures/ - insert them into your report.")


if __name__ == "__main__":
    main()
