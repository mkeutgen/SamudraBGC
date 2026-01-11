#!/usr/bin/env python3
"""
Parse a 'GLOBAL METRICS SUMMARY' text dump and rank models by averaged metrics.

Usage examples:
  python rank_models.py --path /mnt/data/global_metrics.txt
  python rank_models.py --path global_metrics.txt --metric NRMSE
  python rank_models.py --path global_metrics.txt --group prefix --metric MAE
  python rank_models.py --path global_metrics.txt --composite --metrics R2 Corr NRMSE MAE

Notes:
- "Bias" is signed; this script ranks by |Bias| when metric is Bias (lower is better).
- R² / Corr: higher is better. RMSE/MAE/NRMSE/|Bias|: lower is better.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd


MODEL_HEADER_RE = re.compile(r"^\s*([A-Za-z0-9][^:]{2,}):\s*$")
VAR_HEADER_RE = re.compile(r"^\s*([A-Za-z0-9]+_[A-Za-z0-9]+)\s*:\s*$")
METRIC_LINE_RE = re.compile(r"^\s*([^\:]+?)\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")

# Normalize metric names found in file to stable keys
def norm_metric_name(raw: str) -> str:
    s = raw.strip()
    # Handle unicode variants (R²)
    s = s.replace("R²", "R2").replace("R^2", "R2").replace("R 2", "R2")
    s = s.replace("Correlation", "Corr")
    return s

# Metric direction: True => higher is better; False => lower is better
HIGHER_BETTER = {
    "R2": True,
    "Corr": True,
    "RMSE": False,
    "MAE": False,
    "Bias": False,   # treated as |Bias| for ranking/aggregation unless you change below
    "NRMSE": False,
}

DEFAULT_METRICS_FOR_COMPOSITE = ["R2", "Corr", "NRMSE", "MAE"]


def parse_metrics_text(path: str) -> pd.DataFrame:
    """
    Returns a tidy dataframe with columns:
      model, variable, metric, value
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rows: List[Dict[str, object]] = []

    current_model: Optional[str] = None
    current_var: Optional[str] = None

    for ln in lines:
        line = ln.rstrip("\n")

        # model header
        m = MODEL_HEADER_RE.match(line)
        if m:
            current_model = m.group(1).strip()
            current_var = None
            continue

        # variable header
        v = VAR_HEADER_RE.match(line)
        if v and current_model is not None:
            current_var = v.group(1).strip()
            continue

        # metric line
        k = METRIC_LINE_RE.match(line)
        if k and current_model is not None and current_var is not None:
            metric_raw, val_raw = k.group(1), k.group(2)
            metric = norm_metric_name(metric_raw)
            try:
                val = float(val_raw)
            except ValueError:
                continue

            rows.append(
                {"model": current_model, "variable": current_var, "metric": metric, "value": val}
            )

    if not rows:
        raise RuntimeError(f"No metrics parsed from {path}. Check file format / encoding.")

    df = pd.DataFrame(rows)

    # Treat Bias as absolute magnitude for aggregation/ranking (common in model eval)
    df.loc[df["metric"] == "Bias", "value"] = df.loc[df["metric"] == "Bias", "value"].abs()

    return df


def add_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == "none":
        df = df.copy()
        df["group"] = "ALL"
        return df
    if group == "prefix":
        df = df.copy()
        df["group"] = df["variable"].str.split("_", n=1).str[0]
        return df
    raise ValueError(f"Unknown group mode: {group}")


def averaged_table(df: pd.DataFrame, metric: str, group: str) -> pd.DataFrame:
    """
    Returns model x group table of averaged metric over variables.
    """
    metric = norm_metric_name(metric)
    dfg = add_group(df, group)
    sub = dfg[dfg["metric"] == metric].copy()
    if sub.empty:
        avail = sorted(df["metric"].unique().tolist())
        raise ValueError(f"Metric '{metric}' not found. Available: {avail}")

    out = (
        sub.groupby(["model", "group"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": f"mean_{metric}"})
    )
    return out


def rank_models(df_mean: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Takes output of averaged_table() and ranks within each group.
    """
    metric = norm_metric_name(metric)
    col = f"mean_{metric}"
    if col not in df_mean.columns:
        raise ValueError(f"Expected column {col} in df_mean.")

    higher_better = HIGHER_BETTER.get(metric)
    if higher_better is None:
        raise ValueError(f"Don't know direction for metric '{metric}'. Add it to HIGHER_BETTER.")

    ranked = df_mean.copy()
    ranked["rank"] = ranked.groupby("group")[col].rank(
        ascending=not higher_better, method="min"
    ).astype(int)

    ranked = ranked.sort_values(["group", "rank", col], ascending=[True, True, higher_better])
    return ranked


def composite_rank(
    df: pd.DataFrame,
    metrics: List[str],
    group: str,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Composite score per model (and group) using z-scored metrics averaged with optional weights.
    Higher composite is better.
    """
    metrics = [norm_metric_name(m) for m in metrics]
    weights = weights or {m: 1.0 for m in metrics}

    dfg = add_group(df, group)

    # mean over variables for each metric first
    mean_parts = []
    for m in metrics:
        sub = dfg[dfg["metric"] == m].copy()
        if sub.empty:
            continue
        mean_m = sub.groupby(["model", "group"], as_index=False)["value"].mean()
        mean_m = mean_m.rename(columns={"value": f"mean_{m}"})
        mean_parts.append(mean_m)

    if not mean_parts:
        raise RuntimeError("No metrics available for composite ranking.")

    # merge
    merged = mean_parts[0]
    for part in mean_parts[1:]:
        merged = merged.merge(part, on=["model", "group"], how="outer")

    # z-score each metric within each group, flip sign if lower is better
    for m in metrics:
        col = f"mean_{m}"
        if col not in merged.columns:
            continue
        hb = HIGHER_BETTER.get(m)
        if hb is None:
            raise ValueError(f"Unknown direction for metric '{m}'. Add to HIGHER_BETTER.")
        # z within group
        merged[col] = merged[col].astype(float)
        mu = merged.groupby("group")[col].transform("mean")
        sig = merged.groupby("group")[col].transform("std").replace(0.0, float("nan"))
        z = (merged[col] - mu) / sig
        if not hb:
            z = -z
        merged[f"z_{m}"] = z

    # weighted average of z-scores
    zcols = [f"z_{m}" for m in metrics if f"z_{m}" in merged.columns]
    if not zcols:
        raise RuntimeError("No z-scored metrics computed (std may be zero or metrics missing).")

    def row_score(row) -> float:
        num = 0.0
        den = 0.0
        for m in metrics:
            zc = f"z_{m}"
            if zc not in row or pd.isna(row[zc]):
                continue
            w = float(weights.get(m, 1.0))
            num += w * float(row[zc])
            den += w
        return num / den if den > 0 else float("nan")

    merged["composite"] = merged.apply(row_score, axis=1)

    # rank: higher composite is better
    merged["rank"] = merged.groupby("group")["composite"].rank(ascending=False, method="min").astype(int)
    merged = merged.sort_values(["group", "rank", "composite"], ascending=[True, True, False])
    return merged[["group", "model", "composite", "rank"] + [f"mean_{m}" for m in metrics if f"mean_{m}" in merged.columns]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to global_metrics.txt")
    ap.add_argument("--metric", default="NRMSE", help="Metric to average+rank (e.g., NRMSE, MAE, R2, Corr, RMSE, Bias)")
    ap.add_argument("--group", choices=["none", "prefix"], default="none",
                    help="Aggregate variables all together (none) or by variable prefix (prefix: temp/salt/dic/o2/...)")
    ap.add_argument("--composite", action="store_true", help="Compute composite ranking instead of single-metric ranking")
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="Metrics to include in composite (default: R2 Corr NRMSE MAE). Use names as in file (R² ok).")
    args = ap.parse_args()

    df = parse_metrics_text(args.path)

    if args.composite:
        metrics = args.metrics if args.metrics else DEFAULT_METRICS_FOR_COMPOSITE
        out = composite_rank(df, metrics=metrics, group=args.group)
        print("\n=== COMPOSITE RANKING (higher is better) ===")
        print(out.to_string(index=False))
    else:
        mean_df = averaged_table(df, metric=args.metric, group=args.group)
        ranked = rank_models(mean_df, metric=args.metric)
        print(f"\n=== RANKING by mean {norm_metric_name(args.metric)} over variables (group={args.group}) ===")
        print(ranked.to_string(index=False))


if __name__ == "__main__":
    main()
