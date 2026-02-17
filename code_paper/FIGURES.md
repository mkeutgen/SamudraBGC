# GRL BGC Emulator — Figure Generation Guide

**Paper narrative:** ML can replace expensive MOM6-COBALT in ensemble forecasting.

This document specifies the exact layout, data sources, and implementation decisions for
Figures 1–4 of the GRL submission. Figure 5 (ensemble comparison) is handled separately.

---

## Data & Paths

```python
GROUND_TRUTH_PATH = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"

# All ablation predictions live here (test period: 2015-01-01 to 2019-12-31)
OUTPUTS_DIR = "./outputs/"

# Champion model (retrained on train+val before test evaluation)
CHAMPION_PATH = "./outputs/<champion_experiment>_eval/predictions.zarr"

# Test period
TEST_START = "2015-01-01"
TEST_END   = "2019-12-31"

# Validation period (used only for ablation heatmap, not for champion figure)
VAL_START = "2010-01-01"
VAL_END   = "2014-12-31"
```

### BGC variables
| zarr key | long name                  | units    | note                         |
|----------|----------------------------|----------|------------------------------|
| `chl_0`  | Chlorophyll                | mg m⁻³   | log-scale for maps and PDFs  |
| `dic_0`  | Dissolved Inorganic Carbon | µmol kg⁻¹|                              |
| `o2_0`   | Dissolved Oxygen           | µmol kg⁻¹|                              |
| `no3_0`  | Nitrate                    | µmol kg⁻¹|                              |

### Biome boundaries (latitude index, not degrees)
```python
BIOMES = {
    "subtropical": {"lat_min": 0,  "lat_max": 37, "label": "Subtropical Gyre"},
    "jet":         {"lat_min": 37, "lat_max": 43, "label": "Jet Region"},
    "subpolar":    {"lat_min": 43, "lat_max": 90, "label": "Subpolar Gyre"},
}
```
Use `.isel()` or the `_select_region()` helper in `scripts/visualize_comparison.py`.

---

## Ablation Experiment Registry

Ten experiments total (9 ablation + 1 champion). All metrics are computed on the
**validation period (2010–2014)** for the heatmap; the champion is shown on the
**test period (2015–2019)** in Figures 2 and 4.

```python
ABLATION_EXPERIMENTS = {

    # ── Phase 1: Velocity representation ──────────────────────────────────
    "phase1_uv": {
        "label": "u, v",
        "phase": 1,
        "pred_path": "./outputs/jra_fullstate_min_grad05_eval/predictions.zarr",
    },
    "phase1_helmholtz": {
        "label": "ψ, φ",
        "phase": 1,
        "pred_path": "./outputs/jra_helmholtz_min_grad05_eval/predictions.zarr",
        "phase_winner": True,
    },
    "phase1_combined": {
        "label": "u, v, ψ, φ",
        "phase": 1,
        "pred_path": "./outputs/jra_fullstate_helmholtz_min_grad05_eval/predictions.zarr",
    },

    # ── Phase 1.5: Log-transform of BGC state vector ──────────────────────
    "phase15_linear": {
        "label": "Linear",
        "phase": 1.5,
        "pred_path": "./outputs/jra_helmholtz_min_grad05_eval/predictions.zarr",  # same as phase1 winner, no log
    },
    "phase15_log": {
        "label": "Log",
        "phase": 1.5,
        "pred_path": "./outputs/phase15_helmholtz_log_eval/predictions.zarr",
        "phase_winner": True,
    },

    # ── Phase 2: Loss function ────────────────────────────────────────────
    "phase2_mse": {
        "label": "MSE",
        "phase": 2,
        "pred_path": "./outputs/jra_helmholtz_min_mse_eval/predictions.zarr",
    },
    "phase2_mae": {
        "label": "MAE",
        "phase": 2,
        "pred_path": "./outputs/jra_helmholtz_min_grad000_eval/predictions.zarr",
    },
    "phase2_grad010": {
        "label": "MAE + grad (α=0.10)",
        "phase": 2,
        "pred_path": "./outputs/jra_helmholtz_min_grad010_eval/predictions.zarr",
    },
    "phase2_grad025": {
        "label": "MAE + grad (α=0.25)",
        "phase": 2,
        "pred_path": "./outputs/jra_helmholtz_min_grad025_eval/predictions.zarr",
    },
    "phase2_grad050": {
        "label": "MAE + grad (α=0.50)",
        "phase": 2,
        "pred_path": "./outputs/jra_helmholtz_min_grad05_eval/predictions.zarr",
        "phase_winner": True,   # UPDATE once ablation is confirmed
    },

    # ── Phase 3: Architecture ─────────────────────────────────────────────
    "phase3_arch_a": {
        "label": "Arch A",
        "phase": 3,
        "pred_path": "./outputs/<arch_a>_eval/predictions.zarr",
    },
    "phase3_arch_b": {
        "label": "Arch B",
        "phase": 3,
        "pred_path": "./outputs/<arch_b>_eval/predictions.zarr",
    },
    "phase3_arch_c": {
        "label": "Arch C",
        "phase": 3,
        "pred_path": "./outputs/<arch_c>_eval/predictions.zarr",
        "phase_winner": True,   # UPDATE once ablation is confirmed
    },
}

# Champion = phase 3 winner retrained from scratch on train+val
CHAMPION_EXPERIMENT = {
    "label": "Champion",
    "phase": "champion",
    "pred_path": CHAMPION_PATH,
}
```

**Note:** Update `phase_winner` and paths once all ablation runs are complete.
The champion experiment uses the **test period** (2015–2019); all ablation
experiments use the **validation period** (2010–2014).

---

## Metrics Helper

All figures that display quantitative metrics use the following conventions:

```python
BGC_VARS = ["chl_0", "dic_0", "o2_0", "no3_0"]
BGC_LABELS = {"chl_0": "Chl", "dic_0": "DIC", "o2_0": "O₂", "no3_0": "NO₃"}

LEAD_DAYS = [3, 5, 10, 20]  # days used in mean-across-lead-times metrics

def mean_acc_across_leads(pred_ds, gt_ds, varname, lead_days=LEAD_DAYS):
    """
    Compute ACC at each lead day then average arithmetically.
    Climatology = time-mean of ground truth over evaluation period.
    No latitude weighting (regional model).
    """
    ...

def mean_rmse_across_leads(pred_ds, gt_ds, varname, lead_days=LEAD_DAYS):
    """
    Compute RMSE at each lead day then average arithmetically.
    Raw units (not normalised) — normalisation is done in Figure 3 for display.
    """
    ...
```

Normalised RMSE for the heatmap: divide each experiment's RMSE by the
ground-truth temporal standard deviation of that variable over the evaluation
period, so that all four BGC variables live on a comparable [0, ∞) scale.

---

## Figure 1 — System Overview

**Purpose:** Orient the reader. Show what MOM6-COBALT does, what the emulator
replaces, and the overall training/evaluation timeline.

### Panel layout (1 row × 3 columns, or vertical triptych — TBD by journal template)

| Panel | Content | Implementation |
|-------|---------|----------------|
| A | MOM6-COBALT schematic: forcing inputs → ocean state → BGC outputs | Drawn in Inkscape / Illustrator, imported as PDF |
| B | Emulator schematic: same forcing + physical state (T, S, ψ, φ) → BGC at t+1 | Same |
| C | Data timeline: horizontal bar showing train / val / test split | `matplotlib` bar chart |

Panel C specifics:
- x-axis: 1960 → 2019
- Three colored horizontal bars: Train (1960–2009, blue), Val (2010–2014, orange), Test (2015–2019, red)
- No y-axis needed; label bars directly
- Add small annotation: "50 yr training / 5 yr val / 5 yr test"

---

## Figure 2 — Champion Model Performance

**Purpose:** Show the champion model reproduces mean state, seasonal cycle,
interannual variability, and the full marginal distribution of chlorophyll.
Only the champion model appears here — no ablation comparison.

**Data:** Champion predictions vs. ground truth, **test period 2015–2019**.

### Panel A — Chlorophyll snapshot

```
[  ML emulator  ] [  MOM6-COBALT  ]
```

- **Variable:** `chl_0` only
- **Colormap:** `cmocean.algae`, log-normalised (`LogNorm`), shared colorbar
- **Snapshot date:** Choose a spring bloom date (March–May) where the Gulf Stream
  front and subpolar bloom are both visible. Suggested: a day in April 2016 or
  2017 — inspect the ground truth to find the most visually compelling snapshot
  with rich mesoscale structure.
- **No difference map** — the side-by-side is enough for this panel's purpose.
- Thin black line showing the biome boundaries at lat indices 37 and 43.
- Label: "(a) Chlorophyll — ML (left) vs. MOM6-COBALT (right), [date]"

```python
# Suggested implementation
import cmocean
from matplotlib.colors import LogNorm

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
norm = LogNorm(vmin=0.01, vmax=10)  # adjust to data range
cmap = cmocean.cm.algae

for ax, data, title in zip(axes, [ml_chl, gt_chl], ["ML Emulator", "MOM6-COBALT"]):
    im = ax.pcolormesh(lon, lat, data, norm=norm, cmap=cmap)
    ax.axhline(37, color="k", lw=0.8, ls="--")
    ax.axhline(43, color="k", lw=0.8, ls="--")
    ax.set_title(title)

plt.colorbar(im, ax=axes, label="Chl (mg m⁻³)", extend="both")
```

### Panel B — Time series by biome (4 variables × 3 biomes)

Layout: **3 rows (biomes) × 4 columns (BGC variables)**

- Each sub-panel: spatially averaged time series over 2015–2019
- MOM6-COBALT = thick black line
- Champion ML = thick colored line (use a consistent color per variable:
  teal for Chl, blue for DIC, orange for O₂, green for NO₃)
- Weekly smoothing (7-day rolling mean) to reduce noise and improve readability
- Shared y-axis within each column (same variable, different biome)
- x-axis: monthly ticks, year labels only (no month labels — too crowded)
- No legend per panel — add a single legend for the whole figure

```python
fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)

VAR_COLORS = {"chl_0": "teal", "dic_0": "steelblue", "o2_0": "darkorange", "no3_0": "forestgreen"}

for row_idx, (biome_key, biome_info) in enumerate(BIOMES.items()):
    for col_idx, varname in enumerate(BGC_VARS):
        ax = axes[row_idx, col_idx]

        gt_series  = spatial_mean(gt_ds[varname],  biome_info).rolling(time=7).mean()
        ml_series  = spatial_mean(ml_ds[varname],  biome_info).rolling(time=7).mean()

        ax.plot(gt_series.time, gt_series,  color="k",                    lw=1.5, label="MOM6-COBALT")
        ax.plot(ml_series.time, ml_series,  color=VAR_COLORS[varname],    lw=1.5, label="ML Emulator")

        if row_idx == 0:
            ax.set_title(BGC_LABELS[varname])
        if col_idx == 0:
            ax.set_ylabel(biome_info["label"], fontsize=9)
```

### Panel C — Chlorophyll PDFs by biome

- **Variable:** `chl_0` only (log scale)
- x-axis: log scale (`plt.xscale("log")`)
- One set of curves per biome (3 biomes = 3 pairs of curves)
- Ground truth = solid outline (no fill); ML = filled semi-transparent (alpha=0.35)
- Colors: match biome color convention used in Panel B row labels
  (suggest: subtropical=gold, jet=crimson, subpolar=royalblue)
- Use `np.histogram` with log-spaced bins: `np.logspace(-3, 2, 80)`
- Normalise to probability density
- Single legend: biome names + "MOM6-COBALT" / "ML Emulator" entries

```python
fig, ax = plt.subplots(figsize=(5, 4))

BIOME_COLORS = {"subtropical": "goldenrod", "jet": "crimson", "subpolar": "royalblue"}
bins = np.logspace(-3, 2, 80)
bin_centres = np.sqrt(bins[:-1] * bins[1:])   # geometric mid-points for log bins

for biome_key, biome_info in BIOMES.items():
    color = BIOME_COLORS[biome_key]

    gt_vals = select_region(gt_ds["chl_0"], biome_info).values.ravel()
    ml_vals = select_region(ml_ds["chl_0"], biome_info).values.ravel()

    # Remove NaN / non-positive (land mask)
    gt_vals = gt_vals[np.isfinite(gt_vals) & (gt_vals > 0)]
    ml_vals = ml_vals[np.isfinite(ml_vals) & (ml_vals > 0)]

    gt_hist, _ = np.histogram(gt_vals, bins=bins, density=True)
    ml_hist, _ = np.histogram(ml_vals, bins=bins, density=True)

    ax.plot(bin_centres, gt_hist, color=color, lw=2, ls="-")
    ax.fill_between(bin_centres, ml_hist, alpha=0.35, color=color)

ax.set_xscale("log")
ax.set_xlabel("Chlorophyll (mg m⁻³)")
ax.set_ylabel("Probability Density")
```

### Figure 2 — Assembly

```python
# Suggested figure assembly using GridSpec
fig = plt.figure(figsize=(16, 14))
gs  = fig.add_gridspec(3, 4,
                        height_ratios=[1.2, 1, 1],
                        hspace=0.35, wspace=0.3)

# Panel A: spans top row, columns 0-2 (leave col 3 for Panel C)
ax_snap_ml  = fig.add_subplot(gs[0, 0:2])   # ML snapshot
ax_snap_gt  = fig.add_subplot(gs[0, 2:4])   # GT snapshot — adjust to taste

# Panel B: rows 1-2, all 4 columns
axes_ts = [[fig.add_subplot(gs[i+1, j]) for j in range(4)] for i in range(2)]
# Note: only 2 rows left; if you want 3 biome rows you need a 4-row GridSpec

# Panel C: insert as inset or as a 5th column — discuss with co-authors
```

> **Layout note:** Fitting 3 biome rows + the snapshot row into one GRL figure
> (typically ≤ 20 cm tall) is tight. Consider putting Panel C in a separate
> inset within the snapshot row, or merging subtropical+jet into one "gyre"
> panel if space is critical.

---

## Figure 3 — Ablation Scorecard Heatmap

**Purpose:** Show all 10 experiments (9 ablation + 1 champion) across all 4 BGC
variables on both ACC and RMSE. Justify every design choice in one compact figure.

**Data:** Ablation experiments on **validation period (2010–2014)**;
champion on **test period (2015–2019)**.

**Metrics:**
- ACC: mean of ACC at lead days 3, 5, 10, 20 (arithmetic mean)
- RMSE: mean of normalised RMSE at lead days 3, 5, 10, 20
  - Normalise per variable: divide by gt temporal std over the evaluation period
  - This makes DIC, Chl, O₂, NO₃ comparable on a single colour scale

### Layout: two side-by-side heatmaps

```
┌─────────────────────────────────────────────────────┐
│               ACC (higher = better)                 │
│  [10-row × 5-column heatmap]   Mean col →           │
├─────────────────────────────────────────────────────┤
│             Norm. RMSE (lower = better)             │
│  [10-row × 5-column heatmap]   Mean col →           │
└─────────────────────────────────────────────────────┘
```

Or equivalently, two panels side by side (ACC left, RMSE right) with shared row labels.

### Row order (top to bottom)

```
Phase 1   │ u, v
          │ ψ, φ           ← phase winner (bold border)
          │ u, v, ψ, φ
──────────┤
Phase 1.5 │ Linear
          │ Log            ← phase winner (bold border)
──────────┤
Phase 2   │ MSE
          │ MAE
          │ MAE+grad α=0.10
          │ MAE+grad α=0.25
          │ MAE+grad α=0.50  ← phase winner (bold border)
──────────┤
Phase 3   │ Arch A
          │ Arch B
          │ Arch C         ← phase winner (bold border)
══════════╪═══════════════
Champion  │ ★              ← visually distinct row
```

### Column order

```
Chl  │ DIC  │ O₂  │ NO₃  ║ Mean
```

### Colour scales

- **ACC heatmap:** sequential (`cmocean.cm.matter` or `RdYlGn`), range [0.4, 1.0].
  Do not use a diverging scale centred at 0.5 — the interesting variation
  is in the upper range [0.6, 1.0]. Annotate each cell with the value (1 decimal).
- **RMSE heatmap:** sequential reversed (`RdYlGn_r` or `cmocean.cm.thermal`),
  range [0, 1.5]. Lower is better — make sure the colorbar label says so.
  Annotate each cell with the value (2 decimals).

### Visual emphasis

```python
# Highlight phase winners with a bold black border
# Highlight champion row with light grey background

def add_cell_border(ax, row, col, n_cols, color="black", lw=2.5):
    """Draw a rectangle around cell (row, col) in a heatmap axes."""
    ax.add_patch(plt.Rectangle(
        (col - 0.5, row - 0.5), 1, 1,
        fill=False, edgecolor=color, linewidth=lw, clip_on=False
    ))

# Phase separator lines (horizontal)
for ax in [ax_acc, ax_rmse]:
    for separator_row in [2.5, 4.5, 9.5]:   # after row indices 2, 4, 9
        ax.axhline(separator_row, color="white", lw=2.5)

# Champion row: different background
# Easiest approach: after imshow, overwrite that row's colour manually
```

### Implementation sketch

```python
import seaborn as sns

# Build data arrays: shape (10, 5) for both ACC and RMSE
# Rows follow the order above; last column = mean across BGC vars

row_labels = [
    "u, v", "ψ, φ", "u, v, ψ, φ",          # Phase 1
    "Linear", "Log",                           # Phase 1.5
    "MSE", "MAE", "+grad 0.10",               # Phase 2
    "+grad 0.25", "+grad 0.50",
    "Arch A", "Arch B", "Arch C",             # Phase 3
    "★ Champion",                              # Champion
]

col_labels = ["Chl", "DIC", "O₂", "NO₃", "Mean"]

phase_labels = ["Phase 1\nVelocity", "Phase 1.5\nTransform",
                "Phase 2\nLoss", "Phase 3\nArchitecture", ""]

fig, (ax_acc, ax_rmse) = plt.subplots(1, 2, figsize=(12, 7),
                                       gridspec_kw={"wspace": 0.05})

sns.heatmap(acc_matrix,  ax=ax_acc,  annot=True, fmt=".2f",
            cmap="RdYlGn",   vmin=0.4, vmax=1.0,
            xticklabels=col_labels, yticklabels=row_labels,
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "ACC ↑", "shrink": 0.6})

sns.heatmap(rmse_matrix, ax=ax_rmse, annot=True, fmt=".2f",
            cmap="RdYlGn_r", vmin=0.0, vmax=1.5,
            xticklabels=col_labels, yticklabels=False,
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Norm. RMSE ↓", "shrink": 0.6})

# Add phase group annotations on the left y-axis
# Use ax_acc.text() or a secondary y-axis for phase labels

ax_acc.set_title("ACC (mean over lead days 3–20)")
ax_rmse.set_title("Normalised RMSE (mean over lead days 3–20)")
```

### Phase group annotations

Add text labels on the far left axis (outside the heatmap) indicating phase groups.
Use `ax.annotate()` or `ax.text()` with `transform=ax.transData`, positioned at the
vertical midpoint of each group of rows:

```python
phase_label_positions = {
    "Phase 1\nVelocity":    1.0,    # midpoint of rows 0-2
    "Phase 1.5\nTransform": 3.5,    # midpoint of rows 3-4
    "Phase 2\nLoss":        7.0,    # midpoint of rows 5-9
    "Phase 3\nArchitecture":11.0,   # midpoint of rows 10-12
}
for label, y in phase_label_positions.items():
    ax_acc.text(-1.2, y, label, ha="right", va="center",
                fontsize=8, rotation=0, color="dimgray")
```

---

## Figure 4 — Physical–BGC Coupling (gradient fidelity)

**Purpose:** Demonstrate that the champion emulator reproduces spatial gradient
structure — i.e., sharp fronts are preserved and not spuriously blurred or
displaced. This validates that the gradient penalty loss (MAE+grad) works.

**This is not a scatter of BGC vs. physical drivers.** Instead it directly
shows spatial gradient distributions, which are the hardest thing for emulators
to get right and the most scientifically relevant for transport.

**Data:** Champion vs. ground truth, test period 2015–2019.

### Panel layout (2 rows × 2 columns = 4 panels, one per BGC variable)

Each panel: **two overlapping gradient-magnitude PDFs** (champion ML vs. MOM6-COBALT),
split by biome (3 curves per PDF set = 6 curves per panel).

```
┌───────────────────┬───────────────────┐
│  (a) Chlorophyll  │  (b) DIC          │
│   |∇Chl| PDF      │   |∇DIC| PDF      │
├───────────────────┼───────────────────┤
│  (c) O₂           │  (d) NO₃          │
│   |∇O₂| PDF       │   |∇NO₃| PDF      │
└───────────────────┴───────────────────┘
```

### Within each panel

- x-axis: gradient magnitude `|∇var|`, linear scale (not log)
  - Truncate at the 99th percentile of the ground truth to avoid long tails
- y-axis: probability density
- Ground truth = solid lines, 3 biome colours (subtropical=gold, jet=crimson, subpolar=royalblue)
- Champion ML = dashed lines, same 3 biome colours
- No fill — lines only (6 lines per panel is already busy enough)
- Single shared legend across all 4 panels (top of figure or figure caption)

### Gradient computation

```python
def compute_gradient_magnitude(field_2d: np.ndarray) -> np.ndarray:
    """
    Compute |∇f| = sqrt( (∂f/∂x)² + (∂f/∂y)² )
    Uses central differences. Returns same shape as input, with NaN at boundaries.
    """
    dy = np.gradient(field_2d, axis=0)
    dx = np.gradient(field_2d, axis=1)
    return np.sqrt(dx**2 + dy**2)

# For chlorophyll: compute gradient in log space (matches loss function)
log_chl_gt = np.log(gt_chl + 1e-6)
log_chl_ml = np.log(ml_chl + 1e-6)
grad_gt     = compute_gradient_magnitude(log_chl_gt)
grad_ml     = compute_gradient_magnitude(log_chl_ml)
```

> **Important:** For `chl_0`, compute gradients **in log space** (consistent
> with the log-transform applied during training). For DIC, O₂, NO₃, compute
> in linear space.

### Strong vs. weak gradient diagnostics (optional inset or supplementary)

If space allows, add a small 2×2 table inset in panel (a) showing:

| | Strong fronts (top 10% GT gradient) | Weak regions (bottom 50% GT gradient) |
|---|---|---|
| ML gradient (mean) | → should match GT | → should match GT |

This directly tests the two failure modes described in the methods:
(1) spurious gradients in smooth regions, (2) damped gradients at true fronts.

### Implementation sketch

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)
axes = axes.ravel()

BIOME_COLORS = {"subtropical": "goldenrod", "jet": "crimson", "subpolar": "royalblue"}
N_BINS = 100

for ax_idx, varname in enumerate(BGC_VARS):
    ax = axes[ax_idx]

    for biome_key, biome_info in BIOMES.items():
        color = BIOME_COLORS[biome_key]

        gt_field = select_region(gt_ds[varname],  biome_info)
        ml_field = select_region(ml_ds[varname],  biome_info)

        # Log space for chl
        if varname == "chl_0":
            gt_field = np.log(gt_field + 1e-6)
            ml_field = np.log(ml_field + 1e-6)

        # Compute gradient magnitude over all timesteps, flatten
        gt_grads = np.concatenate([
            compute_gradient_magnitude(gt_field.isel(time=t).values).ravel()
            for t in range(0, len(gt_field.time), 30)   # subsample: every 30 days
        ])
        ml_grads = np.concatenate([
            compute_gradient_magnitude(ml_field.isel(time=t).values).ravel()
            for t in range(0, len(ml_field.time), 30)
        ])

        # Remove NaN and cap at 99th percentile
        p99 = np.nanpercentile(gt_grads, 99)
        gt_grads = gt_grads[np.isfinite(gt_grads) & (gt_grads <= p99)]
        ml_grads = ml_grads[np.isfinite(ml_grads) & (ml_grads <= p99)]

        bins = np.linspace(0, p99, N_BINS)
        gt_hist, _ = np.histogram(gt_grads, bins=bins, density=True)
        ml_hist, _ = np.histogram(ml_grads, bins=bins, density=True)
        bin_centres = 0.5 * (bins[:-1] + bins[1:])

        ax.plot(bin_centres, gt_hist, color=color, lw=2.0, ls="-")
        ax.plot(bin_centres, ml_hist, color=color, lw=2.0, ls="--")

    ax.set_title(BGC_LABELS[varname])
    ax.set_xlabel(f"|∇{BGC_LABELS[varname]}|")
    ax.set_ylabel("Probability Density" if ax_idx % 2 == 0 else "")

# Single legend
legend_elements = (
    [Line2D([0], [0], color=c, lw=2, label=BIOMES[b]["label"])
     for b, c in BIOME_COLORS.items()] +
    [Line2D([0], [0], color="k", lw=2, ls="-",  label="MOM6-COBALT"),
     Line2D([0], [0], color="k", lw=2, ls="--", label="ML Emulator")]
)
fig.legend(handles=legend_elements, loc="upper center",
           ncol=5, bbox_to_anchor=(0.5, 1.02))
```

---

## Style Conventions (apply across all figures)

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean

# GRL-compatible style
mpl.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# Save format: PDF for vector figures (maps), PNG for raster-heavy figures
# GRL column widths: single column = 8.4 cm, double column = 17.4 cm
SINGLE_COL_INCHES = 8.4 / 2.54
DOUBLE_COL_INCHES = 17.4 / 2.54
```

**Color palette summary:**
- MOM6-COBALT ground truth: always `"k"` (black)
- Champion ML emulator: `"steelblue"` (or consistent accent colour)
- Biomes: subtropical = `"goldenrod"`, jet = `"crimson"`, subpolar = `"royalblue"`
- BGC variables: Chl = `"teal"`, DIC = `"steelblue"`, O₂ = `"darkorange"`, NO₃ = `"forestgreen"`
- Ablation phase winners: bold black cell border in heatmap

**Land mask:** The MOM6 domain is an ocean-only regional model. Use the wet mask
from the ground truth dataset (any variable's NaN pattern) consistently across
all figures. Do not plot land as white — use `"lightgray"` for any background.

---

## Output Files

```
figures/
├── fig01_system_overview.pdf
├── fig02_champion_performance.pdf
├── fig03_ablation_heatmap.pdf
├── fig04_gradient_pdfs.pdf
└── supplementary/
    └── figS01_all_vars_snapshots.pdf
```

Save all figures with `fig.savefig(path, dpi=300, bbox_inches="tight")`.

---

## Open Items / Decisions Needed

- [ ] Confirm exact experiment paths for Phase 3 architecture variants
- [ ] Confirm champion model path (after retraining on train+val)
- [ ] Choose snapshot date for Figure 2 Panel A (inspect ground truth spring blooms)
- [ ] Decide Figure 2 GridSpec layout — 3 biome rows may require a 4-row GridSpec
      or a 2-row version merging subtropical + jet
- [ ] Confirm whether to show Chl PDF only (Panel C) or add DIC as a second panel
- [ ] Confirm normalisation convention for RMSE in Figure 3 (divide by gt std
      over test period or val period? Use the evaluation period of each experiment)
