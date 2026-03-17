import matplotlib as mpl
mpl.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
from pathlib import Path

rng = np.random.default_rng(42)

# ── time axis ──────────────────────────────────────────────────────────────
n_steps = 61                          # monthly steps over 5 years
t = np.linspace(0, 5, n_steps)        # years

# ── shared seasonal signal ─────────────────────────────────────────────────
seasonal = 8 * np.sin(2 * np.pi * t)  # ±8 mmol seasonal cycle
trend    = 3 * t                       # ~3 mmol/yr uptake trend

# ═══════════════════════════════════════════════════════════════════════════
# LEFT: deterministic emulator — 10 members, tight spread
# ═══════════════════════════════════════════════════════════════════════════
n_det  = 10
dic_0  = 2160.0                        # initial DIC (mmol/m³)

# Each member = shared signal + tiny iid noise (±2 mmol, constant in time)
det_noise = rng.normal(0, 2, size=(n_det, n_steps))
det_members = dic_0 + trend + seasonal + det_noise

# ═══════════════════════════════════════════════════════════════════════════
# RIGHT: probabilistic emulator — 50 members, 30 days, fast divergence
# ═══════════════════════════════════════════════════════════════════════════
n_prob = 50
n_steps_prob = 120                        # sub-daily steps over 30 days
t_prob = np.linspace(0, 30, n_steps_prob) # days

# Shared signal for 30 days: slight trend + small daily variability
seasonal_prob = 2 * np.sin(2 * np.pi * t_prob / 30)  # one month oscillation
trend_prob    = 0.5 * t_prob                           # slight upward trend

# Spread grows quickly — divergence starts around day 5
spread_scale_prob = 0.8 * np.exp(0.18 * np.clip(t_prob - 5, 0, None))

# Correlated perturbations: random walk
prob_noise = rng.normal(0, 1, size=(n_prob, n_steps_prob))
prob_noise_cumulative = np.cumsum(prob_noise * spread_scale_prob[None, :], axis=1)
prob_noise_cumulative -= prob_noise_cumulative[:, :1]   # anchor at 0

prob_members = dic_0 + trend_prob + seasonal_prob + prob_noise_cumulative

# ── figure layout ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4.5),
    sharey=False,
    constrained_layout=True,
)

blue_mid  = "#378ADD"
blue_dark = "#185FA5"
coral_mid = "#D85A30"
coral_drk = "#993C1D"
amber     = "#EF9F27"

# ── helper: shared axis styling ────────────────────────────────────────────
def style_ax(ax, title, badge_text, badge_color, badge_text_color):
    ax.set_xlim(0, 5)
    ax.set_xlabel("Time (years)", fontsize=10, color="#5F5E5A")
    ax.set_ylabel("DIC  (µmol kg⁻¹)", fontsize=10, color="#5F5E5A")
    ax.tick_params(labelsize=9, colors="#888780")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#B4B2A9")
    ax.grid(axis="y", linewidth=0.4, linestyle="--", color="#D3D1C7", alpha=0.7)
    ax.set_facecolor("#FAFAF8")
    ax.set_title(title, fontsize=11, fontweight="normal", pad=10, color="#2C2C2A")
    # Badge
    ax.text(
        0.03, 0.97, badge_text,
        transform=ax.transAxes,
        fontsize=8.5, color=badge_text_color,
        va="top", ha="left",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=badge_color,
            edgecolor="none",
        ),
    )

# ── LEFT panel ─────────────────────────────────────────────────────────────
ax = axes[0]
for i, member in enumerate(det_members):
    ax.plot(t, member, color=blue_mid, lw=0.9, alpha=0.65)

style_ax(
    ax,
    title="Deterministic emulator",
    badge_text="10 members · 5 yr",
    badge_color="#E6F1FB",
    badge_text_color="#185FA5",
)
ax.set_xlim(-0.1, 5.1)

# ── RIGHT panel ────────────────────────────────────────────────────────────
ax = axes[1]

# Shade the ensemble envelope
env_lo = prob_members.min(axis=0)
env_hi = prob_members.max(axis=0)
ax.fill_between(t_prob, env_lo, env_hi, color=coral_mid, alpha=0.12, lw=0)

for member in prob_members:
    ax.plot(t_prob, member, color=coral_mid, lw=0.55, alpha=0.45)

# Style right panel (days, not years)
ax.set_xlabel("Time (days)", fontsize=10, color="#5F5E5A")
ax.set_ylabel("DIC  (µmol kg⁻¹)", fontsize=10, color="#5F5E5A")
ax.tick_params(labelsize=9, colors="#888780")
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
for spine in ax.spines.values():
    spine.set_linewidth(0.6)
    spine.set_color("#B4B2A9")
ax.grid(axis="y", linewidth=0.4, linestyle="--", color="#D3D1C7", alpha=0.7)
ax.set_facecolor("#FAFAF8")
ax.set_title("Probabilistic emulator", fontsize=11, fontweight="normal",
             pad=10, color="#2C2C2A")
ax.text(
    0.03, 0.97, "50 members · 30 days",
    transform=ax.transAxes, fontsize=8.5, color="#993C1D",
    va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FAECE7", edgecolor="none"),
)
ax.set_xlim(-0.5, 31)


out = Path(__file__).resolve().parent / "figures" / "fig00_emulator_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")