import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr

# ---------------------------
# 1) Load data
# ---------------------------
inj_path = Path("data_outputs/injected_transits_output6.csv")
noninj_path = Path("data_outputs/noninjected_transits_output6.csv")

inj = pd.read_csv(inj_path)
non = pd.read_csv(noninj_path)

# Mark ground truth
inj["is_injected"] = True
non["is_injected"] = False

# Concatenate for global metrics
df = pd.concat([inj, non], ignore_index=True)

# Sanity check: ensure required columns exist
required_cols = ["r_s","r_p","a","P_days","inc","period","median","mean","max_depth","oot_variability","snr"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# ---------------------------
# 2) Helper: precision/recall over thresholds
# ---------------------------
def pr_over_thresholds(df, thresholds):
    """
    Ground truth positives = injected rows.
    Predicted positive = snr >= threshold.
    """
    y_true = df["is_injected"].values.astype(bool)
    snr = df["snr"].values

    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = snr >= t
        TP = np.sum(y_pred & y_true)
        FP = np.sum(y_pred & ~y_true)
        FN = np.sum((~y_pred) & y_true)

        precision = safe_div(TP, TP + FP)
        recall = safe_div(TP, TP + FN)
        f1 = safe_div(2 * precision * recall, precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return np.array(precisions, dtype=float), np.array(recalls, dtype=float), np.array(f1s, dtype=float)

def safe_div(a, b):
    return a / b if b > 0 else 0.0

# Reasonable threshold grid based on your data range
snr_min = np.nanmin(df["snr"].values)
snr_max = np.nanmax(df["snr"].values)
# pad a little so the ends are covered
snr_all = df["snr"].to_numpy()
pos_snr = snr_all[snr_all > 0]
lo = max(1.0, np.percentile(pos_snr, 1))   # or 0.5 if you need lower
hi = max(lo * 1.01, np.percentile(pos_snr, 99))  # avoid equal bounds
thr_grid = np.logspace(np.log10(lo), np.log10(hi), 200)

P, R, F1 = pr_over_thresholds(df, thr_grid)
best_idx = int(np.argmax(F1))
best_thr = float(thr_grid[best_idx])
print(f"[INFO] F1-optimal detection threshold: {best_thr:.4f} (P={P[best_idx]:.3f}, R={R[best_idx]:.3f}, F1={F1[best_idx]:.3f})")

# ---------------------------
# 3) Plot: Precision & Recall vs Threshold
#    (colors requested: precision=blue, recall=orange)
# ---------------------------
fig1, ax1 = plt.subplots(figsize=(6,4.2), dpi=150)
ax1.plot(thr_grid, P, label="Precision", linewidth=2.0, color="tab:blue")
ax1.plot(thr_grid, R, label="Recall", linewidth=2.0, color="tab:orange")
ax1.axvline(best_thr, linestyle="--", alpha=0.6, label=f"Best F1 Threshold: {best_thr:.3f}", color="red")
ax1.set_xlabel("Detection Threshold (SNR)")
ax1.set_ylabel("Score")
ax1.set_title("Precision & Recall vs Detection Threshold")
ax1.grid(True, alpha=0.25)
ax1.set_xscale("log")
ax1.legend()
fig1.tight_layout()
fig1.savefig("data_outputs/run6poster/precision_recall_vs_threshold.png", dpi=300)

# ---------------------------
# 4) Plot: Precision–Recall curve (sweep thresholds high→low)
# ---------------------------
# Sort by recall ascending for a nicer PR trace
mask = ~np.isnan(P) & ~np.isnan(R)
P_valid, R_valid = P[mask], R[mask]
order = np.argsort(R_valid)
fig2, ax2 = plt.subplots(figsize=(6,4.2), dpi=150)
ax2.plot(R_valid[order], P_valid[order], linewidth=2.0)
# mark the best-F1 point
ax2.scatter(R[best_idx], P[best_idx], s=30, color="red", label=f"Best F1 Point: {F1[best_idx]:.3f}")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision–Recall Curve")
ax2.grid(True, alpha=0.25)
ax2.legend()
fig2.tight_layout()
fig2.savefig("data_outputs/run6poster/precision_recall_curve.png", dpi=300)

# ---------------------------
# 5) Plot: SNR distribution (injected=blue, non-injected=gray)
# ---------------------------
snr_pos = inj["snr"].to_numpy()
snr_neg = non["snr"].to_numpy()
pos = snr_pos[snr_pos > 0]
neg = snr_neg[snr_neg > 0]
bins = np.logspace(np.log10(min(pos.min(), neg.min())),
                   np.log10(max(pos.max(), neg.max())), 50)

fig3, ax3 = plt.subplots(figsize=(6,4.2), dpi=150)
ax3.hist(pos, bins=bins, alpha=0.8, label="Injected", color="blue", density=True)
ax3.hist(neg, bins=bins, alpha=0.8, label="Non-injected", color="orange", density=True)
ax3.axvline(best_thr, linestyle="--", alpha=0.6, label=f"Best F1 Threshold: {best_thr:.3f}", color="red")
ax3.set_xlabel("SNR")
ax3.set_ylabel("Density")
ax3.set_title("SNR Distribution")
ax3.set_xscale("log"); ax3.set_yscale("log")
ax3.grid(True, alpha=0.25)
ax3.legend()
fig3.tight_layout()
fig3.savefig("data_outputs/run6poster/snr_histograms.png", dpi=300)

# ---------------------------
# 6) 2D Recovery fraction heatmap over radius–period
#     - Use injected set (ground truth exists)
#     - Detection defined by SNR >= best_thr (F1-optimal)
# ---------------------------
inj_det = inj.assign(detected = inj["snr"] >= best_thr)

# Choose bin edges from data quantiles to avoid empty edges
def nice_bins(series, nbins=20):
    # If data are concentrated, quantile bins give better occupancy
    q = np.linspace(0, 1, nbins+1)
    edges = np.unique(np.quantile(series.values, q))
    # fallback to linspace if too many duplicates
    if len(edges) < nbins/2:
        edges = np.linspace(series.min(), series.max(), nbins+1)
    return edges

r_bins = nice_bins(inj_det["r_p"], nbins=18)
p_bins = nice_bins(inj_det["P_days"], nbins=18)

# Count totals and detections per bin
H_total, _, _ = np.histogram2d(inj_det["r_p"], inj_det["P_days"], bins=[r_bins, p_bins])
H_det,   _, _ = np.histogram2d(inj_det.loc[inj_det["detected"], "r_p"],
                               inj_det.loc[inj_det["detected"], "P_days"],
                               bins=[r_bins, p_bins])

with np.errstate(invalid="ignore", divide="ignore"):
    frac = H_det / H_total
frac = np.where(H_total > 0, frac, np.nan)

fig4, ax4 = plt.subplots(figsize=(6.3,5.6), dpi=150)
# Show with pcolormesh to respect bin edges
R, Pp = np.meshgrid(r_bins, p_bins, indexing="ij")
im = ax4.pcolormesh(R, Pp, frac, shading="auto")
cbar = fig4.colorbar(im, ax=ax4, label="Recovery Fraction")
ax4.set_xlabel("Planet radius ($r_E$)")
ax4.set_ylabel("Orbital period (days)")
ax4.set_title("Recovery Fraction (Injected Only)")
ax4.set_xscale("log")
fig4.tight_layout()
fig4.savefig("data_outputs/run6poster/recovery_fraction_heatmap.png", dpi=300)

# ---------------------------
# 7) Corner plot of parameters vs detection metrics
#     - Use injected set
#     - Scatter points colored by detection (>= best_thr)
#     - Annotate Spearman rho with SNR on the diagonal
# ---------------------------
params = ["r_p", "P_days", "inc", "a", "r_s"]
corner_df = inj_det[params + ["snr","detected"]].dropna()

k = len(params)
size = 2.4  # inches per panel
fig5 = plt.figure(figsize=(size*k, size*k), dpi=140)

# Axes grid
axes = [[plt.subplot2grid((k,k), (i,j)) for j in range(k)] for i in range(k)]

for i in range(k):
    for j in range(k):
        ax = axes[i][j]
        if i == j:
            # Diagonal: hist of the parameter + Spearman with SNR
            vals = corner_df[params[i]].values
            ax.hist(vals, bins=min(50, max(20, int(np.sqrt(len(vals))))), alpha=0.8)
            rho, p = spearmanr(vals, corner_df["snr"].values, nan_policy='omit')
            ax.set_title(f"{params[i]}   ρ_s(SNR)={rho:.2f}", fontsize=9)
        elif i > j:
            # Lower triangle: scatter param_j vs param_i
            x = corner_df[params[j]].values
            y = corner_df[params[i]].values
            det = corner_df["detected"].values
            ax.scatter(x[~det], y[~det], s=6, alpha=0.5, label="non-det")
            ax.scatter(x[det],  y[det],  s=6, alpha=0.7, label="det")
            if i == k-1 and j == 0:
                ax.legend(loc="best", fontsize=7, framealpha=0.8)
        else:
            # Upper triangle: blank to keep classic corner look
            ax.axis("off")

        # Ticks light
        if i < k-1:
            ax.set_xticklabels([])
        if j > 0:
            ax.set_yticklabels([])

        # Axis labels only on left/bottom
        if j == 0:
            ax.set_ylabel(params[i])
        if i == k-1:
            ax.set_xlabel(params[j])

plt.suptitle("Corner: Parameter Correlations & Detection (Injected Only)", y=0.92)
plt.tight_layout(rect=[0,0,1,0.95])
fig5.savefig("data_outputs/run6poster/corner_parameters.png", dpi=300)

print("[DONE] Saved figures:\n",
      " - precision_recall_vs_threshold.png\n",
      " - precision_recall_curve.png\n",
      " - snr_histograms.png\n",
      " - recovery_fraction_heatmap.png\n",
      " - corner_parameters.png")