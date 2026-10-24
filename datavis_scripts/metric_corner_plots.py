import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, LogFormatterMathtext
from tdpy.verbosity import print

# -------- global formatters / locators --------
fmt_plain = ScalarFormatter(useOffset=False)
fmt_plain.set_scientific(False)
logloc = LogLocator(base=10.0)
logfmt = LogFormatterMathtext(base=10.0)

# -------- helpers --------
def safe_div(a, b, fill=0.0):
    return (a / b) if b > 0 else fill

def f1_optimal_threshold(snr_pos, snr_neg, n=200, lo=1.0, hi=100.0):
    thresholds = np.logspace(np.log10(lo), np.log10(hi), n)
    y_true = np.concatenate([np.ones_like(snr_pos, int), np.zeros_like(snr_neg, int)])
    snr_all = np.concatenate([snr_pos, snr_neg])
    best_f1, best_thr = -1.0, thresholds[0]
    for thr in thresholds:
        y_pred = (snr_all >= thr).astype(int)
        tp = np.sum((y_true==1)&(y_pred==1))
        fp = np.sum((y_true==0)&(y_pred==1))
        fn = np.sum((y_true==1)&(y_pred==0))
        prec = safe_div(tp, tp+fp)
        rec  = safe_div(tp, tp+fn)
        f1   = safe_div(2*prec*rec, prec+rec)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)

def make_bins(data, kind="log", nbins=24):
    x = data[np.isfinite(data)]
    if x.size == 0:
        return np.array([0, 1])
    if kind == "log":
        pos = x[x > 0]
        xmin = pos.min() if pos.size else 1e-6
        xmax = x.max()
        if xmax <= xmin: xmax = xmin * 1.01
        return np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    elif kind == "linear":
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        if xmax <= xmin: xmax = xmin + 1e-9
        return np.linspace(xmin, xmax, nbins+1)
    elif kind == "quantile":
        qs = np.linspace(0, 1, nbins+1)
        edges = np.unique(np.quantile(x, qs))
        if edges.size < max(5, nbins//3):
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            edges = np.linspace(xmin, xmax, nbins+1)
        return edges
    else:
        raise ValueError("Unknown bin kind")

def metric_in_bins(x_pos, y_pos, det_pos, x_neg, y_neg, det_neg,
                   xb, yb, metric="precision",
                   min_total=5, min_pred=5):
    H_pos_all, _, _ = np.histogram2d(x_pos, y_pos, bins=[xb, yb])
    H_pos_det, _, _ = np.histogram2d(x_pos[det_pos], y_pos[det_pos], bins=[xb, yb])
    H_neg_pred, _, _ = np.histogram2d(x_neg[det_neg], y_neg[det_neg], bins=[xb, yb])

    TP = H_pos_det
    FP = H_neg_pred
    FN = H_pos_all - H_pos_det

    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "precision":
            den = TP + FP
            val = np.where(den > 0, TP/den, np.nan)
            val[den < min_pred] = np.nan
        elif metric == "recall":
            den = TP + FN
            val = np.where(den > 0, TP/den, np.nan)
            val[den < min_total] = np.nan
        else:
            raise ValueError("metric must be 'precision' or 'recall'")
    return val.T  # for pcolormesh (x along columns)

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="injected_transits_output6.csv")
    ap.add_argument("--neg", default="noninjected_transits_output6.csv")
    ap.add_argument("--snr-col", default="snr")
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--min-total", type=int, default=5)
    ap.add_argument("--min-pred",  type=int, default=5)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--thr-lo", type=float, default=1.0)
    ap.add_argument("--thr-hi", type=float, default=100.0)
    ap.add_argument("--n-thr",  type=int, default=200)
    ap.add_argument("--outdir", default="data_outputs/run6poster")
    ap.add_argument("--dpi", type=int, default=300)  # poster-ready
    ap.add_argument("--panel", type=float, default=3.2, help="inches per panel side")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Load
    print(f"Reading from {args.pos}...")
    pos_df = pd.read_csv(args.pos)
    print(f"Reading from {args.neg}...")
    neg_df = pd.read_csv(args.neg)
    need = ["r_p", "r_s", "P_days", "a", "inc", args.snr_col]
    for c in need:
        if c not in pos_df.columns: raise ValueError(f"{c} missing in {args.pos}")
        if c not in neg_df.columns: raise ValueError(f"{c} missing in {args.neg}")

    # ---- Parameters (units)
    R_E = 6.371e6
    AU  = 1.495978707e11

    # radii in meters for ratio
    rp_m_pos = pos_df["r_p"].astype(float) * R_E
    rp_m_neg = neg_df["r_p"].astype(float) * R_E
    rs_m_pos = pos_df["r_s"].astype(float)
    rs_m_neg = neg_df["r_s"].astype(float)

    Rp_Rs_pos = pd.to_numeric(rp_m_pos / rs_m_pos, errors="coerce").to_numpy()
    Rp_Rs_neg = pd.to_numeric(rp_m_neg / rs_m_neg, errors="coerce").to_numpy()

    # period (days), linear
    P_pos = pd.to_numeric(pos_df["P_days"], errors="coerce").to_numpy()
    P_neg = pd.to_numeric(neg_df["P_days"], errors="coerce").to_numpy()

    # a shown in AU for readability (linear)
    a_pos = pd.to_numeric(pos_df["a"], errors="coerce").to_numpy() / AU
    a_neg = pd.to_numeric(neg_df["a"], errors="coerce").to_numpy() / AU
    a_label = r"$a\ \mathrm{(AU)}$"

    # impact parameter using meters (do NOT convert a here)
    i_rad_pos = np.deg2rad(pos_df["inc"].astype(float))
    i_rad_neg = np.deg2rad(neg_df["inc"].astype(float))
    b_pos = (pos_df["a"].astype(float) * np.cos(i_rad_pos)) / rs_m_pos
    b_neg = (neg_df["a"].astype(float) * np.cos(i_rad_neg)) / rs_m_neg

    # pre-log b for better readability
    eps = 1e-12
    logb_pos = np.log10(np.maximum(b_pos, eps))
    logb_neg = np.log10(np.maximum(b_neg, eps))

    # SNR
    snr_pos = pd.to_numeric(pos_df[args.snr_col], errors="coerce").to_numpy()
    snr_neg = pd.to_numeric(neg_df[args.snr_col], errors="coerce").to_numpy()

    # ---- Global F1-optimal threshold
    if args.threshold is None:
        thr, f1 = f1_optimal_threshold(snr_pos[snr_pos > 0], snr_neg[snr_neg > 0],
                                       n=args.n_thr, lo=args.thr_lo, hi=args.thr_hi)
        threshold = thr
        print(f"[INFO] F1-optimal threshold: {threshold:.4f} (global F1={f1:.3f})")
    else:
        threshold = float(args.threshold)
        print(f"[INFO] Fixed threshold: {threshold:.4f}")

    det_pos = (snr_pos >= threshold)
    det_neg = (snr_neg >= threshold)

    # ---- Params spec
    params = [
        (r"$R_p/R_s$",       Rp_Rs_pos, Rp_Rs_neg, "log"),
        (r"$P\ \mathrm{(days)}$", P_pos,    P_neg,    "linear"),
        (r"$\log_{10} b$",   logb_pos,  logb_neg,  "linear"),  # pre-logged
        (a_label,            a_pos,     a_neg,     "linear"),
    ]
    kinds   = [p[3] for p in params]
    labels  = [p[0] for p in params]
    pos_vals = [p[1] for p in params]
    neg_vals = [p[2] for p in params]
    bin_edges = [make_bins(np.concatenate([p[1], p[2]]), kind=p[3], nbins=args.bins) for p in params]
    k = len(params)

    # ---- Figure helpers
    def new_corner_fig(title):
        fig = plt.figure(figsize=(args.panel*k, args.panel*k), dpi=args.dpi)
        axes = [[None]*k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                ax = plt.subplot2grid((k, k), (i, j))
                axes[i][j] = ax
                if i < j:
                    ax.axis("off")
        fig.suptitle(title, y=0.94)
        return fig, axes

    def draw_corner(metric_name, fname, cbar_label):
        fig, axes = new_corner_fig(f"Corner: {cbar_label} in parameter space")
        im_last = None

        def _format_axes(ax, kind_x, kind_y):
            if kind_x == "log":
                ax.set_xscale("log"); ax.xaxis.set_major_locator(logloc); ax.xaxis.set_major_formatter(logfmt)
            else:
                ax.xaxis.set_major_formatter(fmt_plain)
            if kind_y == "log":
                ax.set_yscale("log"); ax.yaxis.set_major_locator(logloc); ax.yaxis.set_major_formatter(logfmt)
            else:
                ax.yaxis.set_major_formatter(fmt_plain)

        for i in range(k):
            for j in range(k):
                ax = axes[i][j]
                if i == j:
                    vpos, vneg, edges = pos_vals[i], neg_vals[i], bin_edges[i]
                    ax.hist(vpos[np.isfinite(vpos)], bins=edges, alpha=0.6, density=True, label="Injected")
                    ax.hist(vneg[np.isfinite(vneg)], bins=edges, alpha=0.6, density=True, label="Non-injected")
                    _format_axes(ax, kinds[i], "linear")  # only x matters
                    if j == 0:
                        ax.legend(fontsize=8)
                elif i > j:
                    x_pos, y_pos = pos_vals[j], pos_vals[i]
                    x_neg, y_neg = neg_vals[j], neg_vals[i]
                    xb, yb = bin_edges[j], bin_edges[i]

                    # finite masks
                    mpos = np.isfinite(x_pos) & np.isfinite(y_pos)
                    mneg = np.isfinite(x_neg) & np.isfinite(y_neg)

                    Z = metric_in_bins(x_pos[mpos], y_pos[mpos], det_pos[mpos],
                                       x_neg[mneg], y_neg[mneg], det_neg[mneg],
                                       xb, yb, metric=metric_name,
                                       min_total=args.min_total, min_pred=args.min_pred)
                    im_last = ax.pcolormesh(xb, yb, Z, shading="auto", vmin=0, vmax=1)
                    _format_axes(ax, kinds[j], kinds[i])

                # labels at edges only
                if i == k-1:
                    ax.set_xlabel(labels[j])
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(labels[i])
                else:
                    ax.set_yticklabels([])

        # shared colorbar + save
        cax = fig.add_axes([0.92, 0.12, 0.02, 0.72])
        fig.colorbar(im_last, cax=cax, label=cbar_label)
        fig.tight_layout(rect=[0, 0, 0.90, 0.95])
        out = os.path.join(args.outdir, fname)
        print(f"Writing to {out}...")
        fig.savefig(out, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)

    # ---- make both figures
    draw_corner("precision", "corner_precision.png", "Precision")
    draw_corner("recall",    "corner_recall.png",    "Recall")

if __name__ == "__main__":
    main()
