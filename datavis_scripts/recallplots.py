# Create an updated script that:
# - Accepts two CSVs: one containing all-positive (true signals) systems, and one containing all-negative systems
# - Uses 'snr' as the score; default detection threshold = 3.70 (overridable)
# - Computes parameters: radius_ratio = r_p * 6_400_000 / r_s; period; impact_parameter = a * cos(inc)/r_s (auto-deg/rad)
# - Builds performance plots over parameter space (1D binned line plots + 2D heatmaps)
# - Outputs confusion matrix, calibration curve, error histograms, and summary
# - Uses matplotlib only (no seaborn), one chart per figure
# - Saves figures and CSVs to --out directory

"""
Classifier Performance Visualization for Positive + Negative CSVs

Usage:
    python perf_viz_two_csv.py --pos_csv positives.csv --neg_csv negatives.csv --out out_dir --thr 3.70 --bins 10

Assumptions:
- The positives CSV contains only TRUE systems (ground-truth label = 1).
- The negatives CSV contains only FALSE systems (ground-truth label = 0).
- Both CSVs contain the same schema and include at least:
    snr, r_p, r_s, a, inc, period
- The classifier score is SNR itself. Predicted label = (snr >= threshold). Default threshold = 3.70.

Outputs (saved into --out):
- Confusion matrix (confusion_matrix.png)
- Calibration curve for SNR-as-score (calibration_curve.png)
- Per-parameter precision/recall/F1 line plots (+ CSVs with metrics)
- 2D heatmaps (precision/recall/F1) for each param pair
- FP/FN histograms over each parameter
- 'summary.txt' with overall metrics and counts
- 'params_overview.csv' with the engineered parameters and labels for reference

Note:
- Inclination 'inc' auto-detected: if >=20% of values > 2π, treat as degrees, else radians.
"""

import argparse
import os
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def ensure_out(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

def safe_cos_inc(inc_series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(inc_series, errors='coerce').to_numpy()
    if np.isnan(vals).all():
        raise ValueError("Column 'inc' could not be parsed as numeric.")
    frac_gt_2pi = np.mean(vals > (2 * np.pi))
    if frac_gt_2pi > 0.2:
        vals = np.deg2rad(vals)
    return np.cos(vals)

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int,int,int,int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def pr_from_conf(tp: int, fp: int, fn: int) -> Tuple[float,float,float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = f1_score(precision, recall)
    return precision, recall, f1

def basic_line_plot(x, y, xlabel, ylabel, title, out_path):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def basic_heatmap(Z, x_edges, y_edges, xlabel, ylabel, title, out_path, vmin=0.0, vmax=1.0):
    plt.figure()
    X, Y = np.meshgrid(x_edges, y_edges)
    im = plt.pcolormesh(X, Y, Z, shading='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label('value')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def confusion_matrix_plot(tp, tn, fp, fn, out_path):
    plt.figure()
    M = np.array([[tp, fp],
                  [fn, tn]])
    plt.imshow(M, aspect='equal')
    plt.xticks([0,1], ['Pred:1', 'Pred:0'])
    plt.yticks([0,1], ['True:1', 'True:0'])
    for (i, j), val in np.ndenumerate(M):
        plt.text(j, i, str(val), ha='center', va='center')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def calibration_curve_plot(y_true, y_score, n_bins, out_path):
    mask = (~np.isnan(y_true)) & (~np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]
    try:
        bins = pd.qcut(y_score, q=n_bins, duplicates='drop')
    except ValueError:
        bins = pd.cut(y_score, bins=n_bins)
    df = pd.DataFrame({'y_true': y_true, 'y_score': y_score, 'bin': bins})
    rows = []
    for b, sub in df.groupby('bin'):
        pred_mean = sub['y_score'].mean()
        obs_rate = sub['y_true'].mean() if len(sub) > 0 else np.nan
        rows.append((pred_mean, obs_rate))
    rows = sorted(rows, key=lambda t: t[0])
    preds = [r[0] for r in rows]
    obs   = [r[1] for r in rows]
    plt.figure()
    plt.plot([0,1],[0,1],'--', linewidth=1)
    # If SNR is not in [0,1], rescale to [0,1] for the x-axis line of identity to be meaningful.
    if len(preds) > 0:
        mn, mx = min(preds), max(preds)
        if not (0.0 <= mn and mx <= 1.0) and mx > mn:
            preds_plot = [(p - mn) / (mx - mn) for p in preds]
        else:
            preds_plot = preds
    else:
        preds_plot = preds
    plt.plot(preds_plot, obs, marker='o')
    plt.xlabel('Predicted score (binned, rescaled if needed)')
    plt.ylabel('Observed positive rate')
    plt.title('Calibration Curve (SNR as score)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def binned_metrics(y_true: np.ndarray, y_pred: np.ndarray, by: np.ndarray, bins: int) -> pd.DataFrame:
    mask = (~np.isnan(by)) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    s = pd.Series(by[mask])
    try:
        cats = pd.qcut(s, q=bins, duplicates='drop')
    except ValueError:
        cats = pd.cut(s, bins=bins)
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'bin': cats, 'by': s.to_numpy()})
    rows = []
    for interval, sub in df.groupby('bin'):
        tp = int(np.sum((sub['y_true'] == 1) & (sub['y_pred'] == 1)))
        tn = int(np.sum((sub['y_true'] == 0) & (sub['y_pred'] == 0)))
        fp = int(np.sum((sub['y_true'] == 0) & (sub['y_pred'] == 1)))
        fn = int(np.sum((sub['y_true'] == 1) & (sub['y_pred'] == 0)))
        precision, recall, f1 = pr_from_conf(tp, fp, fn)
        # Bin center estimate
        if hasattr(interval, 'left'):
            center = (interval.left + interval.right) / 2.0
        else:
            center = float(np.mean(sub['by']))
        rows.append({
            'bin': str(interval),
            'center': center,
            'count': len(sub),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    return pd.DataFrame(rows).sort_values('center').reset_index(drop=True)

def binned_metrics_2d(y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray, y: np.ndarray, bins: int) -> Dict[str, np.ndarray]:
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    x, y, y_true, y_pred = x[mask], y[mask], y_true[mask], y_pred[mask]

    def quantile_edges(arr, k):
        qs = np.linspace(0, 1, k+1)
        return np.unique(np.quantile(arr, qs))
    x_edges = quantile_edges(x, bins)
    y_edges = quantile_edges(y, bins)
    if x_edges.size < 2:
        x_edges = np.linspace(np.min(x), np.max(x), 2)
    if y_edges.size < 2:
        y_edges = np.linspace(np.min(y), np.max(y), 2)

    xi = np.digitize(x, x_edges) - 1
    yi = np.digitize(y, y_edges) - 1
    xi = np.clip(xi, 0, len(x_edges)-2)
    yi = np.clip(yi, 0, len(y_edges)-2)

    H = len(y_edges)-1
    W = len(x_edges)-1
    TP = np.zeros((H, W), dtype=int)
    TN = np.zeros_like(TP)
    FP = np.zeros_like(TP)
    FN = np.zeros_like(TP)
    for i in range(len(xi)):
        r, c = yi[i], xi[i]
        yt, yp = y_true[i], y_pred[i]
        if yt == 1 and yp == 1: TP[r, c] += 1
        elif yt == 0 and yp == 0: TN[r, c] += 1
        elif yt == 0 and yp == 1: FP[r, c] += 1
        elif yt == 1 and yp == 0: FN[r, c] += 1

    def safe_div(a, b):
        out = np.zeros_like(a, dtype=float)
        nz = b != 0
        out[nz] = a[nz] / b[nz]
        return out
    precision = safe_div(TP, TP + FP)
    recall    = safe_div(TP, TP + FN)
    with np.errstate(invalid='ignore'):
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'x_edges': x_edges,
        'y_edges': y_edges,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos_csv', required=True, help='CSV with all TRUE systems')
    ap.add_argument('--neg_csv', required=True, help='CSV with all FALSE systems')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--thr', type=float, default=3.70, help='Detection threshold applied to SNR')
    ap.add_argument('--bins', type=int, default=10, help='Number of bins for 1D/2D binning')
    args = ap.parse_args()

    ensure_out(args.out)

    # Load CSVs and tag labels
    df_pos = pd.read_csv(args.pos_csv)
    df_neg = pd.read_csv(args.neg_csv)
    df_pos['y_true'] = 1
    df_neg['y_true'] = 0
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    # Required columns
    required_cols = {'snr', 'r_p', 'r_s', 'a', 'inc', 'period'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV(s) missing required columns: {missing}")

    # Score and predicted label from SNR
    snr = pd.to_numeric(df['snr'], errors='coerce').to_numpy()
    y_true = pd.to_numeric(df['y_true'], errors='coerce').fillna(0).astype(int).to_numpy()
    y_pred = (snr >= args.thr).astype(int)

    # Engineered parameters
    r_p = pd.to_numeric(df['r_p'], errors='coerce').to_numpy()
    r_s = pd.to_numeric(df['r_s'], errors='coerce').to_numpy()
    period = pd.to_numeric(df['period'], errors='coerce').to_numpy()
    a_vals = pd.to_numeric(df['a'], errors='coerce').to_numpy()
    inc_cos = np.cos(pd.to_numeric(df['inc'], errors='coerce').to_numpy() * (math.pi / 180.0))
    radius_ratio = r_p * 6_400_000.0 / r_s
    impact_param = a_vals * inc_cos / r_s

    # Save a joined reference CSV
    ref = pd.DataFrame({
        'snr': snr,
        'y_true': y_true,
        'y_pred': y_pred,
        'radius_ratio': radius_ratio,
        'period': period,
        'impact_parameter': impact_param
    })
    ref.to_csv(os.path.join(args.out, 'params_overview.csv'), index=False)

    # Overall metrics
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    precision, recall, f1 = pr_from_conf(tp, fp, fn)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write(f'Threshold on SNR: {args.thr}\n')
        f.write(f'Counts: TP={tp}, FP={fp}, FN={fn}, TN={tn}\n')
        f.write(f'Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n')

    # Confusion matrix
    confusion_matrix_plot(tp, tn, fp, fn, os.path.join(args.out, 'confusion_matrix.png'))

    # Per-parameter metrics
    params = {
        'radius_ratio': radius_ratio,
        'period': period,
        'impact_parameter': impact_param
    }
    for pname, arr in params.items():
        res = binned_metrics(y_true, y_pred, arr, bins=args.bins)
        res.to_csv(os.path.join(args.out, f'{pname}_binned_metrics.csv'), index=False)
        basic_line_plot(res['center'], res['precision'], pname, 'Precision',
                        f'Precision vs {pname}', os.path.join(args.out, f'{pname}_precision.png'))
        basic_line_plot(res['center'], res['recall'], pname, 'Recall',
                        f'Recall vs {pname}', os.path.join(args.out, f'{pname}_recall.png'))
        basic_line_plot(res['center'], res['f1'], pname, 'F1 Score',
                        f'F1 vs {pname}', os.path.join(args.out, f'{pname}_f1.png'))

    # 2D heatmaps
    pairs = [
        ('radius_ratio', radius_ratio, 'period', period),
        ('radius_ratio', radius_ratio, 'impact_parameter', impact_param),
        ('period', period, 'impact_parameter', impact_param),
    ]
    for (xname, xarr, yname, yarr) in pairs:
        grid = binned_metrics_2d(y_true, y_pred, xarr, yarr, bins=args.bins)
        basic_heatmap(grid['precision'], grid['x_edges'], grid['y_edges'],
                      xname, yname, f'Precision heatmap: {yname} vs {xname}',
                      os.path.join(args.out, f'heatmap_precision_{xname}_{yname}.png'))
        basic_heatmap(grid['recall'], grid['x_edges'], grid['y_edges'],
                      xname, yname, f'Recall heatmap: {yname} vs {xname}',
                      os.path.join(args.out, f'heatmap_recall_{xname}_{yname}.png'))
        basic_heatmap(grid['f1'], grid['x_edges'], grid['y_edges'],
                      xname, yname, f'F1 heatmap: {yname} vs {xname}',
                      os.path.join(args.out, f'heatmap_f1_{xname}_{yname}.png'))

    # Calibration curve (SNR as score). We rescale x if SNR not in [0,1] for identity line display.
    calibration_curve_plot(y_true, snr, n_bins=args.bins,
                           out_path=os.path.join(args.out, 'calibration_curve.png'))

    # Error histograms
    FP_mask = (y_true == 0) & (y_pred == 1)
    FN_mask = (y_true == 1) & (y_pred == 0)

    def hist_plot(values, xlabel, title, out_path):
        plt.figure()
        plt.hist(values, bins=20)
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    for pname, arr in params.items():
        vals = arr[FP_mask]
        if np.isfinite(vals).sum() > 0:
            hist_plot(vals, xlabel=pname, title=f'False Positives over {pname}',
                      out_path=os.path.join(args.out, f'FP_hist_{pname}.png'))
        vals = arr[FN_mask]
        if np.isfinite(vals).sum() > 0:
            hist_plot(vals, xlabel=pname, title=f'False Negatives over {pname}',
                      out_path=os.path.join(args.out, f'FN_hist_{pname}.png'))

    print(f"Done. Outputs saved in: {args.out}")

if __name__ == '__main__':
    main()

'''
python datavis_scripts/recallplots.py \
  --pos_csv data_outputs/injected_transits_output6.csv \
  --neg_csv data_outputs/noninjected_transits_output6.csv \
  --out data_outputs/run6analysis/recallplots \
  --thr 3.70 \
  --bins 10
'''