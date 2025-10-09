"""
Classifier Performance Visualization (Pos/Neg CSVs) — v2

Changes vs previous version:
- Uses inclination directly (no impact parameter).
- When plotting *radius_ratio* or *inclination*, the corresponding axes use a logarithmic scale.
- Adds CORNER PLOTS (pairwise heatmaps) for both PRECISION and RECALL over parameters:
    ['radius_ratio', 'period', 'inclination', 'a', 'duration', 'mean']

Usage:
    python datavis_scripts/recallplots.py \
        --pos_csv data_outputs/injected_transits_output6.csv \
        --neg_csv data_outputs/noninjected_transits_output6.csv \
        --out data_outputs/run6analysis/recallplots2 \
        --thr 3.70 \
        --bins 20
"""

import argparse
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def ensure_out(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

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

def basic_line_plot(x, y, xlabel, ylabel, title, out_path, x_log=False):
    plt.figure()
    plt.plot(x, y, marker='o')
    if x_log:
        xmin = np.nanmin(np.array(x)[np.array(x) > 0]) if np.any(np.array(x) > 0) else 1.0
        plt.xscale('log')
        plt.xlim(left=xmin*0.95)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def basic_heatmap(Z, x_edges, y_edges, xlabel, ylabel, title, out_path,
                  x_log=False, y_log=False, vmin=0.0, vmax=1.0):
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(x_edges, y_edges)
    im = ax.pcolormesh(X, Y, Z, shading='auto', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('value')
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Make cells visually square
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

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
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1],'--', linewidth=1)
    if len(preds) > 0:
        mn, mx = min(preds), max(preds)
        if not (0.0 <= mn and mx <= 1.0) and mx > mn:
            preds_plot = [(p - mn) / (mx - mn) for p in preds]
        else:
            preds_plot = preds
    else:
        preds_plot = preds
    ax.plot(preds_plot, obs, marker='o')
    ax.set_xlabel('Predicted score (binned, rescaled if needed)')
    ax.set_ylabel('Observed positive rate')
    ax.set_title('Calibration Curve (SNR as score)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def uniform_edges(arr: np.ndarray, bins: int, log_scale: bool) -> np.ndarray:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0.0, 1.0])
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if mx == mn:
        return np.array([mn, mx + (1e-9 if mx == 0 else abs(mx)*1e-9)])
    if log_scale and mn > 0 and mx > 0:
        return np.logspace(np.log10(mn), np.log10(mx), bins + 1)
    else:
        return np.linspace(mn, mx, bins + 1)

def binned_metrics_uniform(y_true: np.ndarray, y_pred: np.ndarray, by: np.ndarray,
                           bins: int, log_scale: bool) -> pd.DataFrame:
    mask = (~np.isnan(by)) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    s = by[mask].astype(float)

    edges = uniform_edges(s, bins, log_scale=log_scale)
    idx = np.digitize(s, edges) - 1
    idx = np.clip(idx, 0, len(edges)-2)

    rows = []
    for k in range(len(edges)-1):
        sel = (idx == k)
        center = 10**((np.log10(edges[k]) + np.log10(edges[k+1]))/2.0) if log_scale and edges[k]>0 and edges[k+1]>0 else (edges[k] + edges[k+1]) / 2.0
        if not np.any(sel):
            rows.append({
                'bin': f'[{edges[k]:.6g}, {edges[k+1]:.6g})',
                'center': center,
                'count': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            })
            continue
        yt = y_true[sel]
        yp = y_pred[sel]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        precision, recall, f1 = pr_from_conf(tp, fp, fn)
        rows.append({
            'bin': f'[{edges[k]:.6g}, {edges[k+1]:.6g})',
            'center': center,
            'count': int(np.sum(sel)),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1
        })
    out = pd.DataFrame(rows)
    return out.sort_values('center').reset_index(drop=True)

def binned_metrics_2d_uniform(y_true: np.ndarray, y_pred: np.ndarray,
                              x: np.ndarray, y: np.ndarray, bins: int,
                              x_log: bool, y_log: bool) -> Dict[str, np.ndarray]:
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    x, y, y_true, y_pred = x[mask].astype(float), y[mask].astype(float), y_true[mask], y_pred[mask]

    x_edges = uniform_edges(x, bins, log_scale=x_log)
    y_edges = uniform_edges(y, bins, log_scale=y_log)

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

def corner_heatmap(params: Dict[str, np.ndarray],
                   y_true: np.ndarray, y_pred: np.ndarray,
                   bins: int, metric: str, out_path: str,
                   log_params: List[str] = None):
    if log_params is None:
        log_params = []
    names = list(params.keys())
    n = len(names)

    fig, axes = plt.subplots(n, n, figsize=(3.2*n, 3.2*n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i <= j:
                ax.axis('off')
                if i == j:
                    ax.text(0.5, 0.5, names[i], ha='center', va='center', fontsize=12)
                continue

            xname, yname = names[j], names[i]
            xarr, yarr = params[xname], params[yname]
            x_log = xname in log_params
            y_log = yname in log_params
            grid = binned_metrics_2d_uniform(y_true, y_pred, xarr, yarr, bins=bins,
                                             x_log=x_log, y_log=y_log)
            Z = grid[metric]
            X, Y = np.meshgrid(grid['x_edges'], grid['y_edges'])
            im = ax.pcolormesh(X, Y, Z, shading='auto', vmin=0.0, vmax=1.0)
            if x_log:
                ax.set_xscale('log')
            if y_log:
                ax.set_yscale('log')
            if i == n-1:
                ax.set_xlabel(xname)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(yname)
            else:
                ax.set_yticklabels([])
            # Force square cells
            ax.set_aspect('equal', adjustable='box')

    # Colorbar
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    mappable = None
    for i in range(n-1, -1, -1):
        for j in range(0, i):
            colls = axes[i, j].collections
            if colls:
                mappable = colls[0]
                break
        if mappable is not None:
            break
    if mappable is not None:
        fig.colorbar(mappable, cax=cax, label=metric)

    fig.suptitle(f'Corner Heatmap — {metric.capitalize()}', y=0.995)
    plt.tight_layout(rect=[0.02, 0.02, 0.90, 0.97])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos_csv', required=True, help='CSV with all TRUE systems')
    ap.add_argument('--neg_csv', required=True, help='CSV with all FALSE systems')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--thr', type=float, default=3.70, help='Detection threshold applied to SNR')
    ap.add_argument('--bins', type=int, default=10, help='Number of bins for 1D/2D binning (same for x/y)')
    args = ap.parse_args()

    ensure_out(args.out)

    # Load CSVs and labels
    df_pos = pd.read_csv(args.pos_csv)
    df_neg = pd.read_csv(args.neg_csv)
    df_pos['y_true'] = 1
    df_neg['y_true'] = 0
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    # Required columns
    required_cols = {'snr', 'r_p', 'r_s', 'a', 'inc', 'period', 'duration', 'mean'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV(s) missing required columns: {missing}")

    # Score & labels
    snr = pd.to_numeric(df['snr'], errors='coerce').to_numpy()
    y_true = pd.to_numeric(df['y_true'], errors='coerce').fillna(0).astype(int).to_numpy()
    y_pred = (snr >= args.thr).astype(int)

    # Parameters
    r_p = pd.to_numeric(df['r_p'], errors='coerce').to_numpy()
    r_s = pd.to_numeric(df['r_s'], errors='coerce').to_numpy()
    radius_ratio = r_p * 6_400_000.0 / r_s
    period = pd.to_numeric(df['period'], errors='coerce').to_numpy()
    inclination = pd.to_numeric(df['inc'], errors='coerce').to_numpy()
    a_vals = pd.to_numeric(df['a'], errors='coerce').to_numpy()
    duration = pd.to_numeric(df['duration'], errors='coerce').to_numpy()
    mean_depth = pd.to_numeric(df['mean'], errors='coerce').to_numpy()

    # Save joined reference CSV
    ref = pd.DataFrame({
        'snr': snr,
        'y_true': y_true,
        'y_pred': y_pred,
        'radius_ratio': radius_ratio,
        'period': period,
        'inclination': inclination,
        'a': a_vals,
        'duration': duration,
        'mean': mean_depth,
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

    # 1D binned metrics (uniform bins; log x-axis ONLY for radius_ratio)
    params_1d = {
        'radius_ratio': radius_ratio,
        'period': period,
        'inclination': inclination,
        'a': a_vals,
        'duration': duration,
        'mean': mean_depth,
    }
    for pname, arr in params_1d.items():
        log_scale = (pname == 'radius_ratio')
        res = binned_metrics_uniform(y_true, y_pred, arr, bins=args.bins, log_scale=log_scale)
        res.to_csv(os.path.join(args.out, f'{pname}_binned_metrics.csv'), index=False)
        basic_line_plot(res['center'], res['precision'], pname, 'Precision',
                        f'Precision vs {pname}', os.path.join(args.out, f'{pname}_precision.png'),
                        x_log=log_scale)
        basic_line_plot(res['center'], res['recall'], pname, 'Recall',
                        f'Recall vs {pname}', os.path.join(args.out, f'{pname}_recall.png'),
                        x_log=log_scale)
        basic_line_plot(res['center'], res['f1'], pname, 'F1 Score',
                        f'F1 vs {pname}', os.path.join(args.out, f'{pname}_f1.png'),
                        x_log=log_scale)

    # 2D heatmaps (uniform bins; square cells; log axes when axis is radius_ratio)
    pairs = [
        ('radius_ratio', radius_ratio, 'period', period),
        ('radius_ratio', radius_ratio, 'inclination', inclination),
        ('period', period, 'inclination', inclination),
        ('radius_ratio', radius_ratio, 'a', a_vals),
        ('radius_ratio', radius_ratio, 'duration', duration),
        ('radius_ratio', radius_ratio, 'mean', mean_depth),
    ]
    for (xname, xarr, yname, yarr) in pairs:
        x_log = (xname == 'radius_ratio')
        y_log = (yname == 'radius_ratio')
        grid = binned_metrics_2d_uniform(y_true, y_pred, xarr, yarr, bins=args.bins,
                                         x_log=x_log, y_log=y_log)
        basic_heatmap(grid['precision'], grid['x_edges'], grid['y_edges'],
                      xname, yname, f'Precision heatmap: {yname} vs {xname}',
                      os.path.join(args.out, f'heatmap_precision_{xname}_{yname}.png'),
                      x_log=x_log, y_log=y_log)
        basic_heatmap(grid['recall'], grid['x_edges'], grid['y_edges'],
                      xname, yname, f'Recall heatmap: {yname} vs {xname}',
                      os.path.join(args.out, f'heatmap_recall_{xname}_{yname}.png'),
                      x_log=x_log, y_log=y_log)
        basic_heatmap(grid['f1'], grid['x_edges'], grid['y_edges'],
                      xname, yname, f'F1 heatmap: {yname} vs {xname}',
                      os.path.join(args.out, f'heatmap_f1_{xname}_{yname}.png'),
                      x_log=x_log, y_log=y_log)

    # Calibration curve (SNR as score). Rescale x if SNR not in [0,1].
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

    for pname, arr in params_1d.items():
        vals = arr[FP_mask]
        if np.isfinite(vals).sum() > 0:
            hist_plot(vals, xlabel=pname, title=f'False Positives over {pname}',
                      out_path=os.path.join(args.out, f'FP_hist_{pname}.png'))
        vals = arr[FN_mask]
        if np.isfinite(vals).sum() > 0:
            hist_plot(vals, xlabel=pname, title=f'False Negatives over {pname}',
                      out_path=os.path.join(args.out, f'FN_hist_{pname}.png'))

    # Corner plots (pairwise heatmaps) for precision and recall (uniform bins; square cells)
    corner_params = {
        'radius_ratio': radius_ratio,
        'period': period,
        'inclination': inclination,
        'a': a_vals,
        'duration': duration,
        'mean': mean_depth,
    }
    log_params = ['radius_ratio']  # only radius_ratio uses log scale
    corner_heatmap(corner_params, y_true, y_pred, bins=args.bins, metric='precision',
                   out_path=os.path.join(args.out, 'corner_precision.png'),
                   log_params=log_params)
    corner_heatmap(corner_params, y_true, y_pred, bins=args.bins, metric='recall',
                   out_path=os.path.join(args.out, 'corner_recall.png'),
                   log_params=log_params)

    print(f"Done. Outputs saved in: {args.out}")

if __name__ == '__main__':
    main()