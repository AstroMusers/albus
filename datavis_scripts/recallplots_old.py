"""
Classifier Performance Visualization (Pos/Neg CSVs) – v2

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
from tdpy.plotting import apply_plot_theme, configure_plot_style
from tdpy.verbosity import print

if __package__:
    from .recallplots import (
        basic_line_plot, binned_metrics_2d_uniform, binned_metrics_uniform,
        calibration_curve_plot, confusion_counts, confusion_matrix_plot,
        f1_score, pr_from_conf, uniform_edges,
    )
else:
    from recallplots import (
        basic_line_plot, binned_metrics_2d_uniform, binned_metrics_uniform,
        calibration_curve_plot, confusion_counts, confusion_matrix_plot,
        f1_score, pr_from_conf, uniform_edges,
    )

def ensure_out(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def _save_figure(figure, out_path):
    print(f'Writing to {out_path}...')
    figure.savefig(out_path, dpi=200, facecolor=figure.get_facecolor())


def basic_heatmap(Z, x_edges, y_edges, xlabel, ylabel, title, out_path,
                  x_log=False, y_log=False, vmin=0.0, vmax=1.0, typeplotback=None):
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
    apply_plot_theme(fig, typeplotback)
    _save_figure(fig, out_path)
    plt.close(fig)

def corner_heatmap(params: Dict[str, np.ndarray],
                   y_true: np.ndarray, y_pred: np.ndarray,
                   bins: int, metric: str, out_path: str,
                   log_params: List[str] = None, typeplotback=None):
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

    fig.suptitle(f'Corner Heatmap – {metric.capitalize()}', y=0.995)
    plt.tight_layout(rect=[0.02, 0.02, 0.90, 0.97])
    apply_plot_theme(fig, typeplotback)
    _save_figure(fig, out_path)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos_csv', required=True, help='CSV with all TRUE systems')
    ap.add_argument('--neg_csv', required=True, help='CSV with all FALSE systems')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--thr', type=float, default=3.70, help='Detection threshold applied to SNR')
    ap.add_argument('--bins', type=int, default=10, help='Number of bins for 1D/2D binning (same for x/y)')
    ap.add_argument('--typeplotback', choices=('white', 'dark'), default='white')
    args = ap.parse_args()
    configure_plot_style(args.typeplotback)

    ensure_out(args.out)

    # Load CSVs and labels
    print(f'Reading from {args.pos_csv}...')
    df_pos = pd.read_csv(args.pos_csv)
    print(f'Reading from {args.neg_csv}...')
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
    overview_path = os.path.join(args.out, 'params_overview.csv')
    print(f'Writing to {overview_path}...')
    ref.to_csv(overview_path, index=False)

    # Overall metrics
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    precision, recall, f1 = pr_from_conf(tp, fp, fn)
    summary_path = os.path.join(args.out, 'summary.txt')
    print(f'Writing to {summary_path}...')
    with open(summary_path, 'w') as summary_file:
        summary_file.write(f'Threshold on SNR: {args.thr}\n')
        summary_file.write(f'Counts: TP={tp}, FP={fp}, FN={fn}, TN={tn}\n')
        summary_file.write(f'Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n')

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
        metrics_path = os.path.join(args.out, f'{pname}_binned_metrics.csv')
        print(f'Writing to {metrics_path}...')
        res.to_csv(metrics_path, index=False)
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
        figure = apply_plot_theme(plt.gcf())
        _save_figure(figure, out_path)
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
