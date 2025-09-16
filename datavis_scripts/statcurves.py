import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_snr(path, col):
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"{col} not found in {path}, available: {df.columns}")
    return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()

def confusion(y_true, y_pred):
    tp = np.sum((y_true==1)&(y_pred==1))
    fp = np.sum((y_true==0)&(y_pred==1))
    tn = np.sum((y_true==0)&(y_pred==0))
    fn = np.sum((y_true==1)&(y_pred==0))
    return tp, fp, tn, fn

def safe_div(a,b): return a/b if b>0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", help="CSV with positive cases", default="data_outputs/injected_transits_output6.csv")
    ap.add_argument("--neg", help="CSV with negative cases", default="data_outputs/noninjected_transits_output6.csv")
    ap.add_argument("--snr-col", default="snr")
    ap.add_argument("--outdir", default="data_outputs/run6analysis")
    ap.add_argument("--n-thresholds", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    snr_pos = load_snr(args.pos, args.snr_col)
    snr_neg = load_snr(args.neg, args.snr_col)

    y_true = np.concatenate([np.ones(len(snr_pos)), np.zeros(len(snr_neg))])
    snr_all = np.concatenate([snr_pos, snr_neg])

    thresholds = np.logspace(np.log10(0.1), np.log10(100), args.n_thresholds)
    records = []
    for thr in thresholds:
        y_pred = (snr_all >= thr).astype(int)
        tp, fp, tn, fn = confusion(y_true, y_pred)
        prec = safe_div(tp, tp+fp)
        rec = safe_div(tp, tp+fn)
        f1 = safe_div(2*prec*rec, prec+rec)
        records.append({"threshold":thr,"TP":tp,"FP":fp,"TN":tn,"FN":fn,
                        "precision":prec,"recall":rec,"f1":f1})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.outdir,"metrics_by_threshold.csv"),index=False)

    # f1 curve
    plt.figure()
    plt.plot(df["threshold"], df["f1"])
    plt.xlabel("Threshold"); plt.ylabel("F1 Score"); plt.title("F1 Curve")
    plt.xscale("log"); plt.ylim(0, 1.05)
    max_f1 = df["f1"].max()
    max_f1_threshold = df.loc[df["f1"].idxmax(), "threshold"]
    plt.axvline(x=max_f1_threshold, color='r', linestyle='--', label=f'Max F1 = {max_f1:.2f} at Threshold = {max_f1_threshold:.2f}')
    plt.legend()
    plt.grid(); plt.savefig(os.path.join(args.outdir, "f1_curve.png")); plt.close()

    # precision curve
    plt.figure()
    plt.plot(df["threshold"], df["precision"])
    plt.xlabel("Threshold"); plt.ylabel("Precision"); plt.title("Precision Curve")
    plt.xscale("log"); plt.ylim(0,1.05)
    plt.axvline(x=max_f1_threshold, color='r', linestyle='--', label=f'Max F1 = {max_f1:.2f} at Threshold = {max_f1_threshold:.2f}')
    plt.legend()
    plt.grid(); plt.savefig(os.path.join(args.outdir,"precision_curve.png")); plt.close()

    # recall curve
    plt.figure()
    plt.plot(df["threshold"], df["recall"])
    plt.xlabel("Threshold"); plt.ylabel("Recall"); plt.title("Recall Curve")
    plt.xscale("log"); plt.ylim(0,1.05)
    plt.axvline(x=max_f1_threshold, color='r', linestyle='--', label=f'Max F1 = {max_f1:.2f} at Threshold = {max_f1_threshold:.2f}')
    plt.legend()
    plt.grid(); plt.savefig(os.path.join(args.outdir,"recall_curve.png")); plt.close()

    # PR curve
    plt.figure()
    plt.plot(df["recall"], df["precision"])
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
    plt.grid(); plt.savefig(os.path.join(args.outdir,"pr_curve.png")); plt.close()

    # ROC curve
    fpr = df["FP"]/(df["FP"]+df["TN"])
    tpr = df["recall"]
    plt.figure()
    plt.plot(fpr,tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.grid(); plt.savefig(os.path.join(args.outdir,"roc_curve.png")); plt.close()

    # Histograms
    plt.figure()
    bins = np.logspace(np.log10(min(snr_all[snr_all>0])), np.log10(max(snr_all)), 50)
    plt.hist(snr_pos,bins=bins,alpha=0.5,label="Pos",density=True)
    plt.hist(snr_neg,bins=bins,alpha=0.5,label="Neg",density=True)
    plt.xscale("log"); plt.yscale("log"); plt.xlabel("SNR"); plt.ylabel("Density")
    plt.legend(); plt.title("SNR Distributions")
    # plt.grid(which="both"); 
    plt.savefig(os.path.join(args.outdir,"snr_hist.png")); plt.close()

if __name__=="__main__":
    main()
