"""
benchmark_max_workers.py
----------------------------------
Benchmark optimal MAX_WORKERS for I/O-bound astroquery workloads.

Example usage:
  python benchmark_max_workers.py \
    --input-fits "/Users/aavikwadivkar/Documents/Exoplanets/Ampersand/gaiaedr3_wd_main.fits" \
    --sample-n 200 \
    --counts 4,8,12,16,24,32,48,64 \
    --repeats 2 \
    --out-csv "benchmark_results.csv"

Notes:
- Does not modify your main pipeline’s files.
- Requires that `process_one(ra, dec)` is in scope or imported.
"""

import argparse, csv, os, random, statistics, time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from astropy.io import fits

# ---------------------------
# If run separately, import your functions/constants:
from readfits_parallel import process_one
# ---------------------------

def _load_work(input_fits):
    """
    Apply the same filtering as in your main script.
    """
    with fits.open(input_fits) as hdul:
        data = hdul[1].data
    pre_cut = data[(data['Pwd'] > 0.9) & (data['bright_N_flag'] == 0)]
    work = [(float(c['ra']), float(c['dec'])) for c in pre_cut]
    return work

def _run_once(work_items, max_workers):
    """
    Run one benchmark iteration at a specific worker count.
    """
    t0 = time.perf_counter()
    n_ok = 0
    n_total = len(work_items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, ra, dec) for (ra, dec) in work_items]
        for fut in tqdm(as_completed(futures), total=n_total,
                        desc=f"workers={max_workers}", leave=False):
            try:
                res = fut.result()
                if isinstance(res, dict) and 'category' in res:
                    n_ok += 1
            except Exception:
                pass
    elapsed = time.perf_counter() - t0
    return elapsed, n_ok, n_total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fits", type=str, required=True)
    parser.add_argument("--sample-n", type=int, default=200)
    parser.add_argument("--counts", type=str, default="4,8,12,16,24,32,48,64")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-n", type=int, default=20)
    parser.add_argument("--cooldown-sec", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    # Prepare CSV path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = args.out_csv or f"benchmark_results_{timestamp}.csv"

    # Load sample
    all_work = _load_work(args.input_fits)
    if len(all_work) == 0:
        print("No rows after static cuts; nothing to benchmark.")
        return
    sample_n = min(args.sample_n, len(all_work))
    sample = random.sample(all_work, sample_n)

    # Warm-up
    if args.warmup_n > 0:
        warm = random.sample(sample, min(args.warmup_n, sample_n))
        print(f"Warm-up: {len(warm)} tasks @ workers=16 (not scored)...")
        _ = _run_once(warm, max_workers=16)
        print("Warm-up complete.\n")

    # Benchmark loop
    counts = [int(x.strip()) for x in args.counts.split(",") if x.strip()]
    results = []

    print(f"Benchmarking {sample_n} tasks across counts: {counts} (repeats={args.repeats})\n")

    for w in counts:
        run_times, ok_rates = [], []
        for r in range(args.repeats):
            batch = random.sample(sample, len(sample))
            print(f"Run {r+1}/{args.repeats} @ workers={w}")
            elapsed, n_ok, n_tot = _run_once(batch, max_workers=w)
            rate = n_ok / n_tot if n_tot else 0.0
            run_times.append(elapsed)
            ok_rates.append(rate)
            print(f"  elapsed={elapsed:.2f}s, ok={n_ok}/{n_tot} ({100*rate:.1f}%), "
                  f"throughput={n_ok/elapsed:.2f} ok/s")
            if args.cooldown_sec > 0 and not (w == counts[-1] and r == args.repeats - 1):
                print(f"  cooldown {args.cooldown_sec:.1f}s...")
                time.sleep(args.cooldown_sec)

        med = statistics.median(run_times)
        mean_ok = statistics.mean(ok_rates)
        median_okps = (sample_n / med) * mean_ok if med > 0 else 0.0

        results.append({
            "workers": w,
            "median_sec": med,
            "mean_ok_rate": mean_ok,
            "median_ok_per_sec": median_okps,
            "runs": run_times,
        })

    # Sort and print summary
    results_sorted = sorted(results, key=lambda d: (-d["median_ok_per_sec"], d["median_sec"]))
    print("\n=== Benchmark Summary (sorted by OK/s) ===")
    print(f"{'workers':>8}  {'median_s':>9}  {'OK_rate%':>8}  {'OK/s':>8}  {'runs (s)':>20}")
    for r in results_sorted:
        print(f"{r['workers']:>8}  {r['median_sec']:>9.2f}  {100*r['mean_ok_rate']:>8.1f}  "
              f"{r['median_ok_per_sec']:>8.2f}  {str([round(x,2) for x in r['runs']]):>20}")

    best = results_sorted[0]
    print("\nRecommended MAX_WORKERS =", best["workers"])
    print(f"(median ~{best['median_sec']:.2f}s, OK rate ~{100*best['mean_ok_rate']:.1f}%, "
          f"OK/s ~{best['median_ok_per_sec']:.2f})")

    # Write results to CSV
    fieldnames = ["workers", "median_sec", "mean_ok_rate", "median_ok_per_sec", "runs"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_sorted:
            writer.writerow({
                "workers": row["workers"],
                "median_sec": f"{row['median_sec']:.3f}",
                "mean_ok_rate": f"{row['mean_ok_rate']:.3f}",
                "median_ok_per_sec": f"{row['median_ok_per_sec']:.3f}",
                "runs": ";".join([f"{x:.2f}" for x in row["runs"]]),
            })

    print(f"\nResults saved to: {os.path.abspath(out_csv)}")

if __name__ == "__main__":
    main()
