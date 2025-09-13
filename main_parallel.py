# parallel_pipeline.py
import os
os.environ["MPLBACKEND"] = "Agg"
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")
os.nice(19)

import csv
import re
import gc
import time
import random
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from preprocess import preprocess
from injections import inject_transit
from BLSFit import BLSfit, BLSResults, FoldedLC
from BLStests import test_depth, test_v_shape, test_snr, test_out_of_transit_variability

MAX_CORES = 32                        # hard cap
N_SAMPLES = 999999                      # how many samples to generate this run
TESS_CSV = 'data_inputs/tess_targets_data.csv'

INJ_OUT = 'data_outputs/injected_transits_output6.csv'
NONINJ_OUT = 'data_outputs/noninjected_transits_output6.csv'

PLOT_DIR_INJ = '../../WD_Plots/Injected'
PLOT_DIR_NON = '../../WD_Plots/Noninjected'

# physics
G = 6.67e-11

def ensure_csv_with_header(path, header):
    is_new = not os.path.exists(path)
    if is_new:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

# Outputs
INJ_HEADER = [
    "ID","tic_id","r_s","e_r_s","r_p","a","P_days","inc",
    "period","duration","vshape","median","mean","max_depth",
    "oot_variability","snr"
]
NON_HEADER = INJ_HEADER[:]

def _to_float(x):
    try:
        if hasattr(x, "value"):
            x = x.value
        if hasattr(x, "item"):
            x = x.item()
        return float(x)
    except Exception:
        import numpy as _np
        try:
            return float(_np.asarray(x).ravel()[0])
        except Exception:
            return float("nan")


def log(id_, msg):
    # print(f"[{time.strftime('%H:%M:%S')}] [{id_}] {msg}", flush=True)
    return

def load_existing_ids_from_csv(path):
    ids = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return ids
    try:
        df = pd.read_csv(path, usecols=['ID'])
        ids = set(pd.to_numeric(df['ID'], errors='coerce').dropna().astype(int).tolist())
    except Exception:
        # file might be mid-write/corrupted; best-effort only
        pass
    return ids

def load_existing_ids_from_plots(*dirs):
    ids = set()
    pat = re.compile(r"ID_(\d{6})_")
    for d in dirs:
        if not os.path.isdir(d):
            continue
        try:
            for name in os.listdir(d):
                m = pat.search(name)
                if m:
                    ids.add(int(m.group(1)))
        except Exception:
            pass
    return ids

def compute_resume_state():
    # Read IDs that already exist per CSV
    seen_non = load_existing_ids_from_csv(NONINJ_OUT)
    seen_inj = load_existing_ids_from_csv(INJ_OUT)

    # Also consider plot folders (per type) for resume safety
    seen_non |= load_existing_ids_from_plots(PLOT_DIR_NON)
    seen_inj |= load_existing_ids_from_plots(PLOT_DIR_INJ)

    # Next ID should advance beyond anything seen in either set
    union = seen_non | seen_inj
    next_id = (max(union) + 1) if union else 0
    return next_id, seen_non, seen_inj

def sample_power_law(min_val, max_val, alpha):
    """
    Draw one sample from PDF f(x) ∝ x^{-alpha}, on [min_val, max_val].
    """
    u = np.random.rand()
    if alpha == 1.0:
        return min_val * (max_val / min_val) ** u
    pow_ = 1.0 - alpha
    a = min_val**pow_
    b = max_val**pow_
    return (a + (b - a) * u) ** (1.0 / pow_)

def find_light_curve(df, max_tries=10, rng=None):
    """
    Randomly pick a TIC from df and try to preprocess. Return tuple or None.
    """
    rng = rng or random
    for _ in range(max_tries):
        idx = rng.randint(0, len(df) - 1)
        massH = float(df['MassH'].iloc[idx])
        tic_id = int(df['Target ID'].iloc[idx])
        m_s = float(massH * 1.989e30)
        r_s = np.cbrt(m_s / (1e9 * (4/3 * np.pi)))  # meters
        e_r_s = r_s * (1/3) * df['E_MassH'].iloc[idx] / df['MassH'].iloc[idx]

        try:
            lc = preprocess(tic_id, TICID=True)
            return tic_id, lc, massH, m_s, r_s, e_r_s
        except Exception:
            continue
    return None

def fit_fold_and_test(lc, folder, ID, tic_id, r_s, e_r_s, r_p, a, P_days, inc):
    """
    Runs BLS, folds on peaks, plots, runs tests, and returns list of rows.
    Each row matches *_HEADER schema.
    """
    os.makedirs(folder, exist_ok=True)
    results = BLSfit(lc)
    high_periods, high_powers, best_period, t0, duration = BLSResults(results, plot='save', folder=folder, ID=ID)
    plt.close('all')

    periods_to_check = list(high_periods) if (high_periods is not None and len(high_periods)) else [best_period]

    rows = []
    for period in periods_to_check:
        folded_lc = FoldedLC(
            lc, period, t0, ID=ID, plot='save', folder=folder,
            bin=False, output=True
        )

        # Transit windows & plotting
        transit_mask = np.abs(folded_lc['time'].value) < 0.6 * duration.value

        # plt.scatter(folded_lc['time'].value, folded_lc['flux'].value, s=1, label='Folded LC')
        # plt.scatter(folded_lc[transit_mask]['time'].value, folded_lc[transit_mask]['flux'].value, s=1, label='Transit')

        oot_variability = test_out_of_transit_variability(folded_lc['flux'], transit_mask)
        transit_mask_sig = transit_mask & (folded_lc['flux'].value < (1 - 3*oot_variability))
        # plt.scatter(folded_lc[transit_mask_sig]['time'].value, folded_lc[transit_mask_sig]['flux'].value, s=5, label='>3 Sigma Points')

        # for k in (1,2,3):
        #     plt.axhline(1 - k*oot_variability, linestyle='--')
        # plt.xlabel('Phase [JD]')
        # plt.ylabel('Normalized Flux')
        # plt.title(f'ID {ID} Folded LC @ Period = {round(period,3)} d')
        # plt.savefig(f'{folder}/ID_{ID}_Folded_LC_Period_{round(period,3)}.png')
        plt.close('all')

        # Tests 
        try:
            median, mean, max_depth = test_depth(folded_lc['time'], folded_lc['flux'], transit_mask_sig)
        except Exception:
            median = mean = max_depth = np.nan
        try:
            vshape = test_v_shape(folded_lc['time'], folded_lc['flux'], transit_mask)
        except Exception:
            vshape = np.nan
        try:
            snr = test_snr(folded_lc['flux'], transit_mask_sig)
        except Exception:
            snr = np.nan

        rows.append([
            str(ID), int(tic_id), _to_float(r_s), _to_float(e_r_s), _to_float(r_p),
            _to_float(a), _to_float(P_days), _to_float(inc),
            _to_float(period), _to_float(duration), _to_float(vshape),
            _to_float(median), _to_float(mean), _to_float(max_depth),
            _to_float(oot_variability), _to_float(snr)
        ])

    return rows

def worker(task):
    """
    One full sample pipeline, executed in a separate process.

    task = {
        'ID': '000123',
        'seed': 12345,
        'df_path': TESS_CSV
    }
    Returns: {'inj_rows': [...], 'non_rows': [...]}
    """
    ID = task['ID']
    log(ID, "start")
    
    # Isolate RNG for reproducibility per task
    seed = task['seed']
    np.random.seed(seed)
    random.seed(seed)

    ID = task['ID']
    df_path = task['df_path']

    # Load df locally in each process (small table; avoids pickling large objects)
    df = pd.read_csv(df_path)

    # 1) pick a star + preprocess LC
    log(ID, "preprocess: begin find_light_curve")
    star = find_light_curve(df, max_tries=10)
    if star is None:
        return {'inj_rows': [], 'non_rows': []}  # nothing to record

    tic_id, lc, massH, m_s, r_s, e_r_s = star
    log(ID, f"preprocess: ok (TIC {tic_id})")

    # 2) sample planet params
    r_p = float(sample_power_law(0.5, 5, 1.5))  # Earth radii
    rho = 1186*r_p**0.4483 if r_p < 2.5 else 2296*r_p**-1.413  # kg/m^3

    roche = np.cbrt((3/2) * np.pi * m_s / rho)
    a_min = _to_float(0.5 * roche)
    a_max = _to_float(10.5 * roche)

    # Bail out if bounds are invalid
    if not (np.isfinite(a_min) and np.isfinite(a_max) and a_min > 0 and a_max > a_min):
        return {'inj_rows': [], 'non_rows': [], 'error': f'Bad a-bounds: a_min={a_min}, a_max={a_max}, rho={rho}, m_s={m_s}'}


    a = None
    P_days = None
    for _ in range(10):
        try:
            a_try = float(np.random.uniform(a_min, a_max))
        except (OverflowError, ValueError):  # ultra-defensive
            break  # will hit the 'a is None' path below

        P_try = float(np.sqrt((4*np.pi**2 * a_try**3) / (G * m_s)) / (24*3600))
        if np.isfinite(P_try) and P_try <= 15:
            a, P_days = a_try, P_try
            break

    if a is None:
        return {'inj_rows': [], 'non_rows': [], 'error': 'Could not find a with P<=15 d'}

    x = np.clip((0.01 + r_p)/a, -1.0, 1.0)
    inc_min = np.degrees(np.arccos(x))
    inc = float(np.random.uniform(inc_min, 90.0))

    err = None
    inj_rows = []
    non_rows = []

    try:
        # 3) non-injected analysis
        log(ID, "BLS noninj: begin")
        non_rows = fit_fold_and_test(
            lc, folder=PLOT_DIR_NON,
            ID=ID, tic_id=tic_id, r_s=r_s, e_r_s=e_r_s,
            r_p=r_p, a=a, P_days=P_days, inc=inc
        )
        log(ID, f"BLS noninj: done (rows={len(non_rows)})")

        # 4) inject → analyze
        log(ID, "inject: begin")
        inj = inject_transit(
            tic_id, lc, lc['time'].value,
            radius_star = r_s / 6.957e+8,
            mass_star = massH,
            radius_planet = r_p * 0.01,
            albedo_planet=0.1,
            period=P_days,
            inclination=inc,
            ID=ID
        )
        plt.close('all')
        log(ID, "inject: done")

        log(ID, "BLS inj: begin")
        inj_rows = fit_fold_and_test(
            inj, folder=PLOT_DIR_INJ,
            ID=ID, tic_id=tic_id, r_s=r_s, e_r_s=e_r_s,
            r_p=r_p, a=a, P_days=P_days, inc=inc
        )
        log(ID, f"BLS inj: done (rows={len(inj_rows)})")
        
    except Exception as e:
        err = f"ID {task['ID']} failed: {type(e).__name__}: {e}"
        # inj_rows = inj_rows or []
        # non_rows = non_rows or []
    finally:
        plt.close('all')
        gc.collect()

    return {'inj_rows': inj_rows, 'non_rows': non_rows, 'error': err}

# ------------------------ MAIN ------------------------
def main():
    # Prepare inputs / outputs
    df = pd.read_csv(TESS_CSV)

    # Determine starting ID from existing injected file
    try:
        out = pd.read_csv(INJ_OUT)
        out_ids = set(pd.to_numeric(out['ID'], errors='coerce').dropna().astype(int))
    except FileNotFoundError:
        out_ids = set()
    next_id, seen_non, seen_inj = compute_resume_state()
    print(
        f"Writing to:\n  INJ: {os.path.abspath(INJ_OUT)}\n  NON: {os.path.abspath(NONINJ_OUT)}\n"
        f"Starting next_id={next_id} (seen_non={len(seen_non)}, seen_inj={len(seen_inj)})",
        flush=True
    )
    

    # Ensure headers exist
    ensure_csv_with_header(INJ_OUT, INJ_HEADER)
    ensure_csv_with_header(NONINJ_OUT, NON_HEADER)

    # Make plot dirs
    os.makedirs(PLOT_DIR_INJ, exist_ok=True)
    os.makedirs(PLOT_DIR_NON, exist_ok=True)

    # Build tasks with pre-assigned IDs and seeds (no race conditions)
    tasks = []
    for k in range(N_SAMPLES):
        id_int = next_id + k
        tasks.append({
            'ID': f"{id_int:06d}",
            'seed': 10_000 + id_int,   # stable per-ID RNG
            'df_path': TESS_CSV
        })

    workers = min(MAX_CORES, os.cpu_count() or MAX_CORES)
    ctx = get_context("spawn")

    # Fan-out/fan-in
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex, \
        open(INJ_OUT, 'a', newline='', buffering=1) as inj_f, \
        open(NONINJ_OUT, 'a', newline='', buffering=1) as non_f:

        inj_writer = csv.writer(inj_f)
        non_writer = csv.writer(non_f)

        futures = [ex.submit(worker, t) for t in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            res = fut.result()
            if res.get('error'):
                print(res['error'], flush=True)

            # Filter out any rows whose ID we already have (idempotent append)
            def new_non_rows(rows):
                out = []
                for row in rows:
                    try:
                        rid = int(row[0])  # ID in first column
                    except Exception:
                        continue
                    if rid not in seen_non:
                        out.append(row)
                        seen_non.add(rid)
                return out

            def new_inj_rows(rows):
                out = []
                for row in rows:
                    try:
                        rid = int(row[0])
                    except Exception:
                        continue
                    if rid not in seen_inj:
                        out.append(row)
                        seen_inj.add(rid)
                return out

            non_new = new_non_rows(res.get('non_rows', []))
            inj_new  = new_inj_rows(res.get('inj_rows', []))

            if non_new:
                non_writer.writerows(non_new)
                non_f.flush(); os.fsync(non_f.fileno())

            if inj_new:
                inj_writer.writerows(inj_new)
                inj_f.flush(); os.fsync(inj_f.fileno())

            print(f"[done] wrote noninj={len(non_new)}, inj={len(inj_new)} "
                f"(seen_non={len(seen_non)}, seen_inj={len(seen_inj)})",
                flush=True)


if __name__ == "__main__":
    if False:  # flip to True to test
        df = pd.read_csv(TESS_CSV)
        res = worker({'ID': "SMOKET", 'seed': 123, 'df_path': TESS_CSV})
        print("SMOKE:", {k: len(v) if isinstance(v, list) else v for k, v in res.items()})
    else:
        main()
