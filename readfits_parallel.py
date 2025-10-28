from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.mast import Catalogs, Observations
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from tqdm import tqdm
import csv, os, json, time, random
import numpy as np
from math import isfinite
from collections import Counter

# ----------------- Config -----------------
INPUT_FITS = 'gaiaedr3_wd_main.fits'
LOG_FILE   = 'albus/albus/data_inputs/wd_progress.json'

CSV_BRIGHT_LC = 'albus/albus/data_inputs/wd_bright_lc_summary.csv'
CSV_FAINT_LC  = 'albus/albus/data_inputs/wd_faint_lc_summary.csv'
FITS_BRIGHT_LC = 'albus/albus/data_outputs/wd_bright_lc.fits'
FITS_FAINT_LC  = 'albus/albus/data_outputs/wd_faint_lc.fits'

RADIUS_ARCSEC = 2.0
TMAG_BRIGHT_CUTOFF = 16.0  # "brighter than 16" => Tmag < 16

# Concurrency:
# On a 128-core server, try WORKERS=128, TIC_MAX=48, LC_MAX=24.
WORKERS = 96           # laptop 16–32; server 64–128
TIC_MAX = 48           # concurrent TIC queries
LC_MAX  = 24           # concurrent LC queries

RETRIES    = 3
BASE_DELAY = 0.6
SAVE_EVERY = 1         # save JSON after every N completions (1 = every result)
# ------------------------------------------

# -------- safe load/save progress ----------
def load_json(path, default):
    if not os.path.exists(path): return default
    try:
        with open(path, 'r') as f:
            s = f.read().strip()
            return json.loads(s) if s else default
    except Exception:
        ts = time.strftime('%Y%m%d-%H%M%S'); os.replace(path, f"{path}.bad-{ts}")
        return default

def save_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

# --------------- backoff ------------------
def backoff_sleep(attempt):
    time.sleep(BASE_DELAY * (2 ** attempt) * (0.7 + 0.6 * random.random()))

# --------------- semaphores ---------------
sem_tic = Semaphore(TIC_MAX)
sem_lc  = Semaphore(LC_MAX)

# --------------- helpers ------------------
def _finite_or_none(x):
    try:
        xv = float(x)
        return xv if np.isfinite(xv) else None
    except Exception:
        return None

def key_ra6dec6_from_vals(ra, dec):
    return f"{float(ra):.6f},{float(dec):.6f}"

def key_ra6dec6_from_row(row):
    return key_ra6dec6_from_vals(row['ra'], row['dec'])

def load_csv_seen(csv_path):
    """Return (seen_radec6: set[str], seen_tic: set[str]) from a CSV, tolerating whitespace headers."""
    seen_radec6, seen_tic = set(), set()
    if not os.path.exists(csv_path):
        return seen_radec6, seen_tic
    with open(csv_path, 'r', newline='') as f:
        import csv as _csv
        reader = _csv.DictReader(f)
        # normalize headers
        fieldmap = {k: k.strip() for k in reader.fieldnames}
        for raw in reader:
            row = {fieldmap.get(k, k): v for k, v in raw.items()}
            ra_s, dec_s = (row.get('ra_deg'), row.get('dec_deg'))
            tic = row.get('tic_id')
            try:
                if ra_s is not None and dec_s is not None:
                    ra = float(str(ra_s).strip()); dec = float(str(dec_s).strip())
                    seen_radec6.add(f"{ra:.6f},{dec:.6f}")
            except Exception:
                pass
            if tic is not None:
                tics = str(tic).strip()
                if tics and tics.lower() not in ('none','nan'):
                    seen_tic.add(tics)
    return seen_radec6, seen_tic


def get_tic_match(ra_deg, dec_deg):
    """
    Query TIC near (ra,dec). Prefer nearest with finite Tmag; else nearest.
    Retries with small/bigger radii to be robust.
    Returns (tic_id, tmag) where either may be None if truly not available.
    """
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    radii = [RADIUS_ARCSEC, 1.0, 4.0]  # try nominal, tighter, then looser
    for attempt in range(RETRIES):
        try:
            for r in radii:
                with sem_tic:
                    res = Catalogs.query_region(coord, radius=f"{r}arcsec", catalog="TIC")
                if res is None or len(res) == 0:
                    continue
                # choose nearest; prefer finite Tmag
                try:
                    tic_ra = np.array(res['ra'], dtype=float)
                    tic_dec = np.array(res['dec'], dtype=float)
                    tic_coords = SkyCoord(ra=tic_ra*u.deg, dec=tic_dec*u.deg)
                    sep = coord.separation(tic_coords).arcsec
                    order = np.argsort(sep)
                except Exception:
                    tic_id = res[0].get('ID', None)
                    tmag = _finite_or_none(res[0].get('Tmag', None))
                    return tic_id, tmag
                # nearest with finite Tmag
                for idx in order:
                    tmag = _finite_or_none(res[idx].get('Tmag', None))
                    if tmag is not None:
                        return res[idx].get('ID', None), tmag
                # else nearest anyway
                j = int(order[0])
                return res[j].get('ID', None), _finite_or_none(res[j].get('Tmag', None))
        except Exception:
            backoff_sleep(attempt)
    # exhausted
    return None, None

def has_tess_lightcurve(tic_id=None, ra_deg=None, dec_deg=None):
    """
    Check LC existence with retries.
    1) numeric tic_id
    2) 'TIC <id>'
    3) coordinate fallback within RADIUS_ARCSEC
    """
    for attempt in range(RETRIES):
        try:
            if tic_id:
                with sem_lc:
                    obs = Observations.query_criteria(
                        target_name=f"{tic_id}",
                        obs_collection="TESS",
                        dataproduct_type="timeseries",
                    )
                if obs is not None and len(obs) > 0:
                    return True
                with sem_lc:
                    obs = Observations.query_criteria(
                        target_name=f"TIC {tic_id}",
                        obs_collection="TESS",
                        dataproduct_type="timeseries",
                    )
                if obs is not None and len(obs) > 0:
                    return True
            # coord fallback
            coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
            with sem_lc:
                obs = Observations.query_criteria(
                    coordinates=coord,
                    radius=f"{RADIUS_ARCSEC} arcsec",
                    obs_collection="TESS",
                    dataproduct_type="timeseries",
                )
            return (obs is not None) and (len(obs) > 0)
        except Exception:
            backoff_sleep(attempt)
    return False

# --------------- main ---------------------
if __name__ == "__main__":
    # Load GAIA and apply static cuts
    with fits.open(INPUT_FITS) as hdul:
        data = hdul[1].data
    names = data.names
    have_source_id = 'source_id' in names

    pre_cut = data[(data['Pwd'] > 0.9) & (data['bright_N_flag'] == 0)]
    print(f"After static cuts: {len(pre_cut)} rows")

    # Load progress & make skip set
    progress_log = load_json(LOG_FILE, {})
    done_keys = set(progress_log.keys())  # these should already be 6-dp strings in your JSON


    # Ensure CSVs exist with headers
    if not os.path.exists(CSV_BRIGHT_LC):
        with open(CSV_BRIGHT_LC, 'w', newline='') as f:
            csv.writer(f).writerow(['ra_deg', 'dec_deg', 'tic_id', 'Tmag'])
    if not os.path.exists(CSV_FAINT_LC):
        with open(CSV_FAINT_LC, 'w', newline='') as f:
            csv.writer(f).writerow(['ra_deg', 'dec_deg', 'tic_id', 'Tmag'])

    def make_key(cand):
        # Canonical progress key = 6-dp RA/DEC string to match existing JSON
        return key_ra6dec6_from_vals(float(cand['ra']), float(cand['dec']))

    seen_b_radec6, seen_b_tic = load_csv_seen(CSV_BRIGHT_LC)
    seen_f_radec6, seen_f_tic = load_csv_seen(CSV_FAINT_LC)
    seen_csv_radec6 = seen_b_radec6 | seen_f_radec6
    seen_csv_tic    = seen_b_tic    | seen_f_tic

    # Build worklist
    work = []
    key_to_row = {}
    for cand in pre_cut:
        k = make_key(cand)  # 6-dp
        key_to_row[k] = cand
        if k in done_keys or k in seen_csv_radec6:
            continue
        work.append(k)

    print(f"To process this run: {len(work)} rows (skipping {len(done_keys)} already logged)")

    # CSV append helpers (main thread only)
    def append_csv(category, ra_val, dec_val, tic_id, tmag):
        # Always write 6-dp to match legacy files and avoid future duplicates
        ra6 = f"{float(ra_val):.6f}"
        dec6 = f"{float(dec_val):.6f}"
        rkey = f"{ra6},{dec6}"

        if category == 'bright_lc':
            if rkey in seen_b_radec6:  # already present, skip writing
                return
            with open(CSV_BRIGHT_LC, 'a', newline='') as f:
                csv.writer(f).writerow([ra6, dec6, tic_id, tmag])
            seen_b_radec6.add(rkey)
        elif category == 'faint_lc':
            if rkey in seen_f_radec6:
                return
            with open(CSV_FAINT_LC, 'a', newline='') as f:
                csv.writer(f).writerow([ra6, dec6, tic_id, tmag])
            seen_f_radec6.add(rkey)

    # One-shot worker: TIC + LC together
    def process_one(k):
        row = key_to_row[k]
        ra = float(row['ra']); dec = float(row['dec'])
        # Resolve TIC (with retries)
        tic_id, tmag = get_tic_match(ra, dec)
        if tic_id is None:
            # true empty -> mark as 'no_tic_or_tmag'
            return (k, 'no_tic_or_tmag', None, None)
        # Check LC (with retries)
        lc = has_tess_lightcurve(tic_id=str(tic_id), ra_deg=ra, dec_deg=dec)
        if lc and (tmag is not None) and (tmag < TMAG_BRIGHT_CUTOFF):
            category = 'bright_lc'
        elif lc and (tmag is not None) and (tmag >= TMAG_BRIGHT_CUTOFF):
            category = 'faint_lc'
        elif lc and (tmag is None):
            category = 'no_lc'   # can’t compare brightness without Tmag
        else:
            category = 'no_lc'
        return (k, category, str(tic_id), (float(tmag) if tmag is not None else None))

    processed_since_save = 0
    diag = Counter()

    # Execute with thread pool; write out each completion immediately in main thread
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(process_one, k) for k in work]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="TIC+LC per target"):
            try:
                k, category, tic_id, tmag = fut.result()
            except Exception:
                # don’t log; leaving it out means it will be retried on next run
                continue

            if k in progress_log:
                continue  # another run may have completed it already

            # Persist JSON (checkpoint)
            if category == 'no_tic_or_tmag':
                progress_log[k] = {'status': 'done', 'category': category, 'tic_id': None, 'Tmag': None}
                diag['no_tic_or_tmag'] += 1
            else:
                progress_log[k] = {'status': 'done', 'category': category, 'tic_id': tic_id, 'Tmag': tmag}
                if category == 'bright_lc': diag['bright_lc'] += 1
                elif category == 'faint_lc': diag['faint_lc'] += 1
                else: diag['no_lc'] += 1

                # Immediately append to CSV if applicable
                if category in ('bright_lc', 'faint_lc'):
                    row = key_to_row[k]
                    append_csv(category, row['ra'], row['dec'], tic_id, tmag)

            processed_since_save += 1
            if processed_since_save >= SAVE_EVERY:
                save_json(LOG_FILE, progress_log)
                processed_since_save = 0

    # Final save
    save_json(LOG_FILE, progress_log)

    # Rebuild FITS from full log so outputs reflect cumulative total
    def build_subset(cat):
        return [key_to_row[k] for k, e in progress_log.items()
                if e.get('status') == 'done' and e.get('category') == cat and k in key_to_row]

    bright_full = build_subset('bright_lc')
    faint_full  = build_subset('faint_lc')

    if len(bright_full) > 0:
        Table(bright_full, names=names).write(FITS_BRIGHT_LC, overwrite=True)
        print(f"Wrote {len(bright_full)} rows to {FITS_BRIGHT_LC}")
    else:
        print("No BRIGHT (Tmag < 16) stars with TESS lightcurves yet.")

    if len(faint_full) > 0:
        Table(faint_full, names=names).write(FITS_FAINT_LC, overwrite=True)
        print(f"Wrote {len(faint_full)} rows to {FITS_FAINT_LC}")
    else:
        print("No FAINT (Tmag ≥ 16) stars with TESS lightcurves yet.")

    print("Diag:", dict(diag))
    print("Checkpoint:", LOG_FILE)
    print("CSVs:", CSV_BRIGHT_LC, CSV_FAINT_LC)