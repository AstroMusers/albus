from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.mast import Catalogs, Observations
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import csv, os, json, time, random
from math import isfinite
import numpy as np
from collections import Counter

# ----------------- Config -----------------
INPUT_FITS = '/Users/aavikwadivkar/Documents/Exoplanets/Ampersand/gaiaedr3_wd_main.fits'
LOG_FILE = 'data_inputs/wd_progress.json'

CSV_BRIGHT_LC = 'data_inputs/wd_bright_lc_summary.csv'
CSV_FAINT_LC  = 'data_inputs/wd_faint_lc_summary.csv'
FITS_BRIGHT_LC = 'data_outputs/wd_bright_lc.fits'
FITS_FAINT_LC  = 'data_outputs/wd_faint_lc.fits'

RADIUS_ARCSEC = 2.0
TMAG_BRIGHT_CUTOFF = 16.0  # "brighter than 16" => Tmag < 16

MAX_WORKERS = 16           
RETRIES = 3
BASE_DELAY = 1.0
SAVE_EVERY = 50            # checkpoint JSON every N processed rows
# ------------------------------------------

# -------- safe load/save progress ----------
def load_progress(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            s = f.read().strip()
            if not s:
                raise ValueError("empty progress file")
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError("not a dict")
            return obj
    except Exception as e:
        ts = time.strftime('%Y%m%d-%H%M%S')
        bad = f"{path}.bad-{ts}"
        try:
            os.replace(path, bad)
            print(f"[warn] Invalid progress file ({e}). Backed up to {bad}. Starting fresh.")
        except Exception:
            print(f"[warn] Invalid progress file ({e}). Could not back up; starting fresh.")
        return {}

def save_progress(progress_log):
    tmp = LOG_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(progress_log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, LOG_FILE)

# --------------- retries ------------------
def with_retries(func, *args, retries=RETRIES, base_delay=BASE_DELAY, **kwargs):
    """Retry with exponential backoff + jitter."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception:
            sleep_s = base_delay * (2 ** attempt) * (0.7 + 0.6 * random.random())
            time.sleep(sleep_s)
    # final grace attempt after a longer sleep
    time.sleep(base_delay * (2 ** retries))
    try:
        return func(*args, **kwargs)
    except Exception:
        return None

# --------------- helpers ------------------
def _finite_or_none(x):
    try:
        if x is np.ma.masked or x is None:
            return None
        xv = float(x)
        return xv if np.isfinite(xv) else None
    except Exception:
        return None

def get_tic_match(ra_deg, dec_deg, radius_arcsec=2.0):
    """
    Robust TIC match near (ra, dec).
    - query within radius, pick nearest match by separation
    - prefer entries with finite Tmag; if none finite, still return TIC ID with Tmag=None
    - if no rows, retry once with 4" radius
    Returns (tic_id, tmag)
    """
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)

    def _query(radius_as):
        return with_retries(
            Catalogs.query_region, coord,
            radius=f"{radius_as}arcsec", catalog="TIC"
        )

    for radius_try in (radius_arcsec, 4.0):
        res = _query(radius_try)
        if res is None or len(res) == 0:
            continue

        try:
            tic_ra = np.array(res['ra'], dtype=float)
            tic_dec = np.array(res['dec'], dtype=float)
            tic_coords = SkyCoord(ra=tic_ra*u.deg, dec=tic_dec*u.deg)
            sep = coord.separation(tic_coords).arcsec
            order = np.argsort(sep)
        except Exception:
            tic_id = res[0].get('ID', None)
            tmag = _finite_or_none(res[0].get('Tmag', None))
            return (tic_id, tmag)

        # nearest with finite Tmag
        for idx in order:
            tmag = _finite_or_none(res[idx].get('Tmag', None))
            if tmag is not None:
                tic_id = res[idx].get('ID', None)
                return (tic_id, tmag)

        # else nearest anyway
        nearest = int(order[0])
        tic_id = res[nearest].get('ID', None)
        tmag = _finite_or_none(res[nearest].get('Tmag', None))  # likely None
        return (tic_id, tmag)

    return (None, None)

def has_tess_lightcurve(tic_id=None, ra_deg=None, dec_deg=None):
    """
    Return True if a TESS timeseries exists.
    Try, in order:
      1) bare numeric TIC id
      2) 'TIC <id>'
      3) coordinate-based within RADIUS_ARCSEC
    """
    # (1) bare numeric
    if tic_id:
        obs = with_retries(
            Observations.query_criteria,
            target_name=f"{tic_id}",
            obs_collection="TESS",
            dataproduct_type="timeseries",
        )
        if obs is not None and len(obs) > 0:
            return True

        # (2) prefixed
        obs = with_retries(
            Observations.query_criteria,
            target_name=f"TIC {tic_id}",
            obs_collection="TESS",
            dataproduct_type="timeseries",
        )
        if obs is not None and len(obs) > 0:
            return True

    # (3) coordinate fallback
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    obs = with_retries(
        Observations.query_criteria,
        coordinates=coord,
        radius=f"{RADIUS_ARCSEC} arcsec",
        obs_collection="TESS",
        dataproduct_type="timeseries",
    )
    return (obs is not None) and (len(obs) > 0)

# --------------- main ---------------------
if __name__ == "__main__":
    # Load data
    with fits.open(INPUT_FITS) as hdul:
        data = hdul[1].data
    colnames = data.names
    have_source_id = 'source_id' in colnames

    pre_cut = data[(data['Pwd'] > 0.9) & (data['bright_N_flag'] == 0)]
    print(f"After static cuts: {len(pre_cut)} rows")

    # Load progress, build skip set
    progress_log = load_progress(LOG_FILE)
    done_keys = set(progress_log.keys())

    # Prepare CSV headers if first run
    if not os.path.exists(CSV_BRIGHT_LC):
        with open(CSV_BRIGHT_LC, 'w', newline='') as f:
            csv.writer(f).writerow(['ra_deg', 'dec_deg', 'tic_id', 'Tmag'])
    if not os.path.exists(CSV_FAINT_LC):
        with open(CSV_FAINT_LC, 'w', newline='') as f:
            csv.writer(f).writerow(['ra_deg', 'dec_deg', 'tic_id', 'Tmag'])

    # Build work list and key map back to rows for FITS aggregation
    work = []
    key_to_row = {}

    def make_key(cand):
        if have_source_id:
            return str(cand['source_id'])
        # use full-precision floats to avoid rounding collisions
        return f"{repr(float(cand['ra']))},{repr(float(cand['dec']))}"

    for cand in pre_cut:
        key = make_key(cand)
        key_to_row[key] = cand
        if key in done_keys:
            continue
        ra = float(cand['ra']); dec = float(cand['dec'])
        work.append((key, ra, dec))

    print(f"To process this run: {len(work)} rows (skipping {len(done_keys)} already logged)")

    processed_since_save = 0
    stats = Counter()

    def append_csv(category, ra_val, dec_val, tic_id, tmag):
        # write the original GAIA values (no rounding)
        if category == 'bright_lc':
            with open(CSV_BRIGHT_LC, 'a', newline='') as f:
                csv.writer(f).writerow([ra_val, dec_val, tic_id, tmag])
        elif category == 'faint_lc':
            with open(CSV_FAINT_LC, 'a', newline='') as f:
                csv.writer(f).writerow([ra_val, dec_val, tic_id, tmag])

    # Worker
    def process_one(ra, dec):
        tic_id, tmag = get_tic_match(ra, dec)
        if tic_id is None:
            stats['tic_query_empty'] += 1
            return ('no_tic_or_tmag', None, None)
        if tmag is None:
            stats['tic_no_tmag'] += 1
        lc = has_tess_lightcurve(tic_id=tic_id, ra_deg=ra, dec_deg=dec)
        if lc and (tmag is not None) and (tmag < TMAG_BRIGHT_CUTOFF):
            return ('bright_lc', tic_id, tmag)
        elif lc and (tmag is not None) and (tmag >= TMAG_BRIGHT_CUTOFF):
            return ('faint_lc', tic_id, tmag)
        elif lc and (tmag is None):
            stats['lc_but_no_tmag'] += 1
            # keep it out of bright/faint sets since we can't compare to cutoff
            return ('no_lc', tic_id, None)
        else:
            return ('no_lc', tic_id, tmag if tmag is not None else None)

    # Thread pool for network-bound concurrency
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_one, ra, dec): (key, ra, dec) for (key, ra, dec) in work}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Parallel TIC+MAST"):
            key, ra, dec = futures[fut]
            category, tic_id, tmag = fut.result()

            if key in progress_log:
                continue

            progress_log[key] = {'status': 'done',
                                 'category': category,
                                 'tic_id': (str(tic_id) if tic_id is not None else None),
                                 'Tmag': (float(tmag) if tmag is not None else None)}
            processed_since_save += 1

            if category in ('bright_lc', 'faint_lc'):
                # write exact GAIA RA/DEC values as stored in the FITS row
                cand = key_to_row[key]
                append_csv(category, cand['ra'], cand['dec'], tic_id, tmag)

            if processed_since_save >= SAVE_EVERY:
                save_progress(progress_log)
                processed_since_save = 0

    # Final save
    save_progress(progress_log)

    # Rebuild FITS from full log so outputs always reflect cumulative total
    def build_subset_from_log(category):
        return [key_to_row[k] for k, e in progress_log.items()
                if e.get('status') == 'done' and e.get('category') == category and k in key_to_row]

    bright_full = build_subset_from_log('bright_lc')
    faint_full  = build_subset_from_log('faint_lc')

    if len(bright_full) > 0:
        Table(bright_full, names=data.names).write(FITS_BRIGHT_LC, overwrite=True)
        print(f"Wrote {len(bright_full)} rows to {FITS_BRIGHT_LC}")
    else:
        print("No BRIGHT (Tmag < 16) stars with TESS lightcurves yet.")

    if len(faint_full) > 0:
        Table(faint_full, names=data.names).write(FITS_FAINT_LC, overwrite=True)
        print(f"Wrote {len(faint_full)} rows to {FITS_FAINT_LC}")
    else:
        print("No FAINT (Tmag ≥ 16) stars with TESS lightcurves yet.")

    print("Progress saved in:", LOG_FILE)
    print(f"CSVs: {CSV_BRIGHT_LC}, {CSV_FAINT_LC}")
    print("Diag:", dict(stats))