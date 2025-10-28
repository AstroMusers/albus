from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.mast import Catalogs, Observations
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import csv, os, json, time
from math import isfinite

# ----------------- Config -----------------
INPUT_FITS = '/Users/aavikwadivkar/Documents/Exoplanets/Ampersand/gaiaedr3_wd_main.fits'
LOG_FILE = 'wd_progress.json'

CSV_BRIGHT_LC = 'wd_bright_lc_summary.csv'
CSV_FAINT_LC  = 'wd_faint_lc_summary.csv'
FITS_BRIGHT_LC = 'wd_bright_lc.fits'
FITS_FAINT_LC  = 'wd_faint_lc.fits'

RADIUS_ARCSEC = 2.0
TMAG_BRIGHT_CUTOFF = 16.0  # "brighter than 16" => Tmag < 16

MAX_WORKERS = 12            # adjust per your machine/network
RETRIES = 3
BASE_DELAY = 1.0
SAVE_EVERY = 50             # checkpoint JSON every N processed rows
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
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(base_delay * (2 ** attempt))
    return None

# --------------- helpers ------------------
def get_tic_match(ra_deg, dec_deg):
    """Return (tic_id, tmag) or (None, None)."""
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    res = with_retries(
        Catalogs.query_region, coord,
        radius=f"{RADIUS_ARCSEC}arcsec", catalog="TIC"
    )
    if res is None or len(res) == 0:
        return None, None
    tic_id = res[0].get('ID', None)
    tmag = res[0].get('Tmag', None)
    try:
        tmag = float(tmag) if tmag is not None else None
    except Exception:
        tmag = None
    return tic_id, tmag

def has_tess_lightcurve(tic_id=None, ra_deg=None, dec_deg=None):
    """Return True if a TESS timeseries exists."""
    if tic_id:
        obs = with_retries(
            Observations.query_criteria,
            target_name=f"TIC {tic_id}",
            obs_collection="TESS",
            dataproduct_type="timeseries"
        )
    else:
        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
        obs = with_retries(
            Observations.query_criteria,
            coordinates=coord,
            radius=f"{RADIUS_ARCSEC} arcsec",
            obs_collection="TESS",
            dataproduct_type="timeseries"
        )
    return (obs is not None) and (len(obs) > 0)

# Worker: pure function that queries TIC+MAST and returns a result dict
def process_one(ra, dec):
    key = f"{ra:.6f},{dec:.6f}"
    tic_id, tmag = get_tic_match(ra, dec)
    if tic_id is None or tmag is None or not isfinite(tmag):
        return {'key': key, 'category': 'no_tic_or_tmag', 'tic_id': None, 'Tmag': None}

    lc = has_tess_lightcurve(tic_id=tic_id, ra_deg=ra, dec_deg=dec)
    if lc and (tmag < TMAG_BRIGHT_CUTOFF):
        return {'key': key, 'category': 'bright_lc', 'tic_id': str(tic_id), 'Tmag': float(tmag)}
    elif lc and (tmag >= TMAG_BRIGHT_CUTOFF):
        return {'key': key, 'category': 'faint_lc',  'tic_id': str(tic_id), 'Tmag': float(tmag)}
    else:
        return {'key': key, 'category': 'no_lc',     'tic_id': str(tic_id), 'Tmag': float(tmag)}

# --------------- main ---------------------
if __name__ == "__main__":
    # Load data
    with fits.open(INPUT_FITS) as hdul:
        data = hdul[1].data

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
    for cand in pre_cut:
        ra = float(cand['ra']); dec = float(cand['dec'])
        key = f"{ra:.6f},{dec:.6f}"
        key_to_row[key] = cand
        if key in done_keys:
            continue
        work.append((ra, dec))

    print(f"To process this run: {len(work)} rows (skipping {len(done_keys)} already logged)")

    processed_since_save = 0

    # Collect categories to rebuild FITS at end
    # (We'll also append CSVs as we go)
    def append_csv(category, ra, dec, tic_id, tmag):
        if category == 'bright_lc':
            with open(CSV_BRIGHT_LC, 'a', newline='') as f:
                csv.writer(f).writerow([ra, dec, tic_id, tmag])
        elif category == 'faint_lc':
            with open(CSV_FAINT_LC, 'a', newline='') as f:
                csv.writer(f).writerow([ra, dec, tic_id, tmag])

    # Thread pool for network-bound concurrency
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one, ra, dec) for (ra, dec) in work]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Parallel TIC+MAST"):
            result = fut.result()
            key = result['key']
            if key in progress_log:
                continue  # edge case if duplicated key slipped in

            # Update log in main thread
            progress_log[key] = {'status': 'done',
                                 'category': result['category'],
                                 'tic_id': result['tic_id'],
                                 'Tmag': result['Tmag']}
            processed_since_save += 1

            # Stream CSV appends in main thread
            if result['category'] in ('bright_lc', 'faint_lc'):
                ra_str, dec_str = key.split(',')
                append_csv(result['category'], float(ra_str), float(dec_str),
                           result['tic_id'], result['Tmag'])

            # Periodic checkpoint
            if processed_since_save >= SAVE_EVERY:
                save_progress(progress_log)
                processed_since_save = 0

    # Final save
    save_progress(progress_log)

    # Rebuild FITS from full log so outputs always reflect cumulative total
    def build_subset_from_log(category):
        selected = []
        for key, entry in progress_log.items():
            if entry.get('status') == 'done' and entry.get('category') == category:
                cand = key_to_row.get(key)
                if cand is not None:
                    selected.append(cand)
        return selected

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
2