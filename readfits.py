from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.mast import Catalogs, Observations
from tdpy.verbosity import print, tqdm
import csv, os, json, time
from math import isfinite
from catalog_values import finite_or_none, CatalogQueryError
from progress_io import load_progress_json, save_progress_json, read_catalog_csv_keys, append_catalog_row, is_completed_record

# --- Parameters ---
INPUT_FITS = 'gaiaedr3_wd_main.fits'
LOG_FILE = 'albus/albus/data_inputs/wd_progress.json'

CSV_BRIGHT_LC = 'albus/albus/data_inputs/wd_bright_lc_summary.csv'
CSV_FAINT_LC  = 'albus/albus/data_inputs/wd_faint_lc_summary.csv'
FITS_BRIGHT_LC = 'albus/albus/data_inputs/wd_bright_lc.fits'
FITS_FAINT_LC  = 'albus/albus/data_inputs/wd_faint_lc.fits'

RADIUS_ARCSEC = 2.0
TMAG_BRIGHT_CUTOFF = 16.0  # "brighter than 16" => Tmag < 16

def save_progress():
    return save_progress_json(LOG_FILE, progress_log)

# --- Simple retry wrapper for network calls ---
def with_retries(func, *args, retries=3, base_delay=1.0, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            if attempt == retries - 1:
                raise CatalogQueryError(f'Catalog request failed after {retries} attempts') from error
            time.sleep(base_delay * (2 ** attempt))
    return None

# --- Helpers ---
def get_tic_match(ra_deg, dec_deg):
    """Return (tic_id, tmag) or (None, None)."""
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    res = with_retries(
        Catalogs.query_region, coord,
        radius=f"{RADIUS_ARCSEC}arcsec", catalog="TIC", retries=3
    )
    if res is None or len(res) == 0:
        return None, None
    tic_id = res[0].get('ID', None)
    tmag = finite_or_none(res[0].get('Tmag', None))
    return tic_id, tmag

def has_tess_lightcurve(tic_id=None, ra_deg=None, dec_deg=None):
    """Return True if a TESS timeseries exists."""
    if tic_id:
        obs = with_retries(
            Observations.query_criteria,
            target_name=f"{tic_id}",
            obs_collection="TESS",
            dataproduct_type="timeseries",
            retries=3
        )
    else:
        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
        obs = with_retries(
            Observations.query_criteria,
            coordinates=coord,
            radius=f"{RADIUS_ARCSEC} arcsec",
            obs_collection="TESS",
            dataproduct_type="timeseries",
            retries=3
        )
    return (obs is not None) and (len(obs) > 0)

# --- Main loop with checkpointing and categorization ---
if __name__ == "__main__":
    print(f'Reading from {INPUT_FITS}...')
    with fits.open(INPUT_FITS) as hdul:
        data = hdul[1].data
    pre_cut = data[(data['Pwd'] > 0.9) & (data['bright_N_flag'] == 0)]
    print(f"After static cuts: {len(pre_cut)} rows")
    progress_log = load_progress_json(LOG_FILE)
    csv_files = {'bright_lc': CSV_BRIGHT_LC, 'faint_lc': CSV_FAINT_LC}
    csv_seen = {}
    for category, filename in csv_files.items():
        if not os.path.exists(filename):
            print(f'Writing to {filename}...')
            with open(filename, 'w', newline='') as stream:
                csv.writer(stream).writerow(['ra_deg', 'dec_deg', 'tic_id', 'Tmag'])
        csv_seen[category] = read_catalog_csv_keys(filename)[0]
    bright_lc_rows = []
    faint_lc_rows = []

    for cand in tqdm(pre_cut, desc="Classifying (TIC + TESS)"):
        ra = float(cand['ra'])
        dec = float(cand['dec'])
        key = f"{ra:.6f},{dec:.6f}"

        # Skip if processed
        if is_completed_record(progress_log.get(key)):
            cat = progress_log[key].get('category')
            # If it was a "bright_lc" or "faint_lc", re-collect row to FITS aggregation
            if cat in ('bright_lc', 'faint_lc'):
                entry = progress_log[key]
                magnitude = finite_or_none(entry.get('Tmag'))
                if entry.get('tic_id') is not None and magnitude is not None:
                    append_catalog_row(csv_files[cat], ra, dec, entry['tic_id'], magnitude, csv_seen[cat])
                if cat == 'bright_lc':
                    bright_lc_rows.append(cand)
                else:
                    faint_lc_rows.append(cand)
            continue

        try:
            tic_id, tmag = get_tic_match(ra, dec)
            if tic_id is None or tmag is None or not isfinite(tmag):
                progress_log[key] = {
                    **progress_log.get(key, {}), 'status': 'done', 'category': 'no_tic_or_tmag',
                    'tic_id': str(tic_id) if tic_id is not None else None, 'Tmag': None,
                }
                save_progress()
                continue
            lc = has_tess_lightcurve(tic_id=tic_id, ra_deg=ra, dec_deg=dec)
        except CatalogQueryError as error:
            print(f'Catalog request for {key} failed; leaving it unfinished: {error}')
            continue

        # Categorize:
        # 1) BRIGHT + LC  => export to bright set
        # 2) FAINT  + LC  => export to faint set (separate listing)
        # 3) Otherwise    => logged only
        if lc and (tmag < TMAG_BRIGHT_CUTOFF):
            progress_log[key] = {
                **progress_log.get(key, {}),
                'status': 'done', 'category': 'bright_lc',
                'tic_id': str(tic_id), 'Tmag': float(tmag)
            }
            save_progress()
            bright_lc_rows.append(cand)
            append_catalog_row(CSV_BRIGHT_LC, ra, dec, tic_id, float(tmag), csv_seen['bright_lc'])

        elif lc and (tmag >= TMAG_BRIGHT_CUTOFF):
            progress_log[key] = {
                **progress_log.get(key, {}),
                'status': 'done', 'category': 'faint_lc',
                'tic_id': str(tic_id), 'Tmag': float(tmag)
            }
            save_progress()
            faint_lc_rows.append(cand)
            append_catalog_row(CSV_FAINT_LC, ra, dec, tic_id, float(tmag), csv_seen['faint_lc'])

        else:
            progress_log[key] = {
                **progress_log.get(key, {}),
                'status': 'done', 'category': 'no_lc',
                'tic_id': str(tic_id), 'Tmag': float(tmag)
            }
            save_progress()

    # --- Finalize FITS outputs (aggregate rows from this run and previous ones) ---
    # We want the FITS to reflect ALL processed rows in each category, including prior runs.
    # Easiest: re-scan the log and pull matching rows from pre_cut to write fresh FITS each time.
    def build_subset_from_log(category):
        selected = []
        for cand in pre_cut:
            ra = float(cand['ra'])
            dec = float(cand['dec'])
            key = f"{ra:.6f},{dec:.6f}"
            entry = progress_log.get(key)
            if is_completed_record(entry) and entry.get('category') == category:
                selected.append(cand)
        return selected

    bright_full = build_subset_from_log('bright_lc')
    faint_full  = build_subset_from_log('faint_lc')

    if len(bright_full) > 0:
        print(f'Writing to {FITS_BRIGHT_LC}...')
        Table(rows=bright_full, names=data.names).write(FITS_BRIGHT_LC, overwrite=True)
        print(f"Wrote {len(bright_full)} rows to {FITS_BRIGHT_LC}")
    else:
        print("No BRIGHT (Tmag < 16) stars with TESS lightcurves yet.")

    if len(faint_full) > 0:
        print(f'Writing to {FITS_FAINT_LC}...')
        Table(rows=faint_full, names=data.names).write(FITS_FAINT_LC, overwrite=True)
        print(f"Wrote {len(faint_full)} rows to {FITS_FAINT_LC}")
    else:
        print("No FAINT (Tmag ≥ 16) stars with TESS lightcurves yet.")

    print("Progress saved in:", LOG_FILE)
    print(f"CSVs: {CSV_BRIGHT_LC}, {CSV_FAINT_LC}")
