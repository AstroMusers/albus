import random
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
import batman
import os
from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import inject_transit, generate_lightcurve
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSTestOutputs
from BLStests import test_period, test_duration, test_depth, test_v_shape, test_snr, test_out_of_transit_variability, create_transit_mask_manual
from injections import calc_a
from main_deleteme import fit_fold_and_test
from readfits import get_tic_match, has_tess_lightcurve
from astroquery.mast import Observations

print("Testing get_tic_match function:")

TESS_MAG_LIMIT = 16.0

ra = 359.389245402092 
dec = -14.902684073316706
key = f"{ra:.6f},{dec:.6f}"

tic_id, tmag = get_tic_match(ra, dec)
print(f"TIC ID: {tic_id}, Tmag: {tmag}")

print("Testing has_tess_lightcurve function:")
if tic_id is not None:
    obs = Observations.query_criteria(
        target_name=f"{tic_id}",
        obs_collection="TESS",
        dataproduct_type="TIMESERIES"
    )
    print(f"TESS lightcurve exists: {obs is not None and len(obs) > 0}")

else:
    print("No TIC ID found; cannot check for TESS lightcurve.")