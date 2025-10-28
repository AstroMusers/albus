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
# from main_testing import fit_fold_and_test
from readfits import get_tic_match, has_tess_lightcurve
from astroquery.mast import Observations

print("Testing get_tic_match function:")

TESS_MAG_LIMIT = 16.0

ra = 351.547491
dec = -27.246801
key = f"{ra:.6f},{dec:.6f}"

tic_id, tmag = get_tic_match(ra, dec)
print(f"RA: {ra}, Dec: {dec} => TIC ID: {tic_id}, Tmag: {tmag}")