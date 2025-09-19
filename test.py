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

df = pd.read_csv('data_inputs/tess_targets_data.csv')
lc = None
while lc is None:
    # rand = random.randint(1, 1290)
    # tic_id = int(df['Target ID'][rand])
    tic_id = 320332794
    print(tic_id)
    try: lc = preprocess(tic_id, TICID=True)
    except: pass

inj = inject_transit(tic_id, lc, lc['time'].value,
                    radius_star = 0.1,   # radius of white dwarf in Solar radii
                    mass_star = 0.6,              # mass of white dwarf in Solar masses
                    radius_planet = 0.1,     # radius of planet in Solar radii
                    albedo_planet=0.1, 
                    period=4,
                    inclination=90,
                    ID=999,
                    #a=a / (r_s / 6.957e+8) # Semi-major axis in Solar radii. DO NOT UNCOMMENT. BUGGED!
        )

fit_fold_and_test(inj, folder='presentation_plots/BLS', output_file='inj_output_file', Injected=True)