import random
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import inject_transit, generate_lightcurve
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSTestOutputs
from tests import test_period, test_duration, test_depth, test_v_shape, test_snr, test_out_of_transit_variability, create_transit_mask_manual

df = pd.read_csv('tess_targets_data.csv')

lc = None  # Reset lc to None at the start of the iteration

# Find random WD lightcurve
while lc is None:
    # rand = random.randint(1, 1291)
    rand = 244
    print(rand)
    tic_id = int(df['Target ID'][rand])
    try: lc = preprocess(tic_id, TICID=True)
    except: pass

# print(max(lc['time'].value))

# Inject transit
inj = inject_transit("ID", tic_id, lc, lc['time'].value,
                radius_star = 0.01, 
                mass_star = 0.6, 
                radius_planet = 1, 
                luminosity_star=0.001,
                albedo_planet=0.1, 
                period=5, 
                inclination=90)

# ttime, tflux, tduration = generate_lightcurve(
#         radius_star=0.01,            # Approx. radius of a white dwarf
#         mass_star= 0.6, # Approx. mass of white dwarf
#         radius_planet= 1,          # Radius of a typical Hot Jupiter
#         luminosity_star=0.001,       # White dwarf luminosity in Solar units
#         albedo_planet=0.1,           # Typical albedo of a gas giant
#         period=5,                    # Orbital period
#         inclination=90,              # Inclination of transit
#         time_array=lc['time'].value
#     )

plt.plot(inj['time'], inj['flux'])
plt.show()