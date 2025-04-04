import random
import pandas as pd
import csv
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import inject_transit
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSTestOutputs
from tests import test_period, test_duration, test_depth, test_v_shape, test_snr, test_out_of_transit_variability, create_transit_mask_manual

df = pd.read_csv('tess_targets_data.csv')
# output_file = 'Pipeline/injected_transits_datapoints.csv'

# out = pd.read_csv('Pipeline/injected_transits_datapoints.csv')
# lc = None



lc = None  # Reset lc to None at the start of the iteration

# Find random WD lightcurve
while lc is None:
    # rand = random.randint(1, 1291)
    rand = 332
    print(rand)
    tic_id = int(df['Target ID'][rand])
    try: lc = preprocess(tic_id, TICID=True)
    except: pass

# Inject transit
inj = inject_transit("ID", tic_id, lc, lc['time'].value,
                radius_star = 0.01, 
                mass_star = 0.6,
                radius_planet = 1, 
                luminosity_star=0.001,
                albedo_planet=0.1, 
                period=5, 
                inclination=90)

# inj = lc

plt.scatter(inj['time'].value, inj['flux'].value, s=1, color='blue', label='Injected Light Curve')
plt.show()
plt.close()

# Run BLS
results = BLSfit(inj)
high_periods, high_powers, best_period, t0, duration = BLSResults(results, ID="ID", plot='show')
print(f"High Periods: {high_periods}")
FoldedLC(inj, best_period, t0, ID="ID", plot='show', bin=True)
print(f"Best period: {best_period}")

# Run tests
depth = test_depth(inj['time'],
                   inj['flux'],
                   create_transit_mask_manual(inj['time'], best_period, t0, duration
                   ))
print(f"Depth: {depth}")
vshape = test_v_shape(inj['time'],
                      inj['flux'],
                      create_transit_mask_manual(inj['time'], best_period, t0, duration
                      ))
print(f"V-shape: {vshape}")
snr = test_snr(inj['flux'],
               create_transit_mask_manual(inj['time'], best_period, t0, duration
               ))
print(f"SNR: {snr}")
oot_var = test_out_of_transit_variability(inj['flux'],
                                          create_transit_mask_manual(inj['time'], best_period, t0, duration
                                          ))
print(f"Out of Transit Variability: {oot_var}")