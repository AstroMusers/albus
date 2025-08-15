import random
import pandas as pd
import csv
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np
from preprocess import preprocess
from injections import inject_transit
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSTestOutputs
from BLStests import test_period, test_duration, test_depth, test_v_shape, test_snr, test_out_of_transit_variability, create_transit_mask_manual

df = pd.read_csv('tess_targets_data.csv')
# output_file = 'Pipeline/injected_transits_datapoints.csv'

# out = pd.read_csv('Pipeline/injected_transits_datapoints.csv')
# lc = None

lc = None

# Find random WD lightcurve
while lc is None:
    # rand = random.randint(1, 1290)
    rand = 100
    print(rand)
    tic_id = int(df['Target ID'][rand])
    try: lc = preprocess(tic_id, TICID=True)
    except: pass

r_s = np.cbrt((df['MassH'][rand]*1.989*10**30)/
                (10**9*(4/3 * np.pi)))        # Assumes density of 10^9 kg/m^3, need citation
e_r_s = r_s * 1/3 * df['E_MassH'][rand]/df['MassH'][rand]

r_p = 1                         # Range of planet radii from 0.01 to 1

rho_p = 1330                                # Density of planet in kg/m^3, need citation 
roche = np.cbrt((3/2)*np.pi * (df['MassH'][rand]*1.989*10**30)/(rho_p))
a = 3*roche
period = np.sqrt((4*np.pi**2*a**3)/(6.67*10**-11*(df['MassH'][rand]*1.989*10**30))) / (24*3600)  # Orbital period in days

# period = 1+(10/res)*j                       # Range of periods from 1 to 10 days
# a = calc_a(0.6, period)/6.957*10**8         # Semi-major axis in meters

inc_min = np.arccos((0.01+r_p)/a)/np.pi*180   # Minimum transit inclination in degrees
inc = 90                                       # Inclination from 90 to i_min degrees

print(f"r_s: {r_s/6.957e+8} solar radii, e_r_s: {e_r_s/6.957e+8}, r_p: {r_p}, a: {a/1.496e+11} au, period: {period}, inc: {inc}")

# Inject transit
inj = inject_transit(tic_id, lc, lc['time'].value,
                radius_star = r_s / 6.957e+8, 
                mass_star = df['MassH'][rand], 
                radius_planet = r_p * 0.01, 
                albedo_planet=0.1, 
                period=period,
                inclination=inc
                )

print(f"Injected light curve: {inj['flux'].value[:10]}...")  # Print first 10 flux values for verification

# Run BLSØ
results = BLSfit(inj)
high_periods, high_powers, best_period, t0, duration = BLSResults(results, plot='show')
print(f"Best period: {best_period}, t0: {t0}, duration: {duration}")
for period in high_periods: 
    FoldedLC(inj, period, t0, plot='show', bin=False)                
    # Run tests
    depth = test_depth(inj['time'],
                    inj['flux'],
                    create_transit_mask_manual(inj['time'], period, t0, duration
                    ))
    vshape = test_v_shape(inj['time'],
                        inj['flux'],
                        create_transit_mask_manual(inj['time'], period, t0, duration
                        ))
    snr = test_snr(inj['flux'], create_transit_mask_manual(inj['time'], period, t0, duration))
    oot_variability = test_out_of_transit_variability(inj['flux'], create_transit_mask_manual(inj['time'], period, t0, duration))
    
    print(f"Period: {period}, Depth: {depth}, V-Shape: {vshape}, SNR: {snr}, OOT Variability: {oot_variability}")

print('done!')