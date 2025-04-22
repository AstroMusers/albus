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
from BLStests import test_period, test_duration, test_depth, test_v_shape, test_snr, test_out_of_transit_variability, create_transit_mask_manual
from injections import calc_a

df = pd.read_csv('tess_targets_data.csv')

lc = None  # Reset lc to None at the start of the iteration

res = 9

rads = []
periods = []
incs = []

for i in range(res):
    for j in range(res):
        for k in (range(res)):
            ID = str(i) + str(j) + str(k)
            # print(ID)
            
            r_p = ((i+1)/res)**2                        # Range of planet radii from 0.01 to 1
            period = 1+(10/res)*j                       # Range of periods from 1 to 10 days
            a = calc_a(0.6, period)/(6.957*10**8)        # Semi-major axis in meters
            inc_min = np.arccos((0.01+r_p)/a)/np.pi*180   # Minimum transit inclination in degrees
            inc = 90-(k*(90-inc_min)/res)                   # Inclination from 90 to i_min degrees
            # print(f"Radius: {r_p}, Period: {period}, Inclination: {inc}")
            rads.append(r_p)
            periods.append(period)
            incs.append(inc)

plt.hist(rads, bins=20, alpha=0.5, label='Radius')
plt.hist(periods, bins=20, alpha=0.5, label='Period')
plt.hist(incs, bins=20, alpha=0.5, label='Inclination')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Parameters')
plt.legend()
plt.show()