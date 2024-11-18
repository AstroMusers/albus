import random
import pandas as pd
import csv
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import inject_transit
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSOutput

df = pd.read_csv('tess_targets_data.csv')
output_file = 'Pipeline/injected_transits.csv'

# with open(output_file, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['ID', 'Periods', 'Powers'])
#         f.close()

out = pd.read_csv('Pipeline/injected_transits.csv')
lc = None

res = 5

for i in range(res):
    for j in range(res):
        for k in tqdm(range(res)):
            here = 0
            ID = str(i) + str(j) + str(k)
            print(ID)
            if int(ID) not in out['ID'].values:  # Ensure check is against column values
                lc = None  # Reset lc to None at the start of the iteration
                
                # Find random WD lightcurve
                while lc is None:
                    rand = random.randint(1, 1291)
                    print(rand)
                    tic_id = int(df['Target ID'][rand])
                    lc = preprocess(tic_id)
                
                # Inject transit
                inj = inject_transit(ID, tic_id, lc, lc['time'].value,
                                radius_star = 0.01, 
                                mass_star = 0.6 * 2 * 10**30, 
                                radius_planet = 1-(1/res)*i, 
                                luminosity_star=0.001,
                                albedo_planet=0.1, 
                                period=1+(10/res)*j, 
                                inclination=90-(10/res)*k)
                
                # Run BLS
                results = BLSfit(inj)
                high_periods, high_powers, best_period, t0 = BLSResults(results, ID)
                FoldedLC(lc, best_period, t0, ID)
                BLSOutput(ID, tic_id, high_periods, high_powers, output_file)
                print('outputted')