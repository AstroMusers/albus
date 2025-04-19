import random
import pandas as pd
import csv
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import inject_transit
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSTestOutputs
from BLStests import test_period, test_duration, test_depth, test_v_shape, test_snr, test_out_of_transit_variability, create_transit_mask_manual

df = pd.read_csv('tess_targets_data.csv')
output_file = '/Users/aavikwadivkar/Documents/Exoplanets/Ampersand/albus/albus/data_outputs/injected_transits_output1.csv'

out = pd.read_csv(output_file)
lc = None

res = 5

for i in range(res):
    for j in range(res):
        for k in tqdm(range(res)):
            ID = str(i) + str(j) + str(k)
            print(ID)
            if int(ID) not in out['ID'].values:  #  Ensure check is against column values
                lc = None  # Reset lc to None at the start of the iteration
                
                # Find random WD lightcurve
                while lc is None:
                    rand = random.randint(1, 1291)
                    print(rand)
                    tic_id = int(df['Target ID'][rand])
                    try: lc = preprocess(tic_id, TICID=True)
                    except: pass
                
                # Inject transit
                inj = inject_transit(ID, tic_id, lc, lc['time'].value,
                                radius_star = 0.01, 
                                mass_star = 0.6, 
                                radius_planet = 1-(1/res)*i, 
                                luminosity_star=0.001,
                                albedo_planet=0.1, 
                                period=1+(10/res)*j, 
                                inclination=90-(1/res)*k)
                
                # Run BLS
                results = BLSfit(inj)
                high_periods, high_powers, best_period, t0, duration = BLSResults(results, folder='WD_Plots6', ID=ID)
                for period in high_periods: 
                    FoldedLC(inj, period, t0, ID=ID, folder='WD_Plots6', bin=False)                
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

                    BLSTestOutputs(ID, tic_id, period, duration, depth, vshape, snr, oot_variability, output_file)
                print('outputted')
                