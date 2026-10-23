import random
import pandas as pd
import csv
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import inject_transit
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSOutput, BLSTestOutputs
from BLStests import test_period, test_duration, test_depth, test_v_shape, test_snr, test_out_of_transit_variability, create_transit_mask_manual

df = pd.read_csv('tess_targets_data.csv')
output_file = '/Users/aavikwadivkar/Documents/Exoplanets/Ampersand/albus/albus/data_outputs/noninjected_transits_output1.csv'
out = pd.read_csv(output_file)
completed_ids = set(out['ID'])
lc = None

for i in tqdm(range(5)):
    if df.empty:
        break
    rand = random.randrange(len(df))
    ID = int(df['Target ID'].iloc[rand])
    print(ID)
    if ID not in completed_ids:
        lc = None  # Reset lc to None at the start of the iteration
        
        tic_id = ID
        
        lc = preprocess(tic_id, TICID=True)

        if lc is not None:
            # Run BLS
            results = BLSfit(lc)
            high_periods, high_powers, best_period, epochs, durations = BLSResults(results, folder='WD_Plots6/Noninjections', ID=ID, candidate_ephemerides=True)
            for period, t0, duration in zip(high_periods, epochs, durations):
                FoldedLC(lc, period, t0, ID=ID, folder='WD_Plots6/Noninjections', bin=False)                
                # Run tests
                transit_mask = create_transit_mask_manual(lc['time'], period, t0, duration)
                depth = test_depth(lc['time'], lc['flux'], transit_mask)
                vshape = test_v_shape(lc['time'], lc['flux'], transit_mask)
                snr = test_snr(lc['flux'], transit_mask)
                oot_variability = test_out_of_transit_variability(lc['flux'], transit_mask)

                BLSTestOutputs(ID, tic_id, period, duration, depth, vshape, snr, oot_variability, output_file)
            completed_ids.add(ID)
            print('outputted')