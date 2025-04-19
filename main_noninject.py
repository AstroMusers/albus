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
lc = None

for i in tqdm(range(5)):
    rand = random.randint(1, 1291)
    ID = int(df['Target ID'][rand])
    print(ID)
    if int(ID) not in out['ID'].values:  # Ensure check is against column values
        lc = None  # Reset lc to None at the start of the iteration
        
        tic_id = ID
        
        lc = preprocess(tic_id, TICID=True)

        if lc is not None:
            # Run BLS
            results = BLSfit(lc)
            high_periods, high_powers, best_period, t0, duration = BLSResults(results, folder='WD_Plots6/Noninjections', ID=ID)
            for period in high_periods:
                FoldedLC(lc, period, t0, ID=ID, folder='WD_Plots6/Noninjections', bin=False)                
                # Run tests
                depth = test_depth(lc['time'],
                                lc['flux'],
                                create_transit_mask_manual(lc['time'], period, t0, duration
                                ))
                vshape = test_v_shape(lc['time'],
                                    lc['flux'],
                                    create_transit_mask_manual(lc['time'], period, t0, duration
                                    ))
                snr = test_snr(lc['flux'], create_transit_mask_manual(lc['time'], period, t0, duration))
                oot_variability = test_out_of_transit_variability(lc['flux'], create_transit_mask_manual(lc['time'], period, t0, duration))

                BLSTestOutputs(ID, tic_id, period, duration, depth, vshape, snr, oot_variability, output_file)
        print('outputted')