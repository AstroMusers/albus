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
output_file = 'Pipeline/no_injections.csv'

# with open(output_file, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['ID', 'Periods', 'Powers'])
#         f.close()

out = pd.read_csv('Pipeline/no_injections.csv')
lc = None

for i in tqdm(range(140)):
    ID = int(df['Target ID'][i])
    print(ID)
    if int(ID) not in out['ID'].values:  # Ensure check is against column values
        lc = None  # Reset lc to None at the start of the iteration
        
        tic_id = ID
        try: 
            lc = preprocess(tic_id)

            if lc is not None:

                # Run BLS
                results = BLSfit(lc)
                high_periods, high_powers, best_period, t0 = BLSResults(results, ID, folder='WD_Plots4')
                FoldedLC(lc, best_period, t0, ID, folder='WD_Plots4')
                BLSOutput(ID, tic_id, high_periods, high_powers, output_file)
                print('outputted')
        except:
            print('something went wrong')