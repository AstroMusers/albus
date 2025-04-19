import random
import pandas as pd
import numpy as np
from tqdm import tqdm
# from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import inject_transit, calc_a
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSTestOutputs
from BLStests import test_depth, test_v_shape, test_snr, test_out_of_transit_variability, create_transit_mask_manual

df = pd.read_csv('tess_targets_data.csv')
output_file = '/Users/aavikwadivkar/Documents/Exoplanets/Ampersand/albus/albus/data_outputs/injected_transits_output2.csv'

out = pd.read_csv(output_file)
lc = None

res = 10

for i in range(res):
    for j in range(res):
        for k in tqdm(range(res)):
            ID = str(i) + str(j) + str(k)
            print(ID)
            if int(ID) not in out['ID'].values:  #  Ensure check is against column values
                lc = None  
                
                # Find random WD lightcurve
                while lc is None:
                    rand = random.randint(1, 1291)
                    print(rand)
                    tic_id = int(df['Target ID'][rand])
                    try: lc = preprocess(tic_id, TICID=True)
                    except: pass
                
                r_p = ((i+1)/res)**2                        # Range of planet radii from 0.01 to 1
                period = 1+(10/res)*j                       # Range of periods from 1 to 10 days
                a = calc_a(0.6, period)/6.957*10**8         # Semi-major axis in meters
                i_min = np.arccos((0.01+r_p)/a)/np.pi*180   # Minimum transit inclination in degrees
                i = 90-(k*(90-i_min)/res)                   # Inclination from 90 to i_min degrees

                # Inject transit
                inj = inject_transit(ID, tic_id, lc, lc['time'].value,
                                radius_star = 0.01, 
                                mass_star = 0.6, 
                                radius_planet = r_p * 0.01, 
                                luminosity_star=0.001,
                                albedo_planet=0.1, 
                                period=period,
                                inclination=i
                                )
                
                # Run BLS
                results = BLSfit(inj)
                high_periods, high_powers, best_period, t0, duration = BLSResults(results, folder='WD_Plots7', ID=ID)
                for period in high_periods: 
                    FoldedLC(inj, period, t0, ID=ID, folder='WD_Plots7', bin=False)                
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
                