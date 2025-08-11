import random
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
# from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import inject_transit, calc_a
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSTestOutputs
from BLStests import test_depth, test_v_shape, test_snr, test_out_of_transit_variability, create_transit_mask_manual

df = pd.read_csv('tess_targets_data.csv')
output_file = 'data_outputs/injected_transits_output3.csv'

out = pd.read_csv(output_file)
lc = None

res = 12

for i in range(res):
    for k in range(res):
        
        lc = None

        while lc is None:
            rand = random.randint(1, 1290)
            print(rand)
            tic_id = int(df['Target ID'][rand])
            try: lc = preprocess(tic_id, TICID=True)
            except: pass
        
        for j in tqdm(range(res)):
            ID = str(i).zfill(2) + str(j).zfill(2) + str(k).zfill(2)
            print(ID)
            if int(ID) not in out['ID'].values:  #  Ensure check is against column values

                m_s = float(df['MassH'][rand] * 1.989 * 10**30)    # Mass of white dwarf in kg
                r_s = np.cbrt((m_s)/(10**9*(4/3 * np.pi)))  # Assumes density of 10^9 kg/m^3, need citation. 
                                                            # Radius in meters
                print(f"Radius of white dwarf: {r_s} m")

                e_r_s = r_s * 1/3 * df['E_MassH'][rand]/df['MassH'][rand]  # Error in radius of white dwarf in meters
                
                # r_p = min(np.random.power(0.5, 1) + 0.5, 10)        # Random radius from 0.5 to 10 Earth Radii
                r_p = 10*(i/res)**2 + 0.5 # Range of planet radii from 0.5 to 10 Earth Radii

                if r_p < 2.5:
                    rho = 1186*r_p**0.4483                     # Density of rock in kg/m^3, https://www.aanda.org/articles/aa/full_html/2020/02/aa36482-19/aa36482-19.html#FD1
                else:
                    rho = 2296*r_p**-1.413                     # Density of gas in kg/m^3, need citation

                roche = np.cbrt((3/2)*np.pi * m_s/rho)
                a = roche*(10*(j/res)**2 + 0.5)                     # Semi-major axis in meters, range from 0.5 to 10 Roche limit
                real_period = np.sqrt((4*np.pi**2*a**3)/(6.67*10**-11*m_s)) / (24*3600)  # Orbital period in days
                if real_period > 15: continue  # Skip if period is greater than 15 days

                # period = 1+(10/res)*j                       # Range of periods from 1 to 10 days
                # a = calc_a(0.6, period)/6.957*10**8         # Semi-major axis in meters
                
                inc_min = np.arccos((0.01+r_p)/a)/np.pi*180   # Minimum transit inclination in degrees
                inc = 90-(k*(90-inc_min)/res)                   # Inclination from 90 to i_min degrees

                print(f"radius of white dwarf: {r_s / 6.957e+8}")

                # Inject transit
                inj = inject_transit(tic_id, lc, lc['time'].value,
                                radius_star = r_s / 6.957e+8,   # radius of white dwarf in Solar radii
                                mass_star = m_s,  # mass of white dwarf in Solar masses
                                radius_planet = r_p * 0.01,     # radius of planet in Solar radii
                                albedo_planet=0.1, 
                                period=real_period,
                                inclination=inc,
                                ID=ID
                                )
                
                # Run BLS
                results = BLSfit(inj)
                high_periods, high_powers, best_period, t0, duration = BLSResults(results, plot='save', folder='WD_Plots8', ID=ID)
                for period in high_periods: 
                    FoldedLC(inj, period, t0, ID=ID, plot = 'save', folder='WD_Plots8', bin=False)                
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

                    # BLSTestOutputs(ID, tic_id, period, duration, depth, vshape, snr, oot_variability, output_file)
                    with open(output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        # writer.writerow(['ID', 'TIC ID', 'Period', 'Duration', 'Depth', 'V-Shape', 'SNR', 'OOT Variability'])
                        writer.writerow([ID, tic_id, r_s, e_r_s, r_p, a, real_period, inc, period, duration, depth, vshape, snr, oot_variability])
                        f.close()
                print('outputted')
                