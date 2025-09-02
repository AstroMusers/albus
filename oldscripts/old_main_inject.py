import random
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import inject_transit, calc_a
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSTestOutputs
from BLStests import test_depth, test_v_shape, test_snr, test_out_of_transit_variability, create_transit_mask_manual
import gc

df = pd.read_csv('tess_targets_data.csv')
output_file = 'data_outputs/injected_transits_output3.csv'

out = pd.read_csv(output_file)
lc = None

res = 20
G = 6.67e-11

out_ids = set(pd.to_numeric(out['ID'], errors='coerce').dropna().astype(int))

for i in range(res):
    for k in range(res):
        js_to_run = [j for j in range(res) if int(f"{i:02}{k:02}{j:02}") not in out_ids]
        tqdm.write(f"i={i}, k={k}, js_to_run: {js_to_run}")
        if not js_to_run:
            continue

        # print(f"js_to_run: {js_to_run}")

        r_p = 10*(i/res)**2 + 0.5
        rho = 1186*r_p**0.4483 if r_p < 2.5 else 2296*r_p**-1.413

        j_arr = np.array(js_to_run, dtype=float)
        scale = 10*(j_arr/res)**2 + 0.5
        K = (2*np.pi/(24 * 3600)) * np.sqrt((1.5*np.pi)/(G*rho))     # depends on rho, not star mass
        P_days = K * (scale**1.5)

        viable_js = j_arr[P_days <= 15].astype(int).tolist()
        if not viable_js:
            # Nothing in this block can ever pass the period cutoff → mark and skip
            tqdm.write(f"Skipping i={i}, k={k} as no viable js found within period limit.")
            out_ids.update(int(f"{i:02}{k:02}{j:02}") for j in js_to_run)
            continue

        lc = None
        tic_id = None
        m_s = None
        r_s = None
        e_r_s = None

        for _ in range(10):
            rand = random.randint(0, len(df) - 1)
            massH = float(df['MassH'].iloc[rand])
            tic_id = int(df['Target ID'].iloc[rand])
            m_s = float(massH * 1.989e30)
            r_s = np.cbrt(m_s / (1e9 * (4/3 * np.pi)))
            e_r_s = r_s * (1/3) * df['E_MassH'].iloc[rand] / df['MassH'].iloc[rand]
            
            try:
                print(f"Trying TIC ID {tic_id} with mass {massH} M_sun, radius {r_s/6.957e+8} R_sun")
                lc = preprocess(tic_id, TICID=True)
                break
            except Exception as e:
                print(f"Failed to preprocess TIC ID {tic_id} due to {e}, retrying...")
                lc = None
                continue
        if lc is None:
            tqdm.write(f"Failed to find a valid light curve for TIC ID {tic_id} after 10 attempts.")
            out_ids.update(int(f"{i:02}{k:02}{j:02}") for j in js_to_run)
            continue
        tqdm.write(f"Found light curve for TIC ID {tic_id}: r_s={r_s/6.957e+8} solar radii")

        roche = np.cbrt((3/2) * np.pi * m_s / rho)

        for j in tqdm(js_to_run, leave=False, desc=f"Processing i={i}, k={k}"):
            ID = f"{i:02}{k:02}{j:02}"
            id_int = int(ID)
            # print(ID)
            # if j % 5 == 0:
            # tqdm.write(f"{ID} processing...")

            a = roche * (10*(j/res)**2 + 0.5)
            real_period = np.sqrt((4*np.pi**2 * a**3) / (G * m_s)) / (24*3600)

            if real_period > 15:
                # mark as done so we never revisit it; no LC work
                out_ids.add(id_int)
                continue

            # period = 1+(10/res)*j                       # Range of periods from 1 to 10 days
            # a = calc_a(0.6, period)/6.957*10**8         # Semi-major axis in meters
            
            x = np.clip((0.01 + r_p)/a, -1.0, 1.0)
            inc_min = np.degrees(np.arccos(x))
            inc = 90 - (k * (90 - inc_min) / res)           # Inclination from 90 to i_min degrees

            print(f"radius of white dwarf: {r_s / 6.957e+8}")

            # Inject transit
            try:
                inj = inject_transit(tic_id, lc, lc['time'].value,
                            radius_star = r_s / 6.957e+8,   # radius of white dwarf in Solar radii
                            mass_star = massH,              # mass of white dwarf in Solar masses
                            radius_planet = r_p * 0.01,     # radius of planet in Solar radii
                            albedo_planet=0.1, 
                            period=real_period,
                            inclination=inc,
                            ID=ID,
                            a=a / (r_s / 6.957e+8)
                )
                plt.close('all')
            
                # Run BLS
                results = BLSfit(inj)
                high_periods, high_powers, best_period, t0, duration = BLSResults(results, plot='save', folder='WD_Plots9', ID=ID)
                plt.close('all')
                
                rows = []
                for period in high_periods: 
                    FoldedLC(inj, period, t0, ID=ID, plot = 'save', folder='WD_Plots9', bin=False)    
                    plt.close('all')
        
                    # Run tests
                    mask = create_transit_mask_manual(inj['time'], period, t0, duration)

                    depth = test_depth(inj['time'], inj['flux'], mask)
                    vshape = test_v_shape(inj['time'], inj['flux'], mask)
                    snr = test_snr(inj['flux'], mask)
                    oot_variability = test_out_of_transit_variability(inj['flux'], mask)

                    rows.append([ID, tic_id, r_s, e_r_s, r_p, a, real_period, inc, period, duration, depth, vshape, snr, oot_variability])

                if rows:
                    with open(output_file, 'a', newline='') as f:
                        csv.writer(f).writerows(rows)

                if j % 5 == 0:
                    tqdm.write(f"{ID} saved ({len(rows)} cand.)")

            except Exception as e:
                tqdm.write(f"Error processing ID {ID}: {e}")
            finally:
                out_ids.add(id_int)
                plt.close('all')
                gc.collect()
