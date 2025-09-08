import random
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import inject_transit
from BLSFit import BLSfit, BLSResults, FoldedLC
from BLStests import test_depth, test_v_shape, test_snr, test_out_of_transit_variability
import gc

df = pd.read_csv('data_inputs/tess_targets_data.csv')
inj_output_file = 'data_outputs/injected_transits_output4.csv'
noninj_output_file = 'data_outputs/noninjected_transits_output4.csv'

try:
    out = pd.read_csv(inj_output_file)
    out_ids = set(pd.to_numeric(out['ID'], errors='coerce').dropna().astype(int))
except FileNotFoundError:
    out_ids = set()

# Next numeric ID (monotone)
next_id = (max(out_ids) + 1) if out_ids else 0

# samples = 400
G = 6.67e-11

def sample_power_law(min_val, max_val, alpha, size=None):
    """
    Draw from PDF f(x) ∝ x^{-alpha}, on [min_val, max_val].
    """
    u = np.random.rand()
    if alpha == 1.0:
        # x = min * (max/min)^u
        return min_val * (max_val / min_val) ** u
    # Inverse CDF for alpha != 1
    pow_ = 1.0 - alpha
    a = min_val**pow_
    b = max_val**pow_
    return (a + (b - a) * u) ** (1.0 / pow_)


def find_light_curve():
    for _ in range(10):
        rand = random.randint(0, len(df) - 1)
        massH = float(df['MassH'].iloc[rand])
        tic_id = int(df['Target ID'].iloc[rand])
        m_s = float(massH * 1.989e30)
        r_s = np.cbrt(m_s / (1e9 * (4/3 * np.pi)))
        e_r_s = r_s * (1/3) * df['E_MassH'].iloc[rand] / df['MassH'].iloc[rand]
        
        try:
            print(f"Trying TIC ID {tic_id} with mass {massH:.3f} M_sun, radius {r_s/6.957e+8:.3f} R_sun")
            lc = preprocess(tic_id, TICID=True)
            return tic_id, lc, massH, m_s, r_s, e_r_s
        except Exception as e:
            print(f"Failed to preprocess TIC ID {tic_id} due to {e}, retrying...")
        return None

def fit_fold_and_test(lc, folder, output_file, Injected=False):
    print(f"Fitting and testing light curve for {'injected' if Injected else 'non-injected'} data...")
    results = BLSfit(lc)
    print(f"Results: {results}")
    high_periods, high_powers, best_period, t0, duration = BLSResults(results, plot='save', folder=folder, ID=ID)
    plt.close('all')

    rows = []
    for period in high_periods:
        folded_lc = FoldedLC(lc, period, t0, ID=ID, plot='', folder=folder, bin=True, time_bin_size=0.001, output=True)
        transit_mask = np.abs(folded_lc['time'].value) < 0.6 * duration.value

        plt.scatter(folded_lc['time'].value, folded_lc['flux'].value, s=1, c='k', label='Folded LC')
        plt.scatter(folded_lc[transit_mask]['time'].value, folded_lc[transit_mask]['flux'].value, s=1, c='r', label='Transit')

        oot_variability = test_out_of_transit_variability(folded_lc['flux'], transit_mask)
        transit_mask_sig = transit_mask & (folded_lc['flux'].value < (1 - 3*oot_variability))
        plt.scatter(folded_lc[transit_mask_sig]['time'].value, folded_lc[transit_mask_sig]['flux'].value, s=5, c='orange', label='>3 Sigma Points')

        plt.axhline(1 - oot_variability, color='b', linestyle='--', label='1 Sigma OOT Variability')
        plt.axhline(1 - 2*oot_variability, color='b', linestyle='--', label='2 Sigma OOT Variability')
        plt.axhline(1 - 3*oot_variability, color='b', linestyle='--', label='3 Sigma OOT Variability')
        plt.xlabel('Phase [JD]')
        plt.ylabel('Normalized Flux')
        plt.legend()
        plt.title(f'ID {ID} Folded Light Curve at Period = {round(period,3)} days')
        plt.savefig(f'../../../Research/{folder}/ID_{ID}_Folded_LC_Period_{round(period,3)}.png')
        plt.close('all')

        # Run tests
        try: median, mean, max_depth = test_depth(folded_lc['time'],
            folded_lc['flux'],
            transit_mask_sig)
        except: median, mean, max_depth = np.nan, np.nan, np.nan
        try: vshape = test_v_shape(folded_lc['time'],
                            folded_lc['flux'],
                            transit_mask
                            )
        except: vshape = np.nan
        try: snr = test_snr(folded_lc['flux'], transit_mask_sig)
        except: snr = np.nan

        rows.append([ID, tic_id, r_s, e_r_s, r_p, a, P_days, inc, 
                        period, duration, vshape, 
                        median, mean, max_depth, 
                        oot_variability, snr])

    if rows:
        with open(output_file, 'a', newline='') as f:
            csv.writer(f).writerows(rows)

while True:
    star_info = find_light_curve()
    if star_info is None:
        tqdm.write("Could not get a valid light curve after retries; continuing to next sample.")
        continue
    tic_id, lc, massH, m_s, r_s, e_r_s = star_info
    tqdm.write(f"Using TIC {tic_id}: r_s={r_s/6.957e+8:.3f} R_sun")

    r_p = float(sample_power_law(0.5, 5, 1.5)) # Earth Radii
    rho = 1186*r_p**0.4483 if r_p < 2.5 else 2296*r_p**-1.413
    
    roche = np.cbrt((3/2) * np.pi * m_s / rho)
    a_min = 0.5 * roche
    a_max = 10.5 * roche

    a = None
    P_days = None
    for _ in range(10):
        a_try = np.random.uniform(a_min, a_max)
        P_try = np.sqrt((4*np.pi**2 * a_try**3) / (G * m_s)) / (24*3600)
        if P_try <= 15: # Filter max of 15 days
            a = a_try
            P_days = P_try
            break
    if a is None:
        tqdm.write("Could not find an 'a' with P<=15 d within 10 tries; skipping this star.")
        continue

    x = np.clip((0.01 + r_p)/a, -1.0, 1.0)
    inc_min = np.degrees(np.arccos(x))
    # inc = 90 - (k * (90 - inc_min) / res)           # Inclination from 90 to i_min degrees
    inc = np.random.uniform(inc_min, 90)

    # print(f"radius of white dwarf: {r_s / 6.957e+8}")

    id_int = next_id
    ID = f"{id_int:06d}"
    next_id += 1
    out_ids.add(id_int)

    # Inject transit
    try:
        inj = inject_transit(tic_id, lc, lc['time'].value,
                    radius_star = r_s / 6.957e+8,   # radius of white dwarf in Solar radii
                    mass_star = massH,              # mass of white dwarf in Solar masses
                    radius_planet = r_p * 0.01,     # radius of planet in Solar radii
                    albedo_planet=0.1, 
                    period=P_days,
                    inclination=inc,
                    ID=ID#,
                    #a=a / (r_s / 6.957e+8) # Semi-major axis in Solar radii. DO NOT UNCOMMENT. BUGGED!
        )
        plt.close('all')

        fit_fold_and_test(lc, folder='WD_Plots10/Noninjected', output_file=noninj_output_file, Injected=False)
        fit_fold_and_test(inj, folder='WD_Plots10/Injected', output_file=inj_output_file, Injected=True)

    except Exception as e:
        tqdm.write(f"Error processing ID {ID}: {e}")
    finally:
        plt.close('all')
        gc.collect()