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
inj_output_file = 'data_outputs/injected_transits_output5.csv'
noninj_output_file = 'data_outputs/noninjected_transits_output5.csv'

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
    print('kill me')

ID = 999

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
        plt.savefig(f'{folder}/ID_{ID}_Folded_LC_Period_{round(period,3)}.png')
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