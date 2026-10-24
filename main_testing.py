import random
from pathlib import Path
import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from paths import get_repository_path
from tdpy.verbosity import print
from preprocess import preprocess
from injections import inject_transit
from injection_sampling import sample_power_law
from BLSFit import BLSfit, BLSResults, FoldedLC
from BLStests import test_depth, test_v_shape, test_snr, test_out_of_transit_variability
import gc

DATA_DIR = get_repository_path()
target_path = DATA_DIR / 'data_inputs' / 'tess_targets_data.csv'
print(f'Reading from {target_path}...')
df = pd.read_csv(target_path)
inj_output_file = 'data_outputs/injected_transits_output6.csv'
noninj_output_file = 'data_outputs/noninjected_transits_output6.csv'


G = 6.67e-11

def find_light_curve():
    print('kill me')

ID = 999

def fit_fold_and_test(lc, folder, output_file, Injected=False):
    print(f"Fitting and testing light curve for {'injected' if Injected else 'non-injected'} data...")
    results = BLSfit(lc)
    print(f"Results: {results}")
    high_periods, high_powers, best_period, epochs, durations = BLSResults(results, plot='save', folder=folder, ID=ID, candidate_ephemerides=True)
    plt.close('all')

    rows = []
    for period, t0, duration in zip(high_periods, epochs, durations):
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
        plot_path = Path(folder) / f'ID_{ID}_Folded_LC_Period_{round(period,3)}.png'
        print(f'Writing to {plot_path}...')
        plt.savefig(plot_path)
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
