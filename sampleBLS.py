import random
import pandas as pd
import csv
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from preprocess import preprocess
from injections import EARTH_RADIUS_M, SOLAR_RADIUS_M, inject_transit, minimum_transit_inclination
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSTestOutputs
from BLStests import test_period, test_duration, test_depth, test_v_shape, test_snr, test_out_of_transit_variability, transit_depth_quantile_phase
from tdpy.verbosity import print


catalog_path = Path('data_inputs/tess_targets_data_old.csv')
print(f'Reading from {catalog_path}...')
df = pd.read_csv(catalog_path)
# output_file = 'Pipeline/injected_transits_datapoints.csv'

# out = pd.read_csv('Pipeline/injected_transits_datapoints.csv')
# lc = None

lc = None

# Find random WD lightcurve
while lc is None:
    rand = random.randint(1, 1290)
    # rand = 602
    # print(rand)
    tic_id = int(df['TargetID'][rand])
    tic_id = 435187907
    print(tic_id)
    try: lc = preprocess(tic_id, TICID=True)
    except: pass

m_s = float(df['MassH'][rand]*1.989*10**30)   # [kg]
r_s = np.cbrt(m_s/(1e9*(4/3 * np.pi)))        # [m], density = 10^9 kg m^-3
e_r_s = r_s * 1/3 * df['E_MassH'][rand]/df['MassH'][rand]  # [m]

r_p = 1                         # [R_earth]

rho = 1186*r_p**0.4483 if r_p < 2.5 else 2296*r_p**-1.413  # [kg m^-3]
roche = np.cbrt((3/2)*np.pi * m_s/(rho))  # [m]
a = 4*roche  # [m]
print(f"4 Roche lobe radius: {a} m")
period = np.sqrt((4*np.pi**2*a**3)/(6.67*10**-11*(m_s))) / (24*3600)  # [day]

# period = 1+(10/res)*j                       # Range of periods from 1 to 10 days
# a = calc_a(0.6, period)/6.957*10**8         # Semi-major axis in meters

inc_min = minimum_transit_inclination(r_s, r_p * EARTH_RADIUS_M, a)  # [deg]
inc = 90                                   # [deg]

    print(f"r_s: {r_s/SOLAR_RADIUS_M} solar radii, e_r_s: {e_r_s/SOLAR_RADIUS_M}, r_p: {r_p} r_e, a: {a/1.496e+11} au, period: {period}, inc: {inc}")

inj = lc
# Inject transit
inj = inject_transit(tic_id, lc, lc['time'].value,
                radius_star = r_s / SOLAR_RADIUS_M,  # [R_sun]
                mass_star = df['MassH'][rand], 
                radius_planet = r_p * EARTH_RADIUS_M / SOLAR_RADIUS_M,  # [R_sun]
                albedo_planet=0.1, 
                period=period,
                inclination=inc
                )

# print(f"Injected light curve: {inj['flux'].value[:10]}...")  # Print first 10 flux values for verification

# Run BLS
results = BLSfit(inj)
high_periods, high_powers, best_period, t0, duration = BLSResults(results, folder='sampleBLS', plot='save', ID=tic_id)
print(f"Best period: {best_period}, t0: {t0}, duration: {duration}")
# for period in high_periods: 
#     # FoldedLC(inj, period, t0, plot='show', bin=False)                
#     # Run tests
#     depth = test_depth(inj['time'],
#                     inj['flux'],
#                     create_transit_mask_manual(inj['time'], period, t0, duration
#                     ))
#     vshape = test_v_shape(inj['time'],
#                         inj['flux'],
#                         create_transit_mask_manual(inj['time'], period, t0, duration
#                         ))
#     snr = test_snr(inj['flux'], create_transit_mask_manual(inj['time'], period, t0, duration))
#     oot_variability = test_out_of_transit_variability(inj['flux'], create_transit_mask_manual(inj['time'], period, t0, duration))
    
#     print(f"Period: {period}, Depth: {depth}, V-Shape: {vshape}, SNR: {snr}, OOT Variability: {oot_variability}")

# transit_mask = create_transit_mask_manual(inj['time'], best_period, t0, 0.1)
# print(len(transit_mask))
# print(len(inj['time']))
# print(len(inj['time'][transit_mask]))
# plt.plot(inj['time'][transit_mask].value, inj['flux'][transit_mask].value)
# plt.show()
# plt.close()

folded_lc = FoldedLC(inj, best_period, t0, bin=True, time_bin_size=0.001, output=True, plot='save', folder='sampleBLS', ID=tic_id)
transit_mask = np.abs(folded_lc['time'].value) < 0.6 * duration.value

print(f"Transit points: {np.sum(transit_mask)}, Total points: {len(folded_lc)}")
folded_lc_np = {
    'time': np.asarray(folded_lc['time'].value, dtype=float),  # phase as floats
    'flux': np.asarray(folded_lc['flux'].value, dtype=float),
}
# res = transit_depth_quantile_phase(
#     folded_lc_np,
#     duration.value,
#     phase0=0,
#     nbins=int(np.sqrt(np.sum(transit_mask)))
# )

# res = transit_depth_quantile_phase(
#     folded_lc_np,
#     duration=duration.value,  # in phase units
#     phase0=0.0,                    # or your expected center
#     nbins=120,                     # fewer bins -> more points per bin
#     tau=0.05,                      # hug the floor more (was 0.20)
#     window_scale=2.0,              # look a bit wider
#     smooth_window=5,               # gentler smoothing
#     smooth_poly=2,
#     guard_factor=1.3               # keep OOT guard but not too aggressive
# )
# print(res['F0'], res['floor'], res['depth'], res['snr'])

# print(f"Depth: {res['depth']:.5f} ± {res['depth_err']:.5f}, SNR={res['snr']:.1f}")

plt.scatter(folded_lc['time'].value, folded_lc['flux'].value, s=1, c='k', label='Folded LC')
plt.scatter(folded_lc[transit_mask]['time'].value, folded_lc[transit_mask]['flux'].value, s=1, c='r', label='Transit')

oot_variability = test_out_of_transit_variability(folded_lc['flux'], transit_mask)
transit_mask_sig = transit_mask & (folded_lc['flux'].value < (1 - 3*oot_variability))
plt.scatter(folded_lc[transit_mask_sig]['time'].value, folded_lc[transit_mask_sig]['flux'].value, s=5, c='orange', label='>3 Sigma Points')

plt.axhline(1 - oot_variability, color='b', linestyle='--', label='1 Sigma OOT Variability')
plt.axhline(1 - 2*oot_variability, color='b', linestyle='--', label='2 Sigma OOT Variability')
plt.axhline(1 - 3*oot_variability, color='b', linestyle='--', label='3 Sigma OOT Variability')

# transit_lc = folded_lc[transit_mask]
# bins = int(np.sqrt(np.sum(transit_mask)))

# binned_transit = transit_lc.bin(bins=bins)
# plt.scatter(binned_transit['time'].value, binned_transit['flux'].value, s=10, c='orange', label='Binned Transit')

# plt.plot(res['bin_centers'], res['bin_quantile'], 
#          color='orange', alpha=0.5, lw=1, label='Lower-quantile (unsmoothed)')
# plt.plot(res['bin_centers'], res['bin_quantile_sm'], 
#          color='red', lw=2, label='Smoothed quantile envelope')

# plt.axhline(res['floor'], color='g', linestyle=':', lw=2, 
#             label=f'Estimated floor (depth={res["depth"]:.4f})')
# plt.axvline(res['floor_phase'], color='g', linestyle=':', lw=1)
plt.title(f'Folded Light Curve for TIC {tic_id} at {best_period:.2f} days')
plt.xlabel('Phase [JD]')
plt.ylabel('Normalized Flux')
plt.legend()
output_path = Path('sampleBLS') / f'Folded_LC_TIC_{tic_id}.png'
print(f'Writing to {output_path}...')
plt.savefig(output_path, dpi=300)

try: median, mean, max_depth = test_depth(folded_lc['time'],
                folded_lc['flux'],
                transit_mask_sig)
except: median, mean, max_depth = np.nan, np.nan, np.nan
vshape = test_v_shape(folded_lc['time'],
                    folded_lc['flux'],
                    transit_mask
                    )
try: snr = test_snr(folded_lc['flux'], transit_mask_sig)
except: snr = np.nan

print(f"Actual Period: {period}, Best Period: {best_period}, V-Shape: {vshape}, "
      f"Median Depth: {median}, Mean Depth: {mean}, Max Depth: {max_depth}"
      f"OOT Variability: {oot_variability}, SNR: {snr}")

print('done!')