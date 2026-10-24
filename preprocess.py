import random

import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from pathlib import Path
from tdpy.verbosity import print

def find_light_curve(targets, max_tries=10, rng=None, *, preprocess_func=None, verbose=False):
    """Select a target with a usable curve, retaining the fixed-density radius estimate."""
    if len(targets) == 0:
        return None
    if rng is None:
        rng = random
    if preprocess_func is None:
        preprocess_func = preprocess
    for _ in range(max_tries):
        index = rng.randint(0, len(targets) - 1)
        mass_solar = float(targets['MassH'].iloc[index])  # [M_sun]
        target_id = int(targets['Target ID'].iloc[index])
        mass_kg = float(mass_solar * 1.989e30)  # [kg]
        radius_m = np.cbrt(mass_kg / (1e9 * (4 / 3 * np.pi)))  # [m]
        radius_error_m = radius_m * (1 / 3) * targets['E_MassH'].iloc[index] / targets['MassH'].iloc[index]  # [m]
        try:
            if verbose:
                print(f"Trying TIC ID {target_id} with mass {mass_solar:.3f} M_sun, radius {radius_m/6.957e+8:.3f} R_sun")
            light_curve = preprocess_func(target_id, TICID=True)
            if light_curve is None:
                if verbose:
                    print(f"No light curve for TIC ID {target_id}, retrying...")
                continue
            return target_id, light_curve, mass_solar, mass_kg, radius_m, radius_error_m
        except Exception as error:
            if verbose:
                print(f"Failed to preprocess TIC ID {target_id} due to {error}, retrying...")
    return None


def preprocess(input, TICID = True, injection = False, plots = False, report=False):
# Download the TESS light curve for a specific TIC ID
    if TICID: search_result = lk.search_lightcurve(f"TIC {input}", mission='TESS')
    else: search_result = lk.search_lightcurve(input, mission='TESS')
    # print("search has resulted")

    try: lcc = search_result.download_all()
    except:
        print('download failed')
        return
    # print("downloaded")

    if lcc is None:
        print('download empty')
        return

    if plots: 
        lcc[0].plot()
        plot_path = Path('presentation_plots') / f'{input}_rawlc.png'
        print(f'Writing to {plot_path}...')
        plt.savefig(plot_path)
        plt.close()

    lc_list = []
    max_day = 0
    for lc_single in lcc:
        # Normalize to ppm
        lc_single = lc_single.remove_nans()
        # Convert to ppm if not already in ppm
        # print(lc_single.flux.unit)
        lc_single.flatten(window_length=301)
        if plots:
            if lc_list == []: 
                max_day = lc_single.time[-1].value
                lc_single.plot()
                plot_path = Path('presentation_plots') / f'{input}_flattenedlc.png'
                print(f'Writing to {plot_path}...')
                plt.savefig(plot_path)
                plt.close()
        
        lc_single = lc_single.normalize()
        lc_list.append(lc_single.remove_outliers())
        if report: print(f"Appended LC with {len(lc_single.time)} points.")

    # Stitch the light curves in lc_list
    lc = lk.LightCurveCollection(lc_list).stitch() 

    # flatlc = lc
    
    trend = medfilt(lc.flux, kernel_size=501)
    if injection: detrended_flux = lc.flux - trend
    else: detrended_flux = lc.flux - trend + 1
    if plots:
        print("Plotting detrended LC...")
        lc_plot = lc[(lc.time.value > 0) & (lc.time.value < max_day)]
        plt.figure(figsize=(10,5))
        lc_plot.scatter(s=1, c='k', label='Raw LC')
        # plt.scatter(lc_plot.time, trend[:1000], s=1, c='r', label='Trend (medfilt)')
        # plt.scatter(lc_plot.time, detrended_flux[:1000], s=1, c='b', label='Detrended LC')
        # plt.plot(lc.time, lc.flux, 'k.', markersize=1)
        # plt.plot(lc.time, trend, 'r-', label='Trend (medfilt)')
        # plt.plot(lc.time, detrended_flux, 'b.', markersize=1, label='Detrended LC')
        plt.xlabel('Time [days]')
        plt.ylabel('Flux')
        plt.title(f'ID {input} Detrending')
        plt.legend()
        plot_path = Path('presentation_plots') / f'{input}_detrendedlc.png'
        print(f'Writing to {plot_path}...')
        plt.savefig(plot_path)
        plt.close()
        print("Plotted detrended LC.")

    detrended_lc = lk.LightCurve(time=lc.time, flux=detrended_flux, flux_err=lc.flux_err)
    return detrended_lc
