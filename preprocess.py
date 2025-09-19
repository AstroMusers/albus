import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

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
        plt.savefig(f'presentation_plots/{input}_rawlc.png')
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
                plt.savefig(f'presentation_plots/{input}_flattenedlc.png')
                plt.close()
        
        if lc_single.flux.unit != 'ppm':
            # print('normalizing!')
            lc_single.flux = lc_single.flux/np.median(lc_single.flux)
            lc_single.flux_err = lc_single.flux_err/np.median(lc_single.flux_err)
            # lc_single.plot()
            # plt.show()
        if lc_single.flux.unit == 'ppm':
            # print('normalizing!')
            lc_single.flux = 1+ lc_single.flux/1000000
            lc_single.flux_err = 1+ lc_single.flux_err/1000000
            # lc_single.plot()
            # plt.show()
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
        plt.savefig(f'presentation_plots/{input}_detrendedlc.png')
        plt.close()
        print("Plotted detrended LC.")

    detrended_lc = lk.LightCurve(time=lc.time, flux=detrended_flux)
    return detrended_lc