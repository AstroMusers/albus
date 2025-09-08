import lightkurve as lk
import numpy as np
from scipy.signal import medfilt

def preprocess(input, TICID = True, injection = False):
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

    lc_list = []
    for lc_single in lcc:
        # Normalize to ppm
        lc_single = lc_single.remove_nans()
        # Convert to ppm if not already in ppm
        # print(lc_single.flux.unit)
        lc_single.flatten(window_length=301)
        
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

    # Stitch the light curves in lc_list
    lc = lk.LightCurveCollection(lc_list).stitch() 

    # flatlc = lc

    trend = medfilt(lc.flux, kernel_size=501)
    if injection: detrended_flux = lc.flux - trend
    else: detrended_flux = lc.flux - trend + 1

    detrended_lc = lk.LightCurve(time=lc.time, flux=detrended_flux)
    return detrended_lc