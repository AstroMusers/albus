"""Legacy Albus names for the shared Miletos transit measurements."""

import numpy as np
from scipy.signal import savgol_filter

from miletos.transit_metrics import (
    _sigma_clip,
    _wrap_centered_phase,
    duration_at_max_power as test_duration,
    out_of_transit_variability as test_out_of_transit_variability,
    period_at_max_power as test_period,
    transit_depth as test_depth,
    transit_depth_quantile_phase,
    transit_snr as test_snr,
    transit_v_shape as test_v_shape,
)
from miletos.transit_search import transit_mask as create_transit_mask_manual


# Example usage: you would call these functions with your actual data
if __name__ == "__main__":
    # For demonstration purposes, we assume the following variables are defined:
    # bls_periodogram: object returned from lightkurve's BLS fit
    # detrended_time: 1D numpy array of time values for the detrended lightcurve
    # detrended_flux: 1D numpy array of normalized flux values
    # transit_mask: boolean array where True indicates data points during the transit
    
    # Example (dummy) inputs:
    class DummyBLS:
        period_at_max_power = 0.5  # days
        duration_at_max_power = 0.01  # days
    
    bls_periodogram = DummyBLS()
    detrended_time = np.linspace(0, 1, 1000)
    # Create a simple synthetic transit: flux drops to 0.8 between indices 400 and 420
    detrended_flux = np.ones_like(detrended_time)
    detrended_flux[400:420] = 0.8
    transit_mask = np.zeros_like(detrended_flux, dtype=bool)
    transit_mask[400:420] = True

    period = test_period(bls_periodogram)
    duration = test_duration(bls_periodogram)
    depth = test_depth(detrended_time, detrended_flux, transit_mask)
    v_shape = test_v_shape(detrended_time, detrended_flux, transit_mask)
    snr = test_snr(detrended_flux, transit_mask)
    oot_variability = test_out_of_transit_variability(detrended_flux, transit_mask)
    
    print("Transit Period (days):", period)
    print("Transit Duration (days):", duration)
    print("Transit Depth:", depth)
    print("V-shape Metric:", v_shape)
    print("Transit SNR:", snr)
    print("Out-of-Transit Variability:", oot_variability)
