import random
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
import os
import lightkurve as lk
from lightkurve import LightCurve
from matplotlib.ticker import ScalarFormatter
from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import generate_lightcurve, inject_transit
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSOutput
from tests import test_period, test_duration, test_depth, test_v_shape, test_snr, test_out_of_transit_variability

preprocessed_lc = preprocess('WD 1856+534 b', TICID = False, injection = False)

def create_transit_mask_manual(time, period, t0, duration):
    """
    Create a transit mask manually given a time array and transit parameters,
    converting astropy Time objects and Quantities to plain floats (in days).

    Parameters:
        time (array-like or astropy.time.core.Time): Time array in days or a Time object.
        period (float or Quantity): Transit period in days.
        t0 (float or Time or Quantity): Transit mid-point (epoch) in days.
        duration (float or Quantity): Transit duration in days.

    Returns:
        mask (np.array of bool): Boolean array marking in-transit points.
    """
    import numpy as np
    from astropy.time import Time
    from astropy import units as u

    # Convert time to a plain numpy array of floats (in days)
    if isinstance(time, Time):
        time = np.array(time.jd, dtype=float)
    else:
        time = np.array(time, dtype=float)

    # Convert period to a plain float in days.
    if hasattr(period, 'to_value'):
        period = period.to_value(u.day)
    else:
        period = float(period)
    
    # Convert t0: if it's a Time object, use .jd; if Quantity, use to_value(u.day); otherwise, float.
    if isinstance(t0, Time):
        t0 = t0.jd
    elif hasattr(t0, 'to_value'):
        t0 = t0.to_value(u.day)
    else:
        t0 = float(t0)
    
    # Convert duration similarly.
    if hasattr(duration, 'to_value'):
        duration = duration.to_value(u.day)
    else:
        duration = float(duration)

    # Compute the phase relative to transit mid-point.
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    return np.abs(phase) < 0.5 * duration


results = BLSfit(preprocessed_lc)
high_periods, high_powers, best_period, t0, duration = BLSResults(results, plot='show')
transit_mask = create_transit_mask_manual(preprocessed_lc.time, best_period, t0, duration)
print(best_period)

period = test_period(results)
duration = test_duration(results)
depth = test_depth(preprocessed_lc.time, preprocessed_lc.flux, transit_mask)
vshape = test_v_shape(preprocessed_lc.time, preprocessed_lc.flux, transit_mask)
snr = test_snr(preprocessed_lc.flux, transit_mask)
oot_variability = test_out_of_transit_variability(preprocessed_lc.flux, transit_mask)

print("Transit Period (days):", period)
print("Transit Duration (days):", duration)
print("Transit Depth:", depth)
print("V-Shape:", vshape)
print("SNR:", snr)
print("OOT Variability:", oot_variability)

# folded_lc = FoldedLC(preprocessed_lc, 1.40793925, 0, plot='show', bin = True)
# preprocessed_lc.scatter()