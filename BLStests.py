import numpy as np

def test_period(bls_periodogram):
    """
    Extract the best-fit transit period from the BLS periodogram.
    
    Parameters:
        bls_periodogram: An object from lightkurve's BLS which should have attributes
                         like 'period_at_max_power' (best-fit period) and potentially others.
    
    Returns:
        period (float): Best-fit period from the periodogram.
    """
    # best_period = bls_periodogram.period_at_max_power  
    # return best_period

def test_duration(bls_periodogram):
    """
    Extract the best-fit transit duration from the BLS periodogram.
    
    Parameters:
        bls_periodogram: BLS periodogram object expected to contain a duration metric,
                         e.g., 'duration_at_max_power'.
    
    Returns:
        duration (float): Best-fit transit duration.
    """
    duration = bls_periodogram.duration_at_max_power  
    return duration

def test_depth(detrended_time, detrended_flux, transit_mask):
    """
    Measure the transit depth from the detrended lightcurve.
    
    Assumes that the lightcurve is normalized (e.g., out-of-transit flux ~1).
    The transit_mask is a boolean array marking which points are considered in-transit.
    
    Parameters:
        detrended_time (np.array): Array of time values.
        detrended_flux (np.array): Array of normalized flux values.
        transit_mask (np.array of bool): Boolean mask indicating in-transit points.
    
    Returns:
        depth (float): Relative transit depth (e.g., if flux drops to 0.8, depth = 0.2).
    """
    in_transit_flux = detrended_flux[transit_mask]

    depth = 1 - np.median(in_transit_flux)
    return depth

def test_v_shape(detrended_time, detrended_flux, transit_mask):
    """
    Compute a V-shape metric to test for V-shaped transits.
    
    The idea here is to estimate the ratio of the ingress/egress durations 
    to the total transit duration. For a box-shaped (flat-bottom) transit, 
    this ratio is low, whereas for a V-shaped transit, the ingress and egress
    dominate the transit.
    
    This implementation uses simple flux thresholding:
    - It computes the transit depth.
    - It defines flux thresholds (e.g., 10% and 90% of the depth) to approximate the 
      boundaries of ingress and egress.
    
    Parameters:
        detrended_time (np.array): Array of time values.
        detrended_flux (np.array): Array of normalized flux values.
        transit_mask (np.array of bool): Boolean mask for in-transit points.
    
    Returns:
        v_shape_metric (float): Ratio (ingress + egress duration)/total transit duration.
                                Values closer to 1 indicate a more V-shaped transit.
    """
    time_in_transit = detrended_time[transit_mask]
    flux_in_transit = detrended_flux[transit_mask]
    
    # Calculate transit depth
    depth = 1 - np.median(flux_in_transit)
    
    # Define thresholds for estimating ingress/egress boundaries.
    flux_10 = 1 - 0.1 * depth
    flux_90 = 1 - 0.9 * depth
    
    # Estimate ingress: the time it takes for flux to drop from baseline to near transit bottom.
    ingress_candidates = np.where(flux_in_transit > flux_10)[0]
    # Estimate egress: the time from near transit bottom back to baseline.
    egress_candidates = np.where(flux_in_transit > flux_90)[0]
    
    if len(ingress_candidates) == 0 or len(egress_candidates) == 0:
        # If thresholds cannot be applied, return NaN to indicate an indeterminate metric.
        return np.nan
    
    t_ingress = time_in_transit[ingress_candidates[-1]] - time_in_transit[0]
    t_egress = time_in_transit[-1] - time_in_transit[egress_candidates[0]]
    total_duration = time_in_transit[-1] - time_in_transit[0]
    
    # v_shape_metric: higher values suggest that most of the transit is spent in ingress/egress,
    # which is indicative of a V-shaped transit.
    v_shape_metric = (t_ingress + t_egress) / total_duration
    return v_shape_metric

def test_snr(detrended_flux, transit_mask):
    """
    Compute the signal-to-noise ratio (SNR) of the transit.
    
    The SNR is defined here as the transit depth divided by the standard deviation 
    of the out-of-transit flux.
    
    Parameters:
        detrended_flux (np.array): Array of normalized flux values.
        transit_mask (np.array of bool): Boolean mask for in-transit points.
    
    Returns:
        snr (float): The calculated signal-to-noise ratio.
    """
    depth = 1 - np.median(detrended_flux[transit_mask])
    noise = np.std(detrended_flux[~transit_mask])
    snr = depth / noise if noise > 0 else np.nan
    return snr

def test_out_of_transit_variability(detrended_flux, transit_mask):
    """
    Evaluate the variability of the out-of-transit flux.
    
    For white dwarfs, we expect minimal variability outside of transit.
    This function returns the standard deviation of the out-of-transit data.
    
    Parameters:
        detrended_flux (np.array): Array of normalized flux values.
        transit_mask (np.array of bool): Boolean mask for in-transit points.
    
    Returns:
        variability (float): Standard deviation of the out-of-transit flux.
    """
    variability = np.std(detrended_flux[~transit_mask])
    return variability

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
