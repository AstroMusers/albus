import random

import numpy as np
from tdpy.verbosity import print
from miletos.photometry import preprocess_tess_light_curve as preprocess

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
