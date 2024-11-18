import batman
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

def calc_a(mass_star, period_days):
    period_seconds = period_days * 24 * 3600
    a = ((6.6743 * 10**-11 * mass_star * period_seconds**2) / (4 * np.pi**2)) ** (1/3)
    return a # in meters

def generate_lightcurve(
    radius_star,                # Radius of the star (in solar radii)
    mass_star,                  # Mass of the star
    radius_planet,              # Radius of the planet (in Jupiter radii)
    luminosity_star,            # Luminosity of the star (arbitrary unit or Solar units)
    albedo_planet,              # Albedo of the planet
    period,                     # Orbital period of the planet (in days)
    inclination,                # Inclination of orbit
    # t0,
    # time_span,                  # Total time span for observation (in days)
    time_array,
    time_resolution=2/(60*24)):
    
    star_radius_meters = radius_star * 6.957*10**8  # Solar radius to meters
    planet_radius_meters = radius_planet * 6.9911*10**7  # Jupiter radius to meters

    # Create a time array
    if isinstance(time_array, float): 
        time = np.array([time_array])
    else: time = np.array(time_array)

    a = calc_a(mass_star, period)
    # Initialize batman parameters
    params = batman.TransitParams()
    if time.size == 1:
        params.t0 = time   # Mid-transit time
    else: 
        params.t0 = min(time)
    params.per = period                   # Planet orbital period
    params.rp = radius_planet             # Planet radius
    params.a = a/(6.957*10**8)            # Semi-major axis in stellar radii
    params.inc = inclination              # Inclination in degrees
    params.ecc = 0                        # Eccentricity
    params.w = 90                         # Longitude of periastron (unused for circular orbits)
    params.u = []                         # Limb-darkening coefficients
    params.limb_dark = "uniform"          # Limb-darkening model

    # Generate the light curve model
    m = batman.TransitModel(params, time)
    flux = m.light_curve(params)

    # Adjust for luminosity and albedo?
    # flux *= luminosity_star * (1 - albedo_planet)

    # Normalize the flux
    flux = flux/np.median(flux)

    # Calculate the transit duration
    b = (a*np.cos(np.radians(inclination)))/radius_star
    # tduration = (period/np.pi)*np.arcsin((radius_star+radius_planet)/a + np.sqrt(1 - b**2))
    tduration = 0

    # tess_time = np.arange(0, time[-1], 120/(24*3600))
    # tess_flux = np.interp(tess_time, time, flux)
    # tess_time = tess_time + t0

    return time, flux, tduration

def inject_transit(
        ID, tic_id, lc, time_array,
        radius_star = 0.01, 
        mass_star = 0.6 * 2 * 10**30, 
        radius_planet = 0.1, 
        luminosity_star=0.001,
        albedo_planet=0.1, 
        period=1, 
        inclination=90):
    
    ttime, tflux, tduration = generate_lightcurve(
        radius_star=radius_star,            # Approx. radius of a white dwarf
        mass_star= mass_star, # Approx. mass of white dwarf
        radius_planet= radius_planet,          # Radius of a typical Hot Jupiter
        luminosity_star=luminosity_star,       # White dwarf luminosity in Solar units
        albedo_planet=albedo_planet,           # Typical albedo of a gas giant
        period=period,                    # Orbital period
        inclination=inclination,              # Inclination of transit
        time_array=time_array
    )

    inj = lk.LightCurve(time=ttime, flux=tflux+lc['flux'].value)

    inj.scatter()
    plt.xlabel("Time (days)")
    plt.ylabel("Relative Flux")
    plt.title(f"ID {ID} with TIC {tic_id} Light Curve")
    plt.savefig(f'/Users/aavikwadivkar/Documents/Exoplanets/Research/WD_Plots4/{ID}_injectedlc.png')
    plt.close()
    return inj

def inject_transit_df(df, radius_star, mass_star, radius_planet, luminosity_star, albedo_planet, period, inclination, time_array):
    ttime, tflux, tduration = generate_lightcurve(
        radius_star=radius_star,            # Approx. radius of a white dwarf
        mass_star= mass_star, # Approx. mass of white dwarf
        radius_planet= radius_planet,          # Radius of a typical Hot Jupiter
        luminosity_star=luminosity_star,       # White dwarf luminosity in Solar units
        albedo_planet=albedo_planet,           # Typical albedo of a gas giant
        period=period,                    # Orbital period
        inclination=inclination,              # Inclination of transit
        time_array=time_array
    )
    df['Flux'] = df['Flux'] + tflux
    return df

