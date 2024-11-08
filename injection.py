import batman
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def calc_a(mass_star, period_days):
    period_seconds = period_days * 24 * 3600
    a = ((6.6743 * 10**-11 * mass_star * period_seconds**2) / (4 * np.pi**2)) ** (1/3)
    return a # in meters

def generate_lightcurve(
    radius_star,            # Radius of the star (in solar radii)
    mass_star,              # Mass of the star
    radius_planet,          # Radius of the planet (in Jupiter radii)
    luminosity_star,        # Luminosity of the star (arbitrary unit or Solar units)
    albedo_planet,          # Albedo of the planet
    period,                 # Orbital period of the planet (in days)
    inclination,            # Inclination of orbit
    time_span,              # Total time span for observation (in days)
    time_resolution=1000    # Number of time points
):
    star_radius_meters = radius_star * 6.957*10**8  # Solar radius to meters
    planet_radius_meters = radius_planet * 6.9911*10**7  # Jupiter radius to meters

    # Create a time array
    time = np.linspace(0, time_span, time_resolution)
    a = calc_a(mass_star, period)
    # Initialize batman parameters
    params = batman.TransitParams()
    params.t0 = time_span / 2             # Mid-transit time
    params.per = period                   # Planet orbital period
    params.rp = radius_planet             # Planet radius
    params.a = a/(6.957*10**8)            # Semi-major axis in stellar radii
    params.inc = inclination              # Inclination in degrees
    params.ecc = 0                        # Eccentricity
    params.w = 90                         # Longitude of periastron (unused for circular orbits)
    params.u = [0.1]                      # Limb-darkening coefficients
    params.limb_dark = "linear"           # Limb-darkening model

    # Generate the light curve model
    m = batman.TransitModel(params, time)
    flux = m.light_curve(params)

    # Adjust for luminosity and albedo
    flux *= luminosity_star * (1 - albedo_planet)

    b = (a*np.cos(np.radians(inclination)))/radius_star
    tduration = (period/np.pi)*np.arcsin((radius_star+radius_planet)/a + np.sqrt(1 - b**2))

    # Plot the generated light curve
    return time, flux, tduration

time, flux, tduration = generate_lightcurve(
    radius_star=0.01,            # Approx. radius of a white dwarf
    mass_star= 0.6 * 2 * 10**30, # Approx. mass of white dwarf
    radius_planet= 0.1,          # Radius of a typical Hot Jupiter
    luminosity_star=0.001,       # White dwarf luminosity in Solar units
    albedo_planet=0.1,           # Typical albedo of a gas giant
    period=1,                    # Orbital period
    inclination=90,              # Inclination of transit
    time_span=10                 # Observation window in days
)

tess_time = np.arange(0, time[-1], 120/(24*3600))

tess_flux = np.interp(tess_time, time, flux)

print(tduration)

# Plot the interpolated lightcurve
plt.plot(time, flux, label="Original Lightcurve", alpha=0.5)
plt.plot(tess_time, tess_flux, label="Interpolated to TESS Cadence", marker='o', markersize=2, linestyle='None')
plt.xlabel("Time (days)")
plt.ylabel("Relative Flux")
plt.title("White Dwarf with Hot Jupiter Transit (TESS-like Cadence)")
plt.legend()
plt.show()