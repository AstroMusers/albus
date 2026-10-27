"""Albus names for shared Miletos transit simulation and orbital geometry."""

from miletos.simulation import (
    EARTH_RADIUS_M,
    SOLAR_MASS_KG,
    SOLAR_RADIUS_M,
    generate_batman_light_curve as generate_lightcurve,
    inject_batman_transit as inject_transit,
    minimum_transit_inclination,
    semimajor_axis_from_period as calc_a,
)

def inject_transit_df(df, radius_star, mass_star, radius_planet, luminosity_star, albedo_planet, period, inclination, time_array):
    ttime, tflux, tduration = generate_lightcurve(
        radius_star=radius_star,            # Approx. radius of a white dwarf
        mass_star= mass_star,               # Approx. mass of white dwarf
        radius_planet= radius_planet,       # Radius of a typical Hot Jupiter
        albedo_planet=albedo_planet,        # Typical albedo of a gas giant
        period=period,                      # Orbital period
        inclination=inclination,            # Inclination of transit
        time_array=time_array
    )
    df['Flux'] = df['Flux'] + tflux
    return df

