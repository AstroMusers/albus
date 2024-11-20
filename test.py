from injections import generate_lightcurve
import numpy as np
from matplotlib import pyplot as plt
import astropy
from scipy.signal import medfilt
import lightkurve as lk

time, flux, tduration = generate_lightcurve(
    radius_star=0.01,            # Approx. radius of a white dwarf
    mass_star= 0.6 * 2 * 10**30, # Approx. mass of white dwarf
    radius_planet= 0.01,          # Radius of a typical Hot Jupiter
    luminosity_star=0.001,       # White dwarf luminosity in Solar units
    albedo_planet=0.1,           # Typical albedo of a gas giant
    period=5,                    # Orbital period
    inclination=90,              # Inclination of transit
    time_array=np.arange(0, 10, 0.01)
)

flatlc = lk.LightCurve(time=time, flux=flux)
flatlc.scatter()
plt.show()
print('generated')

maxday = 10
minday = 1
periods = np.logspace(np.log10(minday), np.log10(maxday), num=2048)
durations = []
for period in periods:
    durations.append(period*0.01)

# Replace with astropy
pg = astropy.timeseries.BoxLeastSquares(flatlc['time'], flatlc['flux'])
results = pg.power(periods, durations)
print("bls")
# print(results.period)

# Find the period with the highest power
index = np.argmax(results.power)
best_period = results.period[index]
t0 = results.transit_time[index]
duration = results.duration[index]
# print(f'best period: {best_period}')

num_bins = 48
bins = np.logspace(np.log10(1), np.log10(10), num_bins + 1)

# Calculate the mean and standard deviation for each bin
bin_centers = []
bin_means = []
bin_stds = []

periods = np.array(results.period.value)
powers = np.array(results.power.value)

ptrend = medfilt(powers, kernel_size=101)
detrended_power = powers - ptrend
powers = detrended_power

high_powers = [results.power[index].value]
high_periods = [best_period.value]
sorted_indices = np.argsort(powers)[::-1]

# Proportional threshold
lower_bound = 0.8  # Lower bound multiplier
upper_bound = 1.2  # Upper bound multiplier

sorted_powers = powers[sorted_indices]
sorted_periods = periods[sorted_indices]
print(sorted_periods[:20])

for speriod, spower in zip(sorted_periods[1:], sorted_powers[1:]):
    add_to_high = True
    for hperiod in high_periods:
        if lower_bound * hperiod <= speriod <= upper_bound * hperiod:
            add_to_high = False
            break
    if add_to_high:
        high_periods.append(speriod)
        high_powers.append(spower)
    if len(high_periods) >= 4:  # Stop after 4 clusters
        break

x, y = [], []

# print("High Periods (up to 4 clusters):", high_periods)
# print("Corresponding Powers:", high_powers)
for i in range(len(high_periods)):
    for j in range(len(high_periods)):
        if i != j:
            print(f'Ratio P_{i}/P_{j}: ', high_periods[i]/high_periods[j])
            x.append(high_periods[i]/high_periods[j])
            print(f'Ratio P_{i}/P_{j}: ', high_powers[i]/high_powers[j])
            y.append(high_powers[i]/high_powers[j])

plt.plot(x, y, 'o')
plt.plot([0, 5], [0, 5], 'r--')
plt.xlabel('Period Ratio')
plt.ylabel('Power Ratio')
plt.title('Period Ratio vs Power Ratio of Periodogram Peaks')
plt.show()

print(x)
print(y)

for i in range(num_bins):
    # Mask for the current bin
    bin_mask = (periods >= bins[i]) & (periods < bins[i + 1])
    
    # Extract values within the current bin
    bin_periods = periods[bin_mask]
    bin_powers = powers[bin_mask]
    
    # Calculate the 1.5 IQR range for outlier filtering
    q1, q3 = np.percentile(bin_powers, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out outliers
    filtered_powers = bin_powers[(bin_powers >= lower_bound) & (bin_powers <= upper_bound)]
    
    # Calculate the bin center, mean, and standard deviation without outliers
    bin_centers.append(np.median(bin_periods))  # Center of each bin
    bin_means.append(np.median(filtered_powers))  # Mean of y-values in each bin
    bin_stds.append(np.std(filtered_powers))  # Std deviation of y-values in each bin

bin_centers = np.array(bin_centers)
bin_means = np.array(bin_means)
bin_stds = np.array(bin_stds)

plt.semilogx(periods, powers, label=f'Periodogram of Ideal Transit')
plt.semilogx(bin_centers, bin_means, 'r-', label='Median')

for n in range(5):
    if best_period.value/(n+1) > 1:
        plt.axvline(best_period.value/(n+1), ls=':', color='r', alpha=0.5)
    if best_period.value*(n+1) < 10:
        plt.axvline(best_period.value*(n+1), ls=':', color='r', alpha=0.5)

plt.fill_between(bin_centers, bin_means - 1*bin_stds, bin_means + 1*bin_stds, color='gray', alpha=0.3, label='1σ Band')
plt.fill_between(bin_centers, bin_means - 2*bin_stds, bin_means + 2*bin_stds, color='gray', alpha=0.3, label='2σ Band')
plt.fill_between(bin_centers, bin_means - 3*bin_stds, bin_means + 3*bin_stds, color='gray', alpha=0.3, label='3σ Band')

# Labels and legend
plt.xlabel('Period')
plt.ylabel('Power')
plt.legend()
plt.title(f'BLS Periodogram for Ideal Transit')
# plt.savefig(f'/Users/aavikwadivkar/Documents/Exoplanets/Research/WD_Plots4/{tic_id}_blsplot.png')
plt.show()
plt.close()

# Fold the light curve at the best period
folded_lc = flatlc.fold(period=best_period, epoch_time = t0)

# Plot the folded light curve
folded_lc.scatter()
plt.xlabel('Phase [JD]')
plt.ylabel('Normalized Flux')
plt.title(f'Ideal Folded Light Curve at Period = {str(round(best_period.value, 3))} days')
# plt.savefig(f'/Users/aavikwadivkar/Documents/Exoplanets/Research/WD_Plots4/{tic_id}_foldedlc.png')
plt.show()
plt.close()