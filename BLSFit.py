import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import astropy
import os
from scipy.signal import medfilt

def BLSfit(flatlc):
    # Perform Box Least Squares periodogram
    maxday = 10
    minday = 1
    periods = np.logspace(np.log10(minday), np.log10(maxday), num=2048)
    durations = []
    for period in periods:
        durations.append(period*0.01)

    # Replace with astropy
    # pg = astropy.timeseries.BoxLeastSquares(flatlc['time'], flatlc['flux'])
    # results = pg.power(periods, durations)
    # print(len(np.array(periods)))
    try:
        pg = flatlc.to_periodogram(method='bls', period=np.array(periods), duration=np.array(durations), frequency_factor=5000)
    except Exception as e:
        return f"Error in BLS fit: {e}"
    # results = pg.flatten()

    # print("bls")
    # return results
    return pg

def BLSResults(results, folder='', ID='', plot=''):

    # Find the period with the highest power
    index = np.argmax(results.power)
    best_period = results.period[index]
    t0 = results.transit_time[index]
    duration = results.duration[index]

    periods = np.array(results.period.value)
    powers = np.array(results.power.value)

    # Detrend the periodogram
    ptrend = medfilt(powers, kernel_size=101)
    detrended_power = powers - ptrend
    powers = detrended_power

    # Find the highest power peaks
    sorted_indices = np.argsort(powers)[::-1]
    high_powers = [results.power[index].value]
    high_periods = [best_period.value]

    # Proportional threshold
    lower_bound = 0.8  # Lower bound multiplier
    upper_bound = 1.2  # Upper bound multiplier

    sorted_powers = powers[sorted_indices]
    sorted_periods = periods[sorted_indices]

    if plot!='':
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
        # Plot the BLS periodogram
        num_bins = 48
        bins = np.logspace(np.log10(1), np.log10(10), num_bins + 1)

        # Calculate the mean and standard deviation for each bin
        bin_centers = []
        bin_means = []
        bin_stds = []

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

        plt.semilogx(periods, powers, label=f'Periodogram of {ID}')
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
        plt.title(f'BLS Periodogram for {ID}')
        if plot=='save': plt.savefig(f'{folder}/{ID}_blsplot.png')
        if plot=='show': plt.show()
        plt.close('all')

    return high_periods, high_powers, best_period, t0, duration
    

def FoldedLC(flatlc, best_period, t0, plot='', ID='', folder='', bin = False, time_bin_size = 0.001, output = False):

    folded_lc = flatlc.fold(period=best_period, epoch_time = t0)
    try: period = best_period.value
    except: period = best_period
    # print(f"Folded period: {period}")
    if bin:
        # binned_lc = folded_lc.bin(bins=bins)
        binned_lc = folded_lc.bin(time_bin_size=time_bin_size)
        return binned_lc
    if plot!='': # Plot the folded light curve
        if bin: binned_lc.plot(label='Binned Data', color='red') 
        else: folded_lc.scatter()
        plt.xlabel('Phase [JD]')
        plt.ylabel('Normalized Flux')
        rounded_period = str(round(period, 3))
        plt.title(f'ID {ID} Folded Light Curve at Period = {rounded_period} days')
        if plot=='save': plt.savefig(f'{folder}/{ID}_{rounded_period}_foldedlc.png')
        if plot=='show': plt.show()
    plt.close('all')
    if output: return folded_lc
    return

def BLSOutput(ID, tic_id, high_periods, high_powers, output_file):
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['ID', 'TIC ID', 'Periods', 'Powers'])
        writer.writerow([ID, tic_id, high_periods, high_powers])
        f.close()

def BLSTestOutputs(ID, tic_id, period, duration, depth, vshape, snr, oot_variability, output_file):
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['ID', 'TIC ID', 'Period', 'Duration', 'Depth', 'V-Shape', 'SNR', 'OOT Variability'])
        writer.writerow([ID, tic_id, period, duration, depth, vshape, snr, oot_variability])
        f.close()