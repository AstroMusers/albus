import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import astropy
import os
from scipy.signal import medfilt

df = pd.read_csv('tess_targets.csv', on_bad_lines='skip', header = 0)
output_file = 'blsfit3.csv'
cand_file = 'candidates3.csv'
cands = pd.read_csv('candidates3.csv', on_bad_lines='skip', header = 0)


with open(output_file, 'a', newline='') as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(['Target ID', 'Highest Period', 'Highest Power', 
                     'Second Highest Period', 'Second Highest Power'])
    f.close()

with open(cand_file, 'a', newline='') as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(['Target ID', 'Highest Period', 'Highest Power', 
                     'Second Highest Period', 'Second Highest Power'])
    f.close()

def BLSfit(tic_id):
    # Download the TESS light curve for a specific TIC ID
    search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS', exptime=120)
    print("search has resulted")

    try:
        lcc = search_result.download_all()
    except:
        return
    print("downloaded")

    if lcc is None:
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

    trend = medfilt(lc.flux, kernel_size=101)
    detrended_flux = lc.flux - trend
    detrended_lc = lk.LightCurve(time=lc.time, flux=detrended_flux)
    flatlc = detrended_lc

    # flatlc.scatter()
    # plt.title('flattened scatter')
    # plt.show()
    print('flattened')

    # Perform Box Least Squares periodogram
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

    num_bins = 64
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

    second_max_index = np.argsort(powers)[-2]  # Gets the index of the second-highest power
    second_highest_power_period = periods[second_max_index]

    sorted_indices = np.argsort(powers)[::-1]

    for idx in sorted_indices[1:]:
        if periods[idx] < 0.9 * best_period.value or periods[idx] > 1.1 * best_period.value:
            second_highest_power = powers[idx]
            second_highest_power_period = periods[idx]
            break

    # Print results
    # print("Period with highest power:", highest_power_period, "with power:", highest_power)
    # if second_highest_power is not None:
    #     print("Period with second-highest power (meeting threshold):", second_highest_power_period, "with power:", second_highest_power)
    # else:
    #     print("No second-highest power found that meets the threshold.")


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

    # print(len(bin_centers))
    # print(bin_means)
    # print(bin_stds)

    plt.semilogx(periods, powers, label=f'Periodogram of TIC {tic_id}')
    plt.semilogx(bin_centers, bin_means, 'r-', label='Mean')

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
    plt.title(f'BLS Periodogram for TIC {tic_id}')
    plt.savefig(f'/Users/aavikwadivkar/Documents/Exoplanets/Research/WD_Plots3/{tic_id}_blsplot.png')
    plt.close()

    # Fold the light curve at the best period
    folded_lc = flatlc.fold(period=best_period, epoch_time = t0)

    # Plot the folded light curve
    folded_lc.scatter()
    plt.xlabel('Phase [JD]')
    plt.ylabel('Normalized Flux')
    plt.title(f'TIC {tic_id} Folded Light Curve at Period = {str(round(best_period.value, 3))} days')
    plt.savefig(f'/Users/aavikwadivkar/Documents/Exoplanets/Research/WD_Plots3/{tic_id}_foldedlc.png')
    plt.close()

    # binned_lc = folded_lc.bin(time_bin_size=0.001)

    # binned_lc.scatter(label='Binned Data')
    # plt.xlabel('Phase [JD]')
    # plt.ylabel('Normalized Flux')
    # plt.title(f'TIC {tic_id} Binned Folded Light Curve at Period = {str(round(best_period.value, 3))} days')
    # plt.savefig(f'/Users/aavikwadivkar/Documents/Exoplanets/Research/WD_Plots2/{tic_id}_binnedlc.png')
    # plt.close()

    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([tic_id, results.period[index], results.power[index]])
        f.close()

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    maxindex = np.where(bin_centers == find_nearest(bin_centers, best_period.value))[0][0]

    if best_period.value > 1.1 and results.power[index] > bin_means[maxindex] + 3*bin_stds[maxindex]:
        with open(cand_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([tic_id, results.period[index], results.power[index], second_highest_power_period, second_highest_power])
            f.close()

for target in tqdm(df['Target ID']):
    here = 0
    print(target)
    for file in os.listdir('/Users/aavikwadivkar/Documents/Exoplanets/Research/WD_Plots3'):
        if str(target) == str(file.split('_')[0]):
            print('already here')
            here = 1
            break
    if here == 0:
        print(f'searching for {target}')
        BLSfit(target)