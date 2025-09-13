import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
injected_file = 'data_outputs/injected_transits_output6.csv'
noninjected_file = 'data_outputs/noninjected_transits_output6.csv'

# Read the CSV files
injected_data = pd.read_csv(injected_file)
noninjected_data = pd.read_csv(noninjected_file)

# Extract the 'snr' column
injected_snr = injected_data['snr']
noninjected_snr = noninjected_data['snr']

# Create custom bins
bins = 30
injected_logbins = np.logspace(np.log10(injected_snr.min()), np.log10(injected_snr.max()), bins)
# noninjected_logbins = np.logspace(np.log10(noninjected_snr.min()), np.log10(noninjected_snr.max()), bins)
noninjected_logbins = injected_logbins

# Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(injected_snr, bins=injected_logbins, alpha=0.7, label='Injected SNR', color='blue')
plt.hist(noninjected_snr, bins=noninjected_logbins, alpha=0.7, label='Non-injected SNR', color='orange')

# Add labels, title, and legend
plt.xscale('log')
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('Count')
plt.title('Histogram of SNR Values')
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('data_outputs/snr_histogram.png', dpi=300)