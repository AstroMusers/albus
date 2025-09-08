import pandas as pd

import matplotlib.pyplot as plt

# File paths
injected_file = 'data_outputs/injected_transits_output4.csv'
noninjected_file = 'data_outputs/noninjected_transits_output4.csv'

# Read the CSV files
injected_data = pd.read_csv(injected_file)
noninjected_data = pd.read_csv(noninjected_file)

# Extract the 'snr' column
injected_snr = injected_data['snr']
noninjected_snr = noninjected_data['snr']

# Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(injected_snr, bins=30, alpha=0.7, label='Injected SNR', color='blue')
plt.hist(noninjected_snr, bins=30, alpha=0.7, label='Non-injected SNR', color='orange')

# Add labels, title, and legend
plt.xlabel('SNR')
plt.ylabel('Count')
plt.title('Histogram of SNR Values')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()