import pandas as pd
import matplotlib.pyplot as plt

# File path
file_path = 'data_outputs/injected_transits_output6.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# Calculate Radius Ratio
data['Radius Ratio'] = data['r_p'] * 6400000 / data['r_s']

# Extract the required columns
radius_ratio = data['Radius Ratio']
period = data['P_days']
inclination = data['inc']

# Create subplots for the histograms
fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=False, sharey=True)

# Plot Radius Ratio histogram
axes[0].hist(radius_ratio, bins=30, color='blue', alpha=0.7)
axes[0].set_title('Radius Ratio')
axes[0].set_xlabel('Radius Ratio')
axes[0].set_ylabel('Frequency')

# Plot Period histogram
axes[1].hist(period, bins=30, color='green', alpha=0.7)
axes[1].set_title('Period')
axes[1].set_xlabel('Period')

# Plot Inclination histogram
axes[2].hist(inclination, bins=30, color='red', alpha=0.7)
axes[2].set_title('Inclination')
axes[2].set_xlabel('Inclination')

# Set y-axis to log scale for all subplots
for ax in axes:
    ax.set_yscale('log')

plt.suptitle('Histograms of Radius Ratio, Period, and Inclination')


# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig('data_outputs/run6analysis/histograms_radius_period_inclination6.png', dpi=300)
plt.show()