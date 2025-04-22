import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from injections import calc_a

df = pd.read_csv('data_outputs/injected_transits_output2.csv')

# Clean the 'snr' column: convert to numeric, drop invalid entries
df['snr'] = pd.to_numeric(df['snr'], errors='coerce')
df = df.dropna(subset=['snr'])
print(df['snr'])

# Ensure IDs are three-digit strings, then split into parameters and multiply by 2
df['ID_str'] = df['ID'].astype(str).str.zfill(3)
df['radius'] = round(((df['ID_str'].str[0].astype(int) + 1)/10)**2,3)    # Parameter 1 → Radius
df['period'] = round(1 + (df['ID_str'].str[1].astype(int) * 10)/9,3)     # Parameter 2 → Period
# df['inclination'] = 90-((df['ID_str'].str[2].astype(int))*
#                     (90-np.arccos((0.01+((df['ID_str'].str[0].astype(int) + 1)/10)**2)
#                     /(calc_a(0.6, (1 + (df['ID_str'].str[1].astype(int) * 10)/9))/6.957*10**8))/np.pi*180)/9)
#                                                                 # Parameter 3 → Inclination

# inc = 90-((df['ID_str'].str[2].astype(int))*
#           (90-np.arccos((0.01+((df['ID_str'].str[0].astype(int) + 1)/10)**2)
#         /(calc_a(0.6, (1 + (df['ID_str'].str[1].astype(int) * 10)/9))/6.957*10**8))/np.pi*180)/9)

# print(calc_a(0.6, (1 + (df['ID_str'].str[1].astype(int) * 10)/9))/(6.957*10**8))

# Prepare averages
avg_radius = df.groupby('radius')['snr'].mean().sort_index()
avg_period = df.groupby('period')['snr'].mean().sort_index()
pivot = df.groupby(['radius', 'period'])['snr'].mean().unstack()

# Single figure with 3 side-by-side subplots
# fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios':[1,1,1.2]})
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1D: Avg SNR vs Radius
axes[0].plot(avg_radius.index, avg_radius.values, marker='o')
axes[0].set_xlabel(r'Radius $(R_{S})$')
axes[0].set_ylabel('Average SNR')
axes[0].set_title('Avg SNR vs Radius')
axes[0].grid(True)

# 1D: Avg SNR vs Period
axes[1].plot(avg_period.index, avg_period.values, marker='o')
axes[1].set_xlabel('Period (days)')
axes[1].set_title('Avg SNR vs Period')
axes[1].grid(True)

# 2D: Heatmap Radius vs Period
# smoothed = gaussian_filter(pivot.values, sigma=0.5)
# im = axes[2].pcolormesh(pivot.columns, pivot.index, smoothed, shading='gourand', cmap='viridis')
im = axes[2].imshow(pivot, aspect='auto', origin='lower', cmap='viridis', interpolation='gaussian')
axes[2].set_xlabel('Period (days)')
axes[2].set_ylabel(r'Radius $(R_{S})$')
axes[2].set_title('Heatmap: Radius vs Period')
axes[2].set_xticks(range(len(pivot.columns)))
axes[2].set_xticklabels(pivot.columns, rotation=45)
axes[2].set_yticks(range(len(pivot.index)))
axes[2].set_yticklabels(pivot.index)

# Colorbar for heatmap
cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
cbar.set_label('Average SNR')

fig.tight_layout()
plt.show()
