import pandas as pd
import matplotlib.pyplot as plt

# Load your data (replace 'data.csv' with your filename)
df = pd.read_csv('data_outputs/injected_transits_output1.csv')

# Clean the 'snr' column: convert to numeric, drop invalid entries
df['snr'] = pd.to_numeric(df['snr'], errors='coerce')
df = df.dropna(subset=['snr'])

# Ensure IDs are three-digit strings, then split into parameters and multiply by 2
df['ID_str'] = df['ID'].astype(str).str.zfill(3)
df['radius'] = round((1 - (df['ID_str'].str[0].astype(int))/5), 3)    # Parameter 1 → Radius
df['period'] = round(1 + (df['ID_str'].str[1].astype(int) * 10)/5, 3)      # Parameter 2 → Period
df['inclination'] = round(90-(df['ID_str'].str[2].astype(int)/5), 3) # Parameter 3 → Inclination

# 1D plots: average SNR versus each single parameter in a single figure
params = ['radius', 'period', 'inclination']
labels = [r'Radius $(R_{J})$', 'Period (days)', 'Inclination (deg)']
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, param, label in zip(axes, params, labels):
    grp = df.groupby(param)['snr'].mean().sort_index()
    ax.plot(grp.index, grp.values, marker='o')
    ax.set_xlabel(label)
    ax.set_title(f'Avg SNR vs {label}')
    ax.grid(True)
axes[0].set_ylabel('Average SNR')
fig.tight_layout()
plt.show()

# 2D heatmaps: average SNR for each pair of parameters in a single figure
pairs = [(r'Radius $(R_{J})$', 'Period (days)'), (r'Radius $(R_{J})$', 'Inclination (degs)'), ('Period (days)', 'Inclination (degs)')]
titles = ['Radius vs Period', 'Radius vs Inclination', 'Period vs Inclination']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (a, b), title in zip(axes, pairs, titles):
    pivot = df.groupby([a, b])['snr'].mean().unstack(fill_value=float('nan'))
    im = ax.imshow(pivot.values, aspect='auto')
    ax.set_xlabel(f'{b}')
    ax.set_ylabel(f'{a}')
    ax.set_title(f'Heatmap: {title}')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, label='Avg SNR')
    ax.axhline(y=0, color='k', lw=1)
fig.tight_layout()
plt.legend()
plt.show()
