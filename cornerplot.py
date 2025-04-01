import pandas as pd
import corner
import matplotlib.pyplot as plt

df = pd.read_csv('Pipeline/injected_transits_datapoints.csv', header=1)

df['best_period'] = df['best_period'].apply(lambda x: float(x.replace(' d', '').strip()))
df['duration'] = df['duration'].apply(lambda x: float(x.replace(' d', '').strip()))

data = df[['best_period', 'duration', 'depth', 'vshape', 'snr', 'oot_variability']].values

labels = ['Best Period (days)', 'Duration (days)', 'Depth', 'V-shape', 'SNR', 'OOT Variability']

figure = corner.corner(data, labels=labels, show_titles=True)
plt.show()
