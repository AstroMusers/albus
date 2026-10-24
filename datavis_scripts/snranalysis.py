import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tdpy.verbosity import print

injected = True

# if injected:
injected_path = 'data_outputs/injected_transits_output1.csv'
print(f'Reading from {injected_path}...')
dfi = pd.read_csv(injected_path, header=0)
snr = np.asarray(dfi['snr'], float)
dfi['snr'] = pd.to_numeric(dfi['snr'], errors='coerce')
# snr = dfi.dropna(subset=['snr'])['snr'].values.astype(float)

plt.figure(figsize=(12, 6))
plt.tight_layout()

# plt.subplot(1, 2, 1)

bins = np.arange(0, 10, 0.1)
print(len([ratio for ratio in snr if ratio > 1]))
plt.hist(snr, bins=bins, color='green', edgecolor='black', label='Injected', alpha=0.5, cumulative=False)
plt.xlabel('SNR')
plt.ylabel('Frequency')
# plt.xscale('log')
plt.yscale('log')
plt.title('SNR Distribution for Injected Transits')
# else: 

# plt.subplot(1, 2, 2)
noninjected_path = 'data_outputs/noninjected_transits_output1.csv'
print(f'Reading from {noninjected_path}...')
dfn = pd.read_csv(noninjected_path, header=0)
snr = np.asarray(dfn[' snr'], float)

plt.hist(snr, bins=bins, color='blue', edgecolor='black', label='Non-injected', alpha=0.5, cumulative=False)
plt.xlabel('SNR')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('SNR Distribution for Non-injected Transits')

plt.legend()
plt.show()