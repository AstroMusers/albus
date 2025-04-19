import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

injected = True

# if injected:
dfi = pd.read_csv('data_outputs/injected_transits_output1.csv', header=0)
snr = np.asarray(dfi[' snr'], float)

plt.figure(figsize=(12, 6))
plt.tight_layout()

# plt.subplot(1, 2, 1)

bins = np.arange(0, 1, 0.01)
print(len([ratio for ratio in snr if ratio > 1]))
plt.hist(snr, bins=bins, color='green', edgecolor='black', label='Injected', alpha=0.5, cumulative=False)
plt.xlabel('SNR')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('SNR Distribution for Injected Transits')
# else: 

# plt.subplot(1, 2, 2)
dfn = pd.read_csv('data_outputs/noninjected_transits_output1.csv', header=0)
snr = np.asarray(dfn[' snr'], float)

plt.hist(snr, bins=bins, color='blue', edgecolor='black', label='Non-injected', alpha=0.5, cumulative=False)
plt.xlabel('SNR')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('SNR Distribution for Non-injected Transits')

plt.legend()
plt.show()