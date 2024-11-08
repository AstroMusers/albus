import csv
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

df = pd.read_csv('candidates3.csv', on_bad_lines='skip', header = 0)

period1, period2, powers = [], [], []

for index, row in tqdm(df.iterrows()):
    period1.append(float(row['Highest Period'].split(' ')[0]))
    period2.append(float(row['Second Highest Period']))
    powers.append(row['Highest Power'])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.hist(period1, bins=50)
ax1.set_xlabel('Period (days)')
ax1.set_ylabel('Frequency')
ax1.set_title('Highest Period Distribution of BLS Peaks')

ax2.hist(period2, bins=50)
ax2.set_xlabel('Period (days)')
ax2.set_ylabel('Frequency')
ax2.set_title('Second Highest Period Distribution of BLS Peaks')


ax3.scatter(period1, powers, marker='.')
ax3.set_yscale('log')
# plt.semilogy(period1, powers)
ax3.set_xlabel('Period (days)')
ax3.set_ylabel('BLS Power')
ax3.set_title('Period vs. BLS Power')

ax4.hist(powers, bins=50)
ax4.set_xscale('log')
ax4.set_xlabel('Period (days)')
ax4.set_ylabel('Frequency')
ax4.set_title('Period Distribution of BLS Peaks')
# plt.show()

fig.tight_layout()
plt.show()