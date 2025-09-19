from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from astroquery.mast import Catalogs, Tesscut
from tqdm import tqdm
import csv

# WDlist = fits.open('~/gaiaedr3_wd_main.fits')
WDlist = fits.open('../../gaiaedr3_wd_main.fits')
data = WDlist[1].data

# df = pd.read_csv('~/albus/albus/data_inputs/tess_targets_data.csv', on_bad_lines='skip', header = 0)
df = pd.read_csv('data_inputs/tess_targets_data.csv', on_bad_lines='skip', header = 0)


# Magcut = data[(data['phot_g_mean_mag_corrected'] < 16)]
# Probcut = Magcut[(Magcut['Pwd'] > 0.9)]
# Flatcut = Probcut[(Probcut['bright_N_flag'] == 0)]

Probcut = data[(data['Pwd'] > 0.9)]
Magcut = data[(data['phot_g_mean_mag_corrected'] < 16)]
Flatcut = data[(data['bright_N_flag'] == 0)]

print(len(Probcut))
print(len(Magcut))
print(len(Flatcut))
print(len(df))

totalcut = data[(data['Pwd'] > 0.9) & (data['bright_N_flag'] == 0) & (data['phot_g_mean_mag_corrected'] < 16)]
# Probility, Bright neighbor flag, photometric cutoff
# chose to save time (for now)

# To calculate abs. magnitude: M = m + 5log(parallax/100)

cuts = [Magcut, Probcut, Flatcut, totalcut]

# for cut in cuts:
#     print(f'Cut is length {len(cut)}')

# plt.plot(data['bp_rp'], data['absG'], '.', label='All WDs', color='black', alpha=0.01)
vallims = [min(data['bp_rp']), max(data['bp_rp']), 
           min(data['absG']) , max(data['absG'])]

print(vallims)
print(len(data))
# plot = np.histogram2d(data['bp_rp'], data['absG'], bins=200)[0]
# print(plot)
# plt.hist2d(data['bp_rp'], data['absG'], bins=100, cmap='plasma', norm =  'log')
# plot = gaussian_filter(plot, sigma=3)
# plot = [plot[0], plot[1]]
# plt.pcolormesh(plot, cmap='viridis', norm='log')

# plt.colorbar()
# plt.hist2d(Probcut['bp_rp'], Probcut['absG'], bins=100, cmap='plasma', norm =  'log')
# plt.colorbar()

def scatterplot():
    plt.scatter(data['bp_rp'], data['absG'], alpha = 0.1, color = 'k', marker='.', label = 'GAIA EDR3 WD Candidates')
    plt.xlabel('Bp - Rp')
    plt.ylabel('Absolute G-band Magnitude')
    plt.gca().invert_yaxis()
    # plt.gca().invert_xaxis()
    plt.title('HR Diagram of White Dwarf Candidates')
    # plt.legend()
    # plt.savefig(f'HR Diagram WD Candidates {i}.png', dpi=300)

scatterplot()
plt.scatter(Probcut['bp_rp'], Probcut['absG'], alpha = 0.3, color = 'r', marker='.', label = 'Probability Cut')
plt.legend()
plt.savefig(f'HR Diagram WD Candidates (Prob).png', dpi=300)
plt.close()

scatterplot()
plt.scatter(Magcut['bp_rp'], Magcut['absG'], alpha = 0.3, color = 'b', marker='.', label = 'Apparent Magnitude Cut')
plt.legend()
plt.savefig(f'HR Diagram WD Candidates (Mag).png', dpi=300)
plt.close()

scatterplot()
plt.scatter(df['BP-RP'], df['GMAG'], alpha=0.3, color = 'g', marker = '.', label = 'TESS Lightcurve Available')
plt.legend()
plt.savefig(f'HR Diagram WD Candidates (TESS).png', dpi=300)
plt.close()


scatterplot()
plt.scatter(totalcut['bp_rp'], totalcut['absG'], alpha = 0.3, color = 'm', marker='.', label = 'All Cuts')
plt.legend()
plt.savefig(f'HR Diagram WD Candidates (All).png', dpi=300)
plt.close()

# plt.scatter(Magcut['bp_rp'], Magcut['absG'], alpha = 0.3, color = 'b', marker='.', label = 'Apparent Magnitude Cut')


# Check in TESS data is available
# valids = []
# with open('TESS_WD.csv', 'w') as csvfile:     
#     csvwriter = csv.writer(csvfile)    
#     csvwriter.writerow(['RA', 'DEC'])

# cut = cut[1404:]
