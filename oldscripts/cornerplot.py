from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import corner
from tqdm import tqdm

WDlist = fits.open('/Users/aavikwadivkar/Documents/Exoplanets/Research/gaiaedr3_wd_main.fits')
FITSdata = WDlist[1].data

data = []

params = ['parallax', 'Pwd', 'phot_g_mean_mag_corrected', 'absG', 'bp_rp', 'teff_H', 'mass_H']


for row in tqdm(FITSdata):
    datrow = []
    for param in params:
        if np.isnan(row[param]):
            # print('nan')
            datrow = []
            break
        else:
            # print(param)
            if param == 'parallax':
                datrow.append(np.log10(row[param]))
            else:
                datrow.append(row[param])
    if datrow != []:
        data.append(datrow)

print(len(data))
data = np.array(data)

fig = corner.corner(data, show_titles=True, labels=params)
plt.savefig('uncutpopulation.png')

df = pd.read_csv('tess_targets_data.csv', names=['Target ID', 'RA', 'DEC', 'Plx', 'Pwd', 'GmagCorr', 'GMAG', 'BP-RP', 
                     'TeffH', 'MassH', 'TeffHe', 'MassHe', 'Teffmix', 'Massmix'], on_bad_lines='skip', header = 1)

cdata = []

params = ['Plx', 'Pwd', 'GmagCorr', 'GMAG', 'BP-RP', 'TeffH', 'MassH']

for index, row in tqdm(df.iterrows()):
    datrow = []
    for param in params:
        if np.isnan(row[param]):
            # print('nan')
            datrow = []
            break
        else:
            if param == 'Plx':
                datrow.append(np.log10(row[param]))
            else:
                datrow.append(row[param])
    if datrow != []:
        cdata.append(datrow)

print(cdata)
cdata = np.array(cdata)

#corner.corner(cdata, show_titles=True, labels=params, fig=fig, color='red')
corner.corner(cdata, show_titles=True, labels=params, color='red')
plt.savefig('cutpopulation.png')
