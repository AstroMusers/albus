import csv
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

cdf = pd.read_csv('candidates3.csv', on_bad_lines='skip', header = 0)
df = pd.read_csv('tess_targets.csv', on_bad_lines='skip', header = 0)
cands = pd.read_csv('candidatesdata.csv', on_bad_lines='skip', header = 0)

WDlist = fits.open('/Users/aavikwadivkar/Documents/Exoplanets/Research/gaiaedr3_wd_main.fits')
FITSdata = WDlist[1].data

output_file = 'candidatesdata.csv'

# with open(output_file, 'a', newline='') as f:
#     writer = csv.writer(f)
#     # Write the header
#     writer.writerow(['Target ID', 'Highest Period', 'Highest Power', 
#                      'Second Highest Period', 'Second Highest Power', 
#                      'G-band Luminosity', 'Bp - Rp', 'RA', 'DEC'])
#     f.close()

# period1, period2, powers = [], [], []

tol = 0.00001

for cindex, crow in tqdm(cdf.iterrows()):
    here = False
    for caindex, cairow in cands.iterrows():
        if int(crow['Target ID']) == int(cairow['Target ID']):
            # print('here')
            here = True
            break
    if here == False:
        id = crow['Target ID']
        period1 = crow['Highest Period']
        power1 = crow['Highest Power']
        period2 = crow['Second Highest Period']
        power2 = crow['Second Highest Power']
        
        for index, row in df.iterrows():
            if int(row['Target ID']) == int(id):
                ra = row['RA']
                dec = row['DEC']
                break
        for drow in FITSdata:
            # print(drow['ra'], drow['dec'])
            if (abs(drow['ra'] - ra) < tol) and (abs(drow['dec'] - dec) < tol):
                lum = drow['absG']
                bp_rp = drow['bp_rp']
                
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    # Write the header
                    writer.writerow([id, period1, power1, 
                                    period2, power2,
                                    lum, bp_rp, ra, dec])
                    f.close()
                break
