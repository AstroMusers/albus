from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import pandas as pd
from astroquery.mast import Catalogs, Tesscut
from tqdm import tqdm
import csv

# Load the CSV file, assuming it has no headers
df = pd.read_csv('tess_targets.csv', names=['RA', 'DEC', 'Target ID'], on_bad_lines='skip', header = 0)

WDlist = fits.open('/Users/aavikwadivkar/Documents/Exoplanets/Research/gaiaedr3_wd_main.fits')
data = WDlist[1].data

cut = data

output_file = 'tess_targets_data.csv'

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(['Target ID', 'RA', 'DEC', 'Plx', 'Pwd', 'GmagCorr', 'GMAG', 'BP-RP', 
                     'TeffH', 'MassH', 'TeffHe', 'MassHe', 'Teffmix', 'Massmix'])
    # TESS ID, Parallax, Ra, Dec, , Probability, G-band mean magnitude (calibration corrected), Absolute G Magnitude, Bp-Rp color, 
    # Effective Temp (H-atmosphere), H Mass, E-temp (He-atmos), He mass, E-temp (H/He mix), H/He mass
    f.close()

tol = 0.00001

# Iterate over each row in the DataFrame
for index, row in tqdm(df.iterrows()):
    ra = row['RA']  # Access the RA column
    dec = row['DEC']  # Access the DEC column
    id = row['Target ID']

    for drow in data:
        if (abs(drow['ra'] - ra) < tol) and (abs(drow['dec'] - dec) < tol):
            print('found one!')
            break
        # else:
        #     print('womp womp')

    # with open(output_file, 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     # Write the data rows
    #     writer.writerow([ra, dec, id])
    #     f.close()

print(f"Results written to {output_file}")
