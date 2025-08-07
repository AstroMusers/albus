from astropy.io import fits
from astropy.table import Table
#import matplotlib.pyplot as plt
from astroquery.mast import Catalogs, Tesscut
from tqdm import tqdm
import csv

WDlist = fits.open('/Users/aavikwadivkar/Documents/Exoplanets/Research/gaiaedr3_wd_main.fits')
data = WDlist[1].data

print(data.names)

# # Load tess_targets_data.csv
# csv_path = '/Users/aavikwadivkar/Documents/Exoplanets/Ampersand/albus/albus/tess_targets_data.csv'
# with open(csv_path, newline='') as csvfile:
#     reader = list(csv.DictReader(csvfile))
#     fieldnames = reader[0].keys()

# print("Loaded CSV with {} rows and {} columns.".format(len(reader), len(fieldnames)))

# # Build a lookup for (RA, DEC) -> emass_H from the FITS data
# fits_lookup = {}
# for row in tqdm(data):
#     ra = round(float(row['RA']), 6)
#     dec = round(float(row['DEC']), 6)
#     emass_h = row['emass_H'] if 'emass_H' in row.array.names else None
#     fits_lookup[(ra, dec)] = emass_h
# print("Built lookup for {} (RA, DEC) pairs.".format(len(fits_lookup)))

# # Prepare to write updated CSV with new column
# new_fieldnames = list(fieldnames) + ['E_MassH']
# updated_rows = []
# for row in tqdm(reader):
#     ra = round(float(row['RA']), 6)
#     dec = round(float(row['DEC']), 6)
#     emass_h = fits_lookup.get((ra, dec), '')
#     row['E_MassH'] = emass_h
#     updated_rows.append(row)
# print("Prepared {} updated rows.".format(len(updated_rows)))

# # Write updated CSV
# with open(csv_path, 'w', newline='') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=new_fieldnames)
#     writer.writeheader()
#     writer.writerows(updated_rows)