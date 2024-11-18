from astropy.io import fits
from astropy.table import Table
#import matplotlib.pyplot as plt
from astroquery.mast import Catalogs, Tesscut
from tqdm import tqdm
import csv

WDlist = fits.open('/Users/aavikwadivkar/Documents/Exoplanets/Research/gaiaedr3_wd_main.fits')
data = WDlist[1].data

cut = data[(data['Pwd'] > 0.9) & (data['bright_N_flag'] == 0) & (data['phot_g_mean_mag_corrected'] < 16)]
# Probility, Bright neighbor flag, photometric cutoff
# chose to save time (for now)

# To calculate abs. magnitude: M = m + 5log(parallax/100)

print(len(cut))

# Check in TESS data is available
# valids = []
# with open('TESS_WD.csv', 'w') as csvfile:     
#     csvwriter = csv.writer(csvfile)    
#     csvwriter.writerow(['RA', 'DEC'])

cut = cut[1404:]

for cand in tqdm(cut):
	target = f'{cand['ra']}, {cand['dec']}'
	target_info = Catalogs.query_object(target, catalog="TIC")

	if len(target_info) > 0:
		target_id = target_info[0]['ID']
		print(f"Target ID: {target_id}")
		
		# Use the TIC ID to check if TESS data exists
		tesscut_data = Tesscut.get_cutouts(coordinates=target)  # 5-arcmin cutout search
		
		if tesscut_data:
			print(f"TESS data available for {target}.")
			# print(tesscut_data[0][1].data)
			# valids.append(target)
			with open('TESS_WD.csv', 'a') as csvfile:   
				csvwriter = csv.writer(csvfile)   
				# writing the data rows   
				csvwriter.writerows([target])

		else:
			print(f"No TESS data available for {target}.")
	else:
		print(f"Target {target} not found in the TIC catalog.")

#print(len(valids))

 
