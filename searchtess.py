import matplotlib.pyplot as plt
import pandas as pd
from astroquery.mast import Catalogs, Tesscut
import csv

with open('tess_targets.csv') as f:
    length = sum(1 for line in f)
    f.close()

# Load the CSV file, assuming it has no headers
df = pd.read_csv('reformatted_file.csv', header=length-1, names=['ra', 'dec'], on_bad_lines='skip')

output_file = 'tess_targets.csv'
with open(output_file, 'a', newline='') as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(['RA', 'DEC', 'Target ID'])
    f.close()

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    ra = row['ra']  # Access the RA column
    dec = row['dec']  # Access the DEC column
    
    # Create the target string for querying (right ascension and declination)
    target = f"{ra}, {dec}"
    print(f"Querying for target: {target}")
    
    # Query the TESS Input Catalog (TIC)
    target_info = Catalogs.query_object(f"{ra} {dec}", catalog="TIC")
    
    if len(target_info) > 0:
        target_id = target_info[0]['ID']
        print(f"Target ID: {target_id}")
    else:
        target_id = "No TESS Data"
        print("No TESS data found for this target.")
    
    # Append the RA, DEC, and Target ID to the output list
    # output_data.append([ra, dec, target_id])
    
    print([ra, dec, target_id])

    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # Write the data rows
        writer.writerow([ra, dec, target_id])
        f.close()

print(f"Results written to {output_file}")

# Create a scatter plot of the RA and DEC values
# df.plot(kind='scatter', x='ra', y='dec')
# plt.xlabel('Right Ascension (RA)')
# plt.ylabel('Declination (DEC)')
# plt.title('Scatter Plot of RA and DEC')
# plt.show()
