from astropy.io import fits
import json
from collections import Counter


# Open the FITS file
with fits.open('/Users/aavikwadivkar/Documents/Exoplanets/Ampersand/gaiaedr3_wd_main.fits') as hdul:
    # Print the header of the primary HDU (first extension)
    # print(hdul[0].header)

    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    

    # Load the JSON file
    with open('data_inputs/wd_progress.json', 'r') as f:
        data = json.load(f)

    # Extract categories and Tmag values
    categories = []
    tmag_values = {}

    for key, value in data.items():
        category = value.get("category")
        tmag = value.get("Tmag")
        if category:
            categories.append(category)
            if category not in tmag_values:
                tmag_values[category] = []
            if tmag is not None:
                tmag_values[category].append(tmag)

    # Count occurrences of each category
    category_counts = Counter(categories)

    # Plot the category occurrences
    plt.figure(figsize=(10, 5))
    plt.bar(category_counts.keys(), category_counts.values(), color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.title('Category Occurrences')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Combine "bright_lc" and "faint_lc" into a single category "lc_combined"
    for key, value in data.items():
        category = value.get("category")
        if category in ["bright_lc", "faint_lc"]:
            value["category"] = "lc_combined"

    # Update categories and Tmag values after combining
    categories = []
    tmag_values = {}

    # Plot histogram for "lc_combined"
    if "lc_combined" in tmag_values:
        plt.figure(figsize=(8, 4))
        plt.hist(tmag_values["lc_combined"], bins=20, color='green', edgecolor='black')
        plt.xlabel('Tmag')
        plt.ylabel('Frequency')
        plt.title('Tmag Distribution for Category: lc_combined')
        plt.tight_layout()
        plt.show()

    # Plot histograms for Tmag values by category
    for category, tmag_list in tmag_values.items():
        plt.figure(figsize=(8, 4))
        plt.hist(tmag_list, bins=20, color='orange', edgecolor='black')
        plt.xlabel('Tmag')
        plt.ylabel('Frequency')
        plt.title(f'Tmag Distribution for Category: {category}')
        plt.tight_layout()
        plt.show()
    print(hdul[1].header)

    ra_dec_values = {}

    for key, value in data.items():
        category = value.get("category")
        tmag = value.get("Tmag")
        if category:
            categories.append(category)
            if category not in tmag_values:
                tmag_values[category] = []
            if tmag is not None:
                tmag_values[category].append(tmag)

            # Extract RA and Dec from the key
            ra, dec = map(float, key.split(","))
            if category not in ra_dec_values:
                ra_dec_values[category] = {"ra": [], "dec": []}
            ra_dec_values[category]["ra"].append(ra)
            ra_dec_values[category]["dec"].append(dec)

    # Scatter plot of RA and Dec labeled by category
    plt.figure(figsize=(10, 6))
    for category, coords in ra_dec_values.items():
        plt.scatter(coords["ra"], coords["dec"], label=category, alpha=0.6)
    plt.xlabel('Right Ascension (degrees)')
    plt.ylabel('Declination (degrees)')
    plt.title('Scatter Plot of RA and Dec by Category')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # To print information about all HDUs in the file
    # hdul.info()