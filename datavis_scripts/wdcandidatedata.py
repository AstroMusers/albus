from astropy.io import fits
import json
from collections import Counter


# Open the FITS file
with fits.open('gaiaedr3_wd_main.fits') as hdul:
    # Print the header of the primary HDU (first extension)
    # print(hdul[0].header)

    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    

    # Load the JSON file
    with open('albus/albus/data_inputs/wd_progress.json', 'r') as f:
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
    plt.savefig('albus/albus/graphics/candidate_lc_histograms')

    # Combine "bright_lc" and "faint_lc" into a single category "lc_combined"
    for key, value in data.items():
        category = value.get("category")
        if category in ["bright_lc", "faint_lc"]:
            value["category"] = "lc_combined"

    # Plot histograms for "no_lc" and "lc_combined" categories
    categories_to_plot = ["no_lc", "lc_combined"]

    fig, axes = plt.subplots(nrows=len(categories_to_plot), figsize=(8, 4 * len(categories_to_plot)), sharex=True)

    for i, category in enumerate(categories_to_plot):
        if category in tmag_values:
            ax = axes[i] if len(categories_to_plot) > 1 else axes
            tmag_list = tmag_values[category]
            ax.hist(tmag_list, bins=20, color='orange', edgecolor='black')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Tmag Distribution for Category: {category}')

    # Set common x-axis label
    plt.xlabel('Tmag')
    plt.tight_layout()
    plt.savefig('albus/albus/graphics/tmag_histograms_no_lc_and_lc_combined.png')

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
    # print(hdul[1].header)

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
            if ',' in key:
                ra, dec = map(float, key.split(","))
                if category not in ra_dec_values:
                    ra_dec_values[category] = {"ra": [], "dec": []}
                ra_dec_values[category]["ra"].append(ra)
                ra_dec_values[category]["dec"].append(dec)
            else:
                # Handle source_id case if needed
                pass

                

    # Scatter plot of RA and Dec labeled by category
    # plt.figure(figsize=(10, 6))
    # for category, coords in ra_dec_values.items():
    #     plt.scatter(coords["ra"], coords["dec"], label=category, alpha=0.6)
    # plt.xlabel('Right Ascension (degrees)')
    # plt.ylabel('Declination (degrees)')
    # plt.title('Scatter Plot of RA and Dec by Category')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('albus/albus/graphics/scatter_plot_ra_dec_by_category.png')

    # To print information about all HDUs in the file
    # hdul.info()