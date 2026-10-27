from astropy.io import fits
import json
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from collections import Counter
from tdpy.verbosity import print, tqdm


# Open the FITS file
fits_path = Path('gaiaedr3_wd_main.fits')
print(f'Reading from {fits_path}...')
with fits.open(fits_path) as hdul:
    # Print the header of the primary HDU (first extension)
    # print(hdul[0].header)

    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    

    # Load the JSON file
    progress_path = Path('albus/albus/data_inputs/wd_progress.json')
    print(f'Reading from {progress_path}...')
    with progress_path.open('r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from JSON.")

    # Extract categories and Tmag values
    categories = []
    tmag_values = {}


    for key, value in data.items():
        category = value.get("category")
        tmag = value.get("Tmag")
        if category in ["bright_lc", "faint_lc"]:
            value["category"] = "lc_combined"
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
    output_path = Path('albus/albus/graphics/candidate_lc_histograms')
    print(f'Writing to {output_path}...')
    plt.savefig(output_path)

    # Plot histograms for "no_lc" and "lc_combined" categories
    categories_to_plot = ["no_lc", "lc_combined"]

    fig, axes = plt.subplots(nrows=len(categories_to_plot), figsize=(8, 4 * len(categories_to_plot)), sharex=True)

    for i, category in tqdm(enumerate(categories_to_plot), total=len(categories_to_plot)):
        if category in tmag_values:
            ax = axes[i] if len(categories_to_plot) > 1 else axes
            tmag_list = tmag_values[category]
            ax.hist(tmag_list, bins=20, color='orange', edgecolor='black')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Tmag Distribution for category {category}')

    # Set common x-axis label
    plt.xlabel('Tmag')
    plt.tight_layout()
    output_path = Path('albus/albus/graphics/tmag_histograms_no_lc_and_lc_combined.png')
    print(f'Writing to {output_path}...')
    plt.savefig(output_path)

    # Update categories and Tmag values after combining
    categories = []
    tmag_values = {}


    # Plot histogram for "lc_combined"
    if "lc_combined" in tmag_values:
        plt.figure(figsize=(8, 4))
        plt.hist(tmag_values["lc_combined"], bins=20, color='green', edgecolor='black')
        plt.xlabel('Tmag')
        plt.ylabel('Frequency')
        plt.title('Tmag Distribution for category lc_combined')
        plt.tight_layout()
        plt.show()

    # Plot histograms for Tmag values by category
    for category, tmag_list in tmag_values.items():
        plt.figure(figsize=(8, 4))
        plt.hist(tmag_list, bins=20, color='orange', edgecolor='black')
        plt.xlabel('Tmag')
        plt.ylabel('Frequency')
        plt.title(f'Tmag Distribution for category {category}')
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

    fig, ax = plt.subplots(len(categories), 2, figsize=(10, 6))
    for i, (category, coords) in enumerate(ra_dec_values.items()):
        ax[i, 0].scatter(coords["ra"], coords["dec"], label=category, alpha=0.6)
        ax[i, 0].set_xlabel('Right Ascension (degrees)')
        ax[i, 1].set_ylabel('Declination (degrees)')
        ax[i, 0].set_title(f'Scatter Plot of RA and Dec for category {category}')
    plt.legend()
    plt.tight_layout()
    output_path = Path('albus/albus/graphics/scatter_plot_ra_dec_by_category.png')
    print(f'Writing to {output_path}...')
    plt.savefig(output_path)

    def ra_to_lon_deg(ra_array):
        ra = np.asarray(ra_array, dtype=float)
        # If RA looks like hours (max ≤ 24), convert to degrees
        if np.nanmax(ra) <= 24.0:
            ra = ra * 15.0
        # Flip so RA increases to the left (astronomy convention), then wrap to [-180, 180)
        lon = (360.0 - ra) % 360.0
        lon = ((lon + 180.0) % 360.0) - 180.0
        return lon

    fig, ax = plt.subplots(len(categories), 1, figsize=(10, 8), subplot_kw={'projection': ccrs.Mollweide()})

    # Show the whole globe without specifying an extent CRS
    for a in ax:
        a.set_global()

    # Optional: nicer gridlines (labels often only supported for PlateCarree)
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, linestyle='--', alpha=0.5)

    for i, (category, coords) in enumerate(ra_dec_values.items()):
        ra = np.asarray(coords["ra"])
        dec = np.asarray(coords["dec"])
        lon = ra_to_lon_deg(ra)        # convert RA → lon
        lat = dec.astype(float)        # Dec is latitude in deg
        ax[i].scatter(lon, lat, s=10, alpha=0.6, label=category,
                transform=ccrs.PlateCarree())  # <-- data CRS!
        ax[i].legend()
        ax[i].set_title(f"Celestial Sphere (Mollweide) - {category}")
        plt.tight_layout()
    output_path = Path('albus/albus/graphics/mollweide_ra_dec_by_category.png')
    print(f'Writing to {output_path}...')
    plt.savefig(output_path, dpi=200)

    # To print information about all HDUs in the file
    # hdul.info()