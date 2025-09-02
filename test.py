import random
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from preprocess import preprocess
from injections import inject_transit, generate_lightcurve
from BLSFit import BLSfit, BLSResults, FoldedLC, BLSTestOutputs
from BLStests import test_period, test_duration, test_depth, test_v_shape, test_snr, test_out_of_transit_variability, create_transit_mask_manual
from injections import calc_a

# Load the CSV file into a DataFrame
df = pd.read_csv('data_outputs/injected_transits_output3.csv')

# Calculate the ratio between 'real_period' and 'period'
df['period_ratio'] = df[' real_period'] / df[' period']

# Plot the histogram
plt.hist(df['period_ratio'], bins=50, color='blue', alpha=0.7, log=True)
plt.xlabel('Ratio (real_period / period)')
plt.ylabel('Frequency')
plt.title('Histogram of Period Ratios')
plt.grid(True)
plt.show()