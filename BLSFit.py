"""Shared Miletos analysis with Albus-specific CSV row formats."""

import csv
from tdpy.verbosity import print
from miletos.box_least_squares import (
    fold_lightkurve as FoldedLC,
    lightkurve_bls_results as BLSResults,
)

def BLSfit(flatlc):
    """Return the existing BLS periodogram, propagating native fit failures."""
    from miletos.box_least_squares import fixed_grid_box_least_squares

    return fixed_grid_box_least_squares(flatlc)

def BLSOutput(ID, tic_id, high_periods, high_powers, output_file):
    print(f'Writing to {output_file}...')
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['ID', 'TIC ID', 'Periods', 'Powers'])
        writer.writerow([ID, tic_id, high_periods, high_powers])
        f.close()

def BLSTestOutputs(ID, tic_id, period, duration, depth, vshape, snr, oot_variability, output_file):
    print(f'Writing to {output_file}...')
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['ID', 'TIC ID', 'Period', 'Duration', 'Depth', 'V-Shape', 'SNR', 'OOT Variability'])
        writer.writerow([ID, tic_id, period, duration, depth, vshape, snr, oot_variability])
        f.close()
