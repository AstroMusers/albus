import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# --------------------- User-tweakable parameters ---------------------
input_csv = 'data_outputs/injected_transits_output4.csv'
n_bins_1d = 12           # number of bins for 1D binned averages
n_bins_2d = 30           # grid resolution for 2D heatmaps (per axis)
sigma = 1.0              # gaussian blur sigma for heatmaps (set 0 to disable smoothing)
count_threshold = 1      # minimum counts per 2D bin to consider valid (for masking)
# --------------------------------------------------------------------

# Load or create demo data
if os.path.exists(input_csv):
    df = pd.read_csv(input_csv)
    print(f"Loaded file: {input_csv}  (rows: {len(df)})")
else:
    print(f"File '{input_csv}' not found.")

# Normalize column names (strip whitespace)
df = df.rename(columns=lambda c: c.strip())

required_cols = ['r_p', 'real_period', 'inc', 'snr']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in the CSV: {missing}")

# Clean snr and drop invalid rows
df['snr'] = pd.to_numeric(df['snr'], errors='coerce')
before = len(df)
df = df.dropna(subset=['snr', 'r_p', 'real_period', 'inc'])
after = len(df)
print(f"Dropped {before-after} rows with invalid/missing snr or parameters. Remaining rows: {after}")

# extract
radius = df['r_p'].values
period = df['real_period'].values
inc = df['inc'].values
snr = df['snr'].values

# helper: binned summary
def binned_summary(x, y, n_bins=10):
    bins = np.linspace(np.nanmin(x), np.nanmax(x), n_bins+1)
    inds = np.digitize(x, bins) - 1
    centers = 0.5*(bins[:-1] + bins[1:])
    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, int)
    for i in range(n_bins):
        mask = inds==i
        counts[i] = np.count_nonzero(mask)
        if counts[i] > 0:
            means[i] = np.nanmean(y[mask])
    return centers, means, counts, bins

# helper: 2D grid
def compute_2d_grid(x, y, z, nx=20, ny=20):
    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), nx+1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), ny+1)
    x_inds = np.digitize(x, x_edges) - 1
    y_inds = np.digitize(y, y_edges) - 1
    grid = np.full((ny, nx), np.nan)
    counts = np.zeros((ny, nx), int)
    for xi in range(nx):
        for yi in range(ny):
            mask = (x_inds == xi) & (y_inds == yi)
            counts[yi, xi] = np.count_nonzero(mask)
            if counts[yi, xi] > 0:
                grid[yi, xi] = np.nanmean(z[mask])
    x_centers = 0.5*(x_edges[:-1] + x_edges[1:])
    y_centers = 0.5*(y_edges[:-1] + y_edges[1:])
    return x_edges, y_edges, x_centers, y_centers, grid, counts

# prepare grids
xp_edges, yp_edges, xp_centers, yp_centers, grid_rp, counts_rp = compute_2d_grid(radius, period, snr, nx=n_bins_2d, ny=n_bins_2d)
pp_edges, pi_edges, pp_centers, pi_centers, grid_pi, counts_pi = compute_2d_grid(period, inc, snr, nx=n_bins_2d, ny=n_bins_2d)
xr_edges, xi_edges, xr_centers, xi_centers, grid_ri, counts_ri = compute_2d_grid(radius, inc, snr, nx=n_bins_2d, ny=n_bins_2d)

# smoothing with Gaussian that respects missing bins: use mask-weighted smoothing
def smooth_grid(grid, counts, sigma=1.0):
    if sigma is None or sigma <= 0:
        # don't smooth, but ensure grid remains nan where counts==0
        out = grid.copy()
        out[counts < count_threshold] = np.nan
        return out
    mask = np.where(np.isfinite(grid), 1.0, 0.0)
    grid_filled = np.nan_to_num(grid, 0.0)
    num = gaussian_filter(grid_filled * mask, sigma=sigma, mode='nearest')
    den = gaussian_filter(mask, sigma=sigma, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        sm = num / den
    sm[den == 0] = np.nan
    sm[counts < count_threshold] = np.nan
    return sm

sm_rp = smooth_grid(grid_rp, counts_rp, sigma=sigma)
sm_pi = smooth_grid(grid_pi, counts_pi, sigma=sigma)
sm_ri = smooth_grid(grid_ri, counts_ri, sigma=sigma)

# Create single figure with 2 rows x 3 cols
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
plt.subplots_adjust(wspace=0.35, hspace=0.35)

# Top row: 1D plots (Radius, Period, Inclination)
# Radius
ax = axes[0,0]
ax.scatter(radius, snr, s=8, alpha=0.25)
centers, means, counts, _ = binned_summary(radius, snr, n_bins=n_bins_1d)
ax.plot(centers, means, marker='o', color='C1', linewidth=1.5)
ax.set_xlabel(r'Radius ($R_{\mathrm{J}}$)')
ax.set_ylabel('SNR')
ax.set_title('Radius vs SNR (scatter + binned avg)')
ax.grid(True, linestyle=':', linewidth=0.5)

# Period
ax = axes[0,1]
ax.scatter(period, snr, s=8, alpha=0.25)
centers, means, counts, _ = binned_summary(period, snr, n_bins=n_bins_1d)
ax.plot(centers, means, marker='o', color='C1', linewidth=1.5)
ax.set_xlabel('Period (days)')
ax.set_title('Period vs SNR (scatter + binned avg)')
ax.grid(True, linestyle=':', linewidth=0.5)

# Inclination
ax = axes[0,2]
ax.scatter(inc, snr, s=8, alpha=0.25)
centers, means, counts, _ = binned_summary(inc, snr, n_bins=n_bins_1d)
ax.plot(centers, means, marker='o', color='C1', linewidth=1.5)
ax.set_xlabel('Inclination (deg)')
ax.set_title('Inclination vs SNR (scatter + binned avg)')
ax.grid(True, linestyle=':', linewidth=0.5)

# Bottom row: heatmaps (Radius vs Period, Period vs Inclination, Radius vs Inclination)
# Heatmap 1: Radius vs Period (use xp_edges, yp_edges)
ax = axes[1,0]
mesh1 = ax.pcolormesh(xp_edges, yp_edges, sm_rp, shading='auto')
ax.set_xlabel(r'Radius ($R_{\mathrm{J}}$)')
ax.set_ylabel('Period (days)')
ax.set_title('Heatmap: Radius vs Period (avg SNR)')
cbar1 = fig.colorbar(mesh1, ax=ax, fraction=0.046, pad=0.04)
cbar1.set_label('Average SNR')

# Heatmap 2: Period vs Inclination
ax = axes[1,1]
mesh2 = ax.pcolormesh(pp_edges, pi_edges, sm_pi, shading='auto')
ax.set_xlabel('Period (days)')
ax.set_ylabel('Inclination (deg)')
ax.set_title('Heatmap: Period vs Inclination (avg SNR)')
cbar2 = fig.colorbar(mesh2, ax=ax, fraction=0.046, pad=0.04)
cbar2.set_label('Average SNR')

# Heatmap 3: Radius vs Inclination
ax = axes[1,2]
mesh3 = ax.pcolormesh(xr_edges, xi_edges, sm_ri, shading='auto')
ax.set_xlabel(r'Radius ($R_{\mathrm{J}}$)')
ax.set_ylabel('Inclination (deg)')
ax.set_title('Heatmap: Radius vs Inclination (avg SNR)')
cbar3 = fig.colorbar(mesh3, ax=ax, fraction=0.046, pad=0.04)
cbar3.set_label('Average SNR')

plt.suptitle('SNR vs Parameters — 1D and 2D summaries', fontsize=16, y=0.98)
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()

print('Done: single figure with 6 subplots generated. Adjust input_csv, n_bins_1d, n_bins_2d, sigma as needed.')
