#!/usr/bin/env python3
"""
Plot observed data, prior prediction, MAP prediction, and GN posterior
prediction ensembles.

Compares two GN posterior sampling approaches:
  1. CG precision solves with prior preconditioning
  2. Prior-preconditioned low-rank shrinkage

One subplot per rate type, one row per well plus total.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
GN_DIR = Path(__file__).resolve().parent
DATA_DIR = GN_DIR / 'DATA'
RESULTS_FILE = GN_DIR / 'laplace_results.npz'

# Load observed data
print("Loading data...")
data_df = pd.read_pickle(DATA_DIR / 'true_data.pkl')
var_df = pd.read_pickle(DATA_DIR / 'true_data_var.pkl')
n_t = len(data_df.index)

# Load results
print(f"Loading results from {RESULTS_FILE}")
results = np.load(RESULTS_FILE, allow_pickle=True)
d_obs = results['d_obs']
d_pred_map = results['d_pred_map']
d_pred_prior_samples = results['d_pred_prior_samples']  # (n_samples, n_d)
d_pred_post_samples = results['d_pred_post_samples']    # CG posterior
d_pred_post_gn = results['d_pred_post_gn']              # Low-rank GN posterior

# Reconstruct as DataFrames
n_cols = len(data_df.columns)
data_obs_df = pd.DataFrame(
    d_obs.reshape((n_t, n_cols), order='F'),
    index=data_df.index,
    columns=data_df.columns
)
pred_prior_df = pd.DataFrame(
    d_pred_prior_samples.mean(axis=0).reshape((n_t, n_cols), order='F'),
    index=data_df.index,
    columns=data_df.columns
)
pred_map_df = pd.DataFrame(
    d_pred_map.reshape((n_t, n_cols), order='F'),
    index=data_df.index,
    columns=data_df.columns
)

# Reshape ensemble samples: (n_samples, n_t, n_cols) in F-order
n_prior_samples = d_pred_prior_samples.shape[0]
n_post_samples = d_pred_post_samples.shape[0]
n_post_gn = d_pred_post_gn.shape[0]
prior_samples_3d = d_pred_prior_samples.reshape((n_prior_samples, n_t, n_cols), order='F')
post_samples_3d = d_pred_post_samples.reshape((n_post_samples, n_t, n_cols), order='F')
post_gn_3d = d_pred_post_gn.reshape((n_post_gn, n_t, n_cols), order='F')

# Build measurement std vector (same ordering as d_obs)
cd_diag_plot = np.zeros(n_t * n_cols)
idx = 0
for col in data_df.columns:
    for row in data_df.index:
        entry = var_df.loc[row, col]
        cd_diag_plot[idx] = entry[1] if isinstance(entry, list) else float(entry)
        idx += 1
cd_std_plot = np.sqrt(cd_diag_plot).reshape((n_t, n_cols), order='F')

print("Creating plots...")

# Create plots: rows are wells (plus total), columns are data types.
rate_types = ['WOPR', 'WGPR', 'WWPR']
rate_labels = ['Oil Production Rate (STB/D)', 'Gas Production Rate (MSCF/D)', 'Water Production Rate (STB/D)']
well_rows = ['P1', 'P2', 'P3', 'P4', 'TOTAL']
fig, axes = plt.subplots(len(well_rows), len(rate_types), figsize=(18, 16), sharex=True)

method_styles = [
    ('CG', post_samples_3d, '#2ca02c', 0.22),
    ('Low-rank GN', post_gn_3d, '#d62728', 0.22),
]

for row_idx, well in enumerate(well_rows):
    for col_idx, (rate_type, rate_label) in enumerate(zip(rate_types, rate_labels)):
        ax = axes[row_idx, col_idx]

        if well == 'TOTAL':
            type_cols = [c for c in data_df.columns if c.startswith(rate_type + ':')]
            col_indices = [data_df.columns.get_loc(c) for c in type_cols]
            obs_series = data_obs_df[type_cols].sum(axis=1)
            map_series = pred_map_df[type_cols].sum(axis=1)
            # Ensemble envelopes (sum across wells)
            prior_env = prior_samples_3d[:, :, col_indices].sum(axis=2)
            # Measurement std: sum of variances -> sqrt
            obs_std = np.sqrt((cd_std_plot[:, col_indices]**2).sum(axis=1))
            row_label = 'Total'
        else:
            col = f'{rate_type}:{well}'
            col_idx_data = data_df.columns.get_loc(col)
            obs_series = data_obs_df[col]
            map_series = pred_map_df[col]
            # Ensemble envelopes
            prior_env = prior_samples_3d[:, :, col_idx_data]
            obs_std = cd_std_plot[:, col_idx_data]
            row_label = well

        dates = data_obs_df.index

        # Prior ensemble envelope
        prior_lo = prior_env.min(axis=0)
        prior_hi = prior_env.max(axis=0)
        ax.fill_between(dates, prior_lo, prior_hi,
                        alpha=0.15, color='#ff7f0e', label='Prior ensemble')

        # Prior mean prediction
        prior_mean_env = prior_env.mean(axis=0)
        ax.plot(dates, prior_mean_env, '--',
                color='#ff7f0e', linewidth=1.5, alpha=0.9, label='Prior mean')

        # Posterior ensemble envelopes
        for method_name, post_3d, color, alpha in method_styles:
            if well == 'TOTAL':
                post_env = post_3d[:, :, col_indices].sum(axis=2)
            else:
                post_env = post_3d[:, :, col_idx_data]
            post_lo = post_env.min(axis=0)
            post_hi = post_env.max(axis=0)
            ax.fill_between(dates, post_lo, post_hi,
                            alpha=alpha, color=color,
                            label=f'{method_name} posterior')

        # MAP prediction
        ax.plot(dates, map_series, '-',
                label='MAP', color='k', linewidth=1.8, alpha=0.8)

        # Observed data as scatter with 2-sigma error bars
        ax.errorbar(dates, obs_series, yerr=2.0 * obs_std,
                    fmt='o', color='#1f77b4', markersize=4, capsize=3,
                    elinewidth=1.0, label='Data ± 2σ')

        if row_idx == 0:
            ax.set_title(rate_label, fontsize=12, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(row_label, fontsize=11, fontweight='bold')
        if row_idx == len(well_rows) - 1:
            ax.set_xlabel('Date', fontsize=10)

        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, linestyle='--')
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=9, loc='best', framealpha=0.95)

plt.suptitle('GN Laplace comparison: CG precision solves vs low-rank shrinkage',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(GN_DIR / 'predictions_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Build Cd diagonal using same convention as run_script.py
cd_diag = np.zeros(n_t * n_cols)
idx = 0
for col in data_df.columns:
    for row in data_df.index:
        entry = var_df.loc[row, col]
        cd_diag[idx] = entry[1] if isinstance(entry, list) else float(entry)
        idx += 1
cd_inv = 1.0 / cd_diag

# Print weighted data-misfit stats
print("\n" + "="*70)
print("PREDICTION STATISTICS")
print("="*70)

method_preds = {
    'Prior (mean)': d_pred_prior_samples.mean(axis=0),
    'MAP': d_pred_map,
    'CG (mean)': d_pred_post_samples.mean(axis=0),
    'Low-rank GN (mean)': d_pred_post_gn.mean(axis=0),
}

for rate_type, rate_label in zip(rate_types, rate_labels):
    cols = [c for c in data_df.columns if c.startswith(rate_type)]
    col_mask = np.zeros(n_cols, dtype=bool)
    for c in cols:
        col_mask[data_df.columns.get_loc(c)] = True

    flat_mask = np.repeat(col_mask, n_t)
    cd_inv_sub = cd_inv[flat_mask]
    obs_vals = d_obs[flat_mask]
    n_sub = int(flat_mask.sum())

    print(f"\n{rate_label} (n_data={n_sub}):")
    for mname, mpred in method_preds.items():
        r = mpred[flat_mask] - obs_vals
        misfit = float(r @ (cd_inv_sub * r))
        print(f"  {mname:20s}  misfit = {misfit:.6e}  (per-datum = {misfit/n_sub:.4e})")


# ============================================================
# Parameter samples figure
# ============================================================
print("Creating parameter samples figure...")

LOG_MEAN = float(np.log(500))
mask_arr = np.asarray(results['mask']).astype(bool)
grid_shape = mask_arr.shape
flat_mask = mask_arr.ravel(order='F')


def vec_to_grid(vec):
    """Map an active-cell vector to a 2-D array with NaN inactive cells."""
    grid = np.full(flat_mask.shape, np.nan)
    grid[flat_mask] = vec
    return grid.reshape(grid_shape, order='F').squeeze()


N_SHOW = min(
    10,
    results['prior_samples'].shape[0],
    results['post_samples_cg'].shape[0],
    results['post_samples_low_rank'].shape[0],
)

prior_vecs = LOG_MEAN + results['prior_samples'][:N_SHOW]
cg_vecs = results['post_samples_cg'][:N_SHOW]
gn_vecs = results['post_samples_low_rank'][:N_SHOW]

all_grids = np.stack(
    [vec_to_grid(v) for v in np.vstack([prior_vecs, cg_vecs, gn_vecs])]
)
finite = all_grids[np.isfinite(all_grids)]
vmin, vmax = np.percentile(finite, 2), np.percentile(finite, 98)

method_labels = ['Prior', 'CG', 'Low-rank GN']
method_colors = ['#ff7f0e', '#2ca02c', '#d62728']

fig2, axes2 = plt.subplots(6, 5, figsize=(14, 17))
fig2.subplots_adjust(hspace=0.05, wspace=0.05, right=0.88, top=0.94)

for panel_idx in range(3 * N_SHOW):
    method_idx = panel_idx // N_SHOW
    sample_idx = panel_idx % N_SHOW
    row = (method_idx * 2) + (sample_idx // 5)
    col = sample_idx % 5
    ax = axes2[row, col]

    ax.imshow(
        all_grids[panel_idx].T,
        origin='lower',
        aspect='equal',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    if col == 0 and sample_idx == 0:
        ax.set_ylabel(method_labels[method_idx], fontsize=11, fontweight='bold',
                      color=method_colors[method_idx], rotation=90, labelpad=4)
    if row == 0:
        ax.set_title(f's{sample_idx+1}', fontsize=8)

cbar_ax = fig2.add_axes([0.90, 0.05, 0.025, 0.87])
sm = plt.cm.ScalarMappable(cmap='viridis',
                            norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig2.colorbar(sm, cax=cbar_ax)
cbar.set_label('log-permeability (log mD)', fontsize=11)

fig2.suptitle('Parameter samples: Prior / CG / Low-rank GN',
              fontsize=13, fontweight='bold')

out_path2 = GN_DIR / 'parameter_samples.png'
fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"Saved plot to {out_path2}")
plt.close(fig2)


# ============================================================
# Empirical variance figure
# ============================================================
print("Creating empirical variance figure...")

all_vecs = {
    'Prior': results['prior_samples'] + LOG_MEAN,
    'CG': results['post_samples_cg'],
    'Low-rank GN': results['post_samples_low_rank'],
}
var_grids = {name: vec_to_grid(vecs.var(axis=0)) for name, vecs in all_vecs.items()}

all_var_finite = np.concatenate([
    grid[np.isfinite(grid)].ravel() for grid in var_grids.values()
])
vvar_min, vvar_max = 0.0, np.percentile(all_var_finite, 98)

fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.8))
fig3.subplots_adjust(wspace=0.08, right=0.87, top=0.88)

for idx, (name, color) in enumerate(zip(method_labels, method_colors)):
    ax = axes3[idx]
    ax.imshow(
        var_grids[name].T,
        origin='lower',
        aspect='equal',
        cmap='hot_r',
        vmin=vvar_min,
        vmax=vvar_max,
    )
    ax.set_title(name, fontsize=12, fontweight='bold', color=color)
    ax.set_xticks([])
    ax.set_yticks([])

cbar_ax3 = fig3.add_axes([0.89, 0.12, 0.025, 0.72])
sm3 = plt.cm.ScalarMappable(cmap='hot_r',
                             norm=plt.Normalize(vmin=vvar_min, vmax=vvar_max))
sm3.set_array([])
cbar3 = fig3.colorbar(sm3, cax=cbar_ax3)
cbar3.set_label('Empirical variance (log mD)^2', fontsize=11)

fig3.suptitle('Empirical posterior variance: Prior vs CG vs Low-rank GN',
              fontsize=12, fontweight='bold')

out_path3 = GN_DIR / 'variance_comparison.png'
fig3.savefig(out_path3, dpi=150, bbox_inches='tight')
print(f"Saved plot to {out_path3}")
plt.close(fig3)

print("\nDone!")
