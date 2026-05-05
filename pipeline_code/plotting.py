import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import corner
import corner.corner
from scipy.ndimage import generic_filter
import numpy as np
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal
import os
import os
from .kernels import get_kernel
from .modeling import run_mcmc, get_best_params
from .file_formatting import load_h5_data, save_corr_data

def plot_results(time, flux, error, x_pred, mu_total, gp_std, mu_f, mu_s, residuals, samples, mcmc_results, mcmc_mu_samples, date, output_dir="./outputs"):
    """
    Main plotting function that saves high-res diagnostics to the output folder.

    :param time: 1D array of time points
    :type time: np.ndarray
    :param flux: 1D array of flux values (normalized)
    :type flux: np.ndarray
    :param error: 1D array of flux uncertainties (normalized)
    :type error: np.ndarray
    :param x_pred: 1D array of time points for GP predictions (same as time or a smooth grid)
    :type x_pred: np.ndarray
    :param mu_total: 1D array of total GP prediction at x_pred (median parameters)
    :type mu_total: np.ndarray
    :param gp_std: 1D array of GP standard deviation at x_pred (median parameters)
    :type gp_std: np.ndarray
    :param mu_f: 1D array of the "fast" component of the GP at x_pred
    :type mu_f: np.ndarray
    :param mu_s: 1D array of the "slow" component of the GP at x_pred
    :type mu_s: np.ndarray
    :param residuals: 1D array of (flux - GP prediction) at the original time points
    :type residuals: np.ndarray
    :param samples: 2D array of MCMC samples (n_samples x n_params)
    :type samples: np.ndarray
    :param mcmc_results: 2D array of percentiles (16th, 50th, 84th) for each parameter
    :type mcmc_results: np.ndarray
    :param mcmc_mu_samples: List of 1D arrays, each being GP prediction at x_pred for a random MCMC sample
    :type mcmc_mu_samples: list of np.ndarray
    :param date: Visit date for labeling the plot
    :type date: str
    :param output_dir: Directory to save the plots (default: "./outputs")
    :type output_dir: str
    :return: None (saves diagnostic plots to disk)
    :rtype: None
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Top Panel: Data + GP Model ---
    ax1.errorbar(time, flux, yerr=error, fmt='.', color='gray', alpha=0.3, label='Raw Data')
    
    # Plot the "Spaghetti" (MCMC Uncertainty)
    for s_mu in mcmc_mu_samples[:50]: # Cap at 50 for speed
        ax1.plot(x_pred, s_mu, color='green', alpha=0.05, lw=0.5)
        
    ax1.plot(x_pred, mu_total, color='black', lw=2, label='Total GP Model')
    ax1.plot(x_pred, mu_s, 'b--', alpha=0.8, label='Slow Trend (Systematics)')
    ax1.plot(x_pred, mu_f, 'r:', alpha=0.8, label='Fast Jitter')
    
    ax1.set_ylabel("Relative Flux")
    ax1.legend(loc='best', ncol=2)
    # Using 'r' for raw string to avoid SyntaxWarnings with \chi
    ax1.set_title(rf"PSO318 Jitter Correction - Visit: {date}")

    # --- Bottom Panel: Residuals ---
    ax2.scatter(time, residuals, color='purple', s=8, alpha=0.5)
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_ylabel("Residuals")
    ax2.set_xlabel("Time (Hours)")

    plt.tight_layout()

    # --- SAVE LOGIC ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, f"Diagnostic_{date}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"Diagnostic_{date}.pdf"))
    
    print(f"✅ Plots saved to: {output_dir}/Diagnostic_{date}.png and .pdf")
    plt.show() # Shows it on screen while running
    plt.close(fig)

def plot_binned_results(binned_results, visit_date, output_dir="./outputs"):
    """
    Diagnostic plot showing Raw vs GP-Corrected data, Scatter Reduction, and Periodogram Analysis.

    :param binned_results: List of dicts, each containing GP correction results for a specific wavelength bin (output of run_binned_analysis)
    :type binned_results: list of dict
    :param visit_date: Visit date for labeling the plot
    :type visit_date: str
    :param output_dir: Directory to save the plots (default: "./outputs")
    :type output_dir: str
    :return: None (saves diagnostic plots to disk for each wavelength bin)
    :rtype: None
    """
    for res in binned_results:
        # Extract data from your compute_gp_correction dictionary
        t = res['time_hr']
        t_smooth = res['t_smooth']
        y_raw = res['flux_raw']
        y_corr = res['flux_corr']
        mu_slow_fine = res['mu_slow_smooth']
        mu_fast_fine = res['mu_fast_smooth']
        
        # Calculate Scatter (RMS)
        # We compare raw data against the slow trend vs corrected data against 1.0
        res_raw = y_raw - res['mu_slow'] 
        res_corr = y_corr - res['mu_slow'] 
        
        rms_raw = np.nanstd(res_raw)
        rms_corr = np.nanstd(res_corr)
        improvement = (1 - (rms_corr / rms_raw)) * 100

        print(f"\n--- Statistics for Bin {res['label']} ---")
        print(f"Old Scatter (RMS): {rms_raw:.6f}")
        print(f"New Scatter (RMS): {rms_corr:.6f}")
        print(f"Scatter Reduction: {improvement:.2f}%")

        # --- Create Figure ---
        fig = plt.figure(figsize=(8, 10))
        gs = plt.GridSpec(3, 1, height_ratios=[2, 0.6, 1.2])

        # 1. LIGHT CURVE COMPARISON
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t, y_raw, 'o', color='gray', alpha=0.2, label="Old Light Curve (Raw)")
        ax1.plot(t, y_corr, 'o', color='red', markersize=4, label="Corrected Points")
        ax1.plot(t_smooth, mu_slow_fine, color='blue', lw=2, label="GP Slow Trend")
        
        ax1.set_title(f"{visit_date} | Bin: {res['label']} | Improv: {improvement:.1f}%", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Normalized Flux")
        ax1.legend(loc='lower right', fontsize=8, ncol=2)

        # 2. RESIDUALS & HISTOGRAM
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.scatter(t, res_raw, color='gray', s=15, alpha=0.3, label="Old Resids")
        ax2.scatter(t, res_corr, color='red', s=20, marker='+', alpha=0.8, label="New Resids")
        ax2.axhline(0, color='black', ls='--', alpha=0.5)
        ax2.set_ylabel("Residuals")
        
        ax_h = inset_axes(ax2, width="20%", height="100%", loc='lower right', bbox_to_anchor=(0.05, 0, 1, 1), bbox_transform=ax2.transAxes)
        ax_h.hist(res_raw, bins=15, orientation='horizontal', color='gray', alpha=0.2, density=True)
        ax_h.hist(res_corr, bins=15, orientation='horizontal', color='red', alpha=0.4, density=True)
        ax_h.set_axis_off()

        # 3. LOMB-SCARGLE (The Period Proof)
        ax3 = fig.add_subplot(gs[2])
        periods_min = np.linspace(2, 150, 2000)
        ang_freqs = 2 * np.pi / (periods_min / 60.0)
        
        p_raw = signal.lombscargle(np.ascontiguousarray(t), np.ascontiguousarray(res_raw), ang_freqs)
        p_corr = signal.lombscargle(np.ascontiguousarray(t), np.ascontiguousarray(res_corr), ang_freqs)
        
        ax3.plot(periods_min, p_raw, color='gray', alpha=0.4, label="Raw Power")
        ax3.plot(periods_min, p_corr, color='red', lw=1.5, label="Corrected Power")
        ax3.axvline(13.0, color='blue', ls=':', alpha=0.6, label="13m Jitter")
        
        ax3.set_xlabel("Period [minutes]")
        ax3.set_ylabel("Power")
        ax3.set_xlim(2, 150)
        ax3.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        
        # SAVE
        save_base = f"Binned_Summary_{visit_date}_{res['label'].replace('.', 'p')}"
        plt.savefig(os.path.join(output_dir, f"{save_base}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{save_base}.pdf"))
        plt.close(fig)