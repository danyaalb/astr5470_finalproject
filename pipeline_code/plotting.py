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

import os

def plot_correction_diagnostics(v, binned_results, pixel_data):
    """
    v: The object containing raw data (v.wave_1d, v.flux_2d, etc.)
    binned_results: List of dicts from run_binned_analysis (using new compute_gp_correction keys)
    pixel_data: Dict from apply_spectrum_correction
    """
    # Create a 2x2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("GP Multiplicative Correction Diagnostics", fontsize=16, fontweight='bold')

    # --- 1. LIGHTCURVE COMPARISON (Top Left) ---
    # Shows Raw vs. Corrected (Systematics removed)
    ax = axes[0, 0]
    res = binned_results[0] # Plotting the first bin (usually the whole range)
    
    ax.errorbar(v.time, res['flux_raw'], yerr=res['eureka_err'], fmt='.', color='gray', alpha=0.3, label='Raw Binned')
    
    # Plot the smooth trend line
    ax.plot(res['t_smooth'], res['mu_slow_smooth'], 'r-', lw=2, label='Slow Trend (Systematics)')
    
    # Plot the CORRECTED points (multiplicatively cleaned)
    ax.scatter(v.time, res['flux_corr'], color='black', s=10, label='Corrected Lightcurve', zorder=3)
    
    ax.set_title("Lightcurve Level: Systematic Removal")
    ax.set_xlabel("Time (hrs)")
    ax.set_ylabel("Relative Flux")
    ax.legend()

    # --- 2. SPECTRUM SHIFT (Top Right) ---
    ax = axes[0, 1]
    raw_spec = np.nanmedian(v.flux_2d, axis=0) # Median over time (Jy)
    
    all_waves = []
    all_corr_flux = []
    k_factors = []
    
    # Sorting keys to ensure the line plot doesn't "criss-cross"
    sorted_labels = sorted(pixel_data.keys(), key=lambda x: float(x.split('-')[0]))
    
    for label in sorted_labels:
        data = pixel_data[label]
        all_waves.append(data['wavelengths'])
        all_corr_flux.append(np.nanmedian(data['fluxes_corr_jy'], axis=0))
        k_factors.append(data['err_factors'])

    ax.plot(v.wave_1d, raw_spec, color='red', alpha=0.5, label='Original Spectrum')
    ax.plot(np.concatenate(all_waves), np.concatenate(all_corr_flux), 'k--', label='Corrected Spectrum (Jy)')
    ax.set_title("Spectrum Level: Multiplicative Change")
    ax.set_xlabel("Wavelength ($\mu$m)")
    ax.set_ylabel("Flux (Jy)")
    ax.legend()

    # --- 3. ERROR MULTIPLIER (Bottom Left) ---
    ax = axes[1, 0]
    bin_centers = [np.mean(w) for w in all_waves]
    # Dynamically adjust width based on number of bins
    bar_width = (max(bin_centers) - min(bin_centers)) / len(bin_centers) if len(bin_centers) > 1 else 0.1
    
    ax.bar(bin_centers, k_factors, width=bar_width*0.8, color='teal', alpha=0.7)
    ax.axhline(1.0, color='red', linestyle='--')
    ax.set_title("Noise Scaling: K-Factors")
    ax.set_xlabel("Wavelength ($\mu$m)")
    ax.set_ylabel("Multiplier (k)")

    # --- 4. RESIDUAL SCATTER REDUCTION (Bottom Right) ---
    ax = axes[1, 1]
    # Raw scatter: relative data vs total trend
    std_raw = np.nanstd(res['flux_raw'] - res['mu_total']) 
    # Standard deviation of the corrected lightcurve (should be lower/whiter)
    std_corr = np.nanstd(res['flux_corr'] - 1.0) 
    
    ax.bar(['Raw (Model Resids)', 'Corrected (RMS)'], [std_raw, std_corr], color=['salmon', 'skyblue'])
    ax.set_ylabel("Fractional Std Dev")
    ax.set_title(f"Scatter Reduction: {((std_raw-std_corr)/std_raw)*100:.1f}%")
    
    plt.show()

def plot_multiplication_factors(binned_results):
    n_bins = len(binned_results)
    fig, axes = plt.subplots(n_bins, 1, figsize=(10, 4 * n_bins), sharex=True)
    
    if n_bins == 1: axes = [axes]

    for i, res in enumerate(binned_results):
        ax = axes[i]
        
        # Calculate the factor using the new keys: mu_slow / flux_raw
        # This represents the 'flattening' multiplier
        mult_factor = res['mu_slow'] / res['flux_raw']
        
        ax.plot(res['time'], mult_factor, 'p-', color='darkblue', label='Flux Scaling Factor')
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
        
        ax.set_title(f"Correction Factors: Bin {res['label']}")
        ax.set_ylabel("Scaling Factor")
        
        # Display the mean distance (error) relative to the trend
        avg_dist = np.mean(res['dist_error'])
        textstr = f"Mean Dist from Trend: {avg_dist:.5f}"
        ax.text(0.02, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axes[-1].set_xlabel("Time (BJD)")
    plt.tight_layout()
    plt.show()


def plot_results(time, flux, error, x_pred, mu_total, gp_std, mu_f, mu_s, residuals, samples, mcmc_results, mcmc_mu_samples, date, output_dir="./outputs"):
    """
    Main plotting function that saves high-res diagnostics to the output folder.
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
def show_corner_plot(samples, mcmc_results):
    corner.corner(samples, labels=["Amp1", "Gam1", "P_fast", "Amp2", "Gam2", "P_slow"], truths=mcmc_results[1])
    plt.show()


def plot_visit_diagnostic(v, bin_res):
    t = bin_res['time_hr']
    y = bin_res['y_raw']
    mu_s_fine = bin_res['mu_slow_fine']
    t_smooth = bin_res['t_smooth']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    
    # --- Top: Lightcurve View ---
    # Plotting raw data and the slow trend
    ax1.errorbar(t, y, yerr=np.abs(bin_res['residuals_sub']), fmt=".k", alpha=0.3, label="Raw Binned")
    ax1.plot(t_smooth, mu_s_fine, "r-", lw=2, label="Slow Trend (Systematics)")
    ax1.plot(t, bin_res['y_sub'], "ko", markersize=3, label="Corrected (Fast Removed)")
    
    ax1.set_title(f"GP Correction: {v.date}")
    ax1.set_ylabel("Relative Flux")
    ax1.legend()

    # --- Bottom: Multiplication Factor ---
    # This shows the actual per-integration shift
    mult_factor = bin_res['y_sub'] / bin_res['y_raw']
    ax2.plot(t, mult_factor, 'p-', color='blue', alpha=0.6)
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel("Multiplication Factor")
    ax2.set_xlabel("Time (BJD)")
    
    plt.tight_layout()
    plt.show()

def plot_visit_results(bin_res, mcmc_results, chains, date, mcmc_samples=None):
    """
    Diagnostic plot using dictionary keys from binned analysis.
    """
    # 1. Extract Data
    t = bin_res['time_hr']           # Raw data timestamps (e.g., 68 points)
    y = bin_res['y_raw']
    t_smooth = bin_res['t_smooth']   # Smooth grid (e.g., 1000 points)
    
    # Model components
    mu_total_fine = bin_res['mu_slow_fine'] + (bin_res['mu_fast_fine'] - np.mean(bin_res['mu_fast_fine']))
    mu_s_fine = bin_res['mu_slow_fine']
    mu_f_fine = bin_res['mu_fast_fine']
    # Distance-based error for raw data plotting
    err_dist = np.abs(bin_res['residuals_sub']) 
    
    # 2. Extract MCMC Parameters for labels
    p_f = np.exp(mcmc_results[1, 2]) * 60  # minutes
    p_s = np.exp(mcmc_results[1, 5])       # hours
    
    # 3. Stats
    residuals = bin_res['residuals_sub']
    dof = len(y) - len(mcmc_results[1])
    chi2_red = np.sum((residuals / bin_res['y_sub_err'])**2) / dof
    
    # 4. Setup Figure
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0:2, :]) # Main Plot
    ax2 = fig.add_subplot(gs[2, 0])   # Histogram
    ax3 = fig.add_subplot(gs[2, 1])   # Residuals vs Time
    
    # --- AX1: MAIN ANALYSIS ---
    # Raw Data
    ax1.errorbar(t, y, yerr=err_dist, fmt=".k", alpha=0.4, label="Data (Dist from Trend)")
    
    # Spaghetti lines (MCMC uncertainty)
    # FIX: Must plot against t_smooth to match the 1000-point dimension
    if mcmc_samples is not None:
        for m_sample in mcmc_samples[:40]: 
            if len(m_sample) == len(t_smooth):
                ax1.plot(t_smooth, m_sample, color="green", alpha=0.1, lw=0.5)
            elif len(m_sample) == len(t):
                ax1.plot(t, m_sample, color="green", alpha=0.1, lw=0.5)
    
    # Best Fit Lines
    ax1.plot(t_smooth, mu_total_fine, "green", lw=1.5, label="Full GP Model")
    ax1.plot(t_smooth, mu_s_fine, "blue", lw=2, ls="--", label=f"Slow Trend ({p_s:.2f}h)")
    ax1.plot(t_smooth, mu_f_fine, "orange", lw=2, ls="--", label=f"Fast Trend ({p_f:.2f}min)")
    
    # Cleaned data points
    ax1.scatter(t, bin_res['y_sub'], color='red', s=5, alpha=0.6, label="Corrected (y_sub)")

    ax1.set_title(f"Visit Analysis: {date} | $\chi^2_\\nu$: {chi2_red:.3f}")
    ax1.set_ylabel("Normalized Flux")
    ax1.legend(loc="lower right", ncol=2)

    # --- AX2: RESIDUAL HISTOGRAM ---
    std_res = residuals / bin_res['y_sub_err']
    ax2.hist(std_res, bins=20, density=True, color='skyblue', alpha=0.7)
    x_g = np.linspace(-4, 4, 100)
    ax2.plot(x_g, stats.norm.pdf(x_g, 0, 1), 'r--', label='Ideal')
    ax2.set_title("Standardized Residuals")

    # --- AX3: RESIDUALS VS TIME ---
    ax3.scatter(t, residuals, marker='.', color='black', alpha=0.4)
    ax3.axhline(0, color='red', linestyle='--')
    ax3.set_title("Residuals vs. Time")

    plt.tight_layout()
    plt.show()

    # --- SUMMARY PRINT & CORNER ---
    print(f"\n--- {date} SUMMARY ---")
    print(f"Fast Period: {p_f:.4f} min | Slow Period: {p_s:.4f} hours")
    print(f"Red Chi2: {chi2_red:.4f}")
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal
import os

def plot_binned_results(binned_results, visit_date, output_dir="./outputs"):
    """
    Diagnostic plot showing Raw vs GP-Corrected data, Scatter Reduction,
    and Periodogram Analysis.
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