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

    
# --- UPDATED 4. PLOTTING (COMBINED VIEW) ---
def plot_results(t, y, err, x_pred, mu, gp_std, mu_f, mu_s, residuals, samples, mcmc_results, mcmc_samples, date):
    y_mean = np.mean(y)
    p_f = np.exp(mcmc_results[1, 2]) * 60
    p_s = np.exp(mcmc_results[1, 5])
    
    # Stats: Reduced Chi-Squared
    dof = len(y) - len(mcmc_results[1])
    chi2_red = np.sum((residuals / err)**2) / dof
    
    fig = plt.figure(figsize=(10, 8))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # --- COMBINED MAIN PLOT ---
    ax1 = fig.add_subplot(gs[0:1, :]) # Takes up top two rows
    
    # 1. Raw Data
    ax1.errorbar(t, y, yerr=err, fmt=".k", alpha=1.0, label="Data")
    
    # 2. MCMC Parameter Uncertainty (The "Spaghetti" lines)
    for m_sample in mcmc_samples:
        ax1.plot(x_pred, m_sample, color="green", alpha=0.2, lw=0.5)
    
    # 3. GP Uncertainty (The Shaded Region)
    ax1.fill_between(x_pred, mu - gp_std, mu + gp_std, color="black", alpha=0.15, label="GP Uncertainty")
    
    # 4. Total Best Fit
    ax1.plot(x_pred, mu, "green", lw=1, label="Full GP Model (Median)")
    
    # 5. Decomposed Components (Overlayed)
    # Centered at mean for visualization
    ax1.plot(x_pred, mu_f-1 + y_mean, "red", lw=1, alpha=0.8, label=f"Fast Pulsation ({p_f:.2f}m)")
    ax1.plot(x_pred, mu_s-1 + y_mean, "blue", lw=1.5, ls="--", alpha=0.8, label=f"Slow Trend ({p_s:.2f}h)")
    
    ax1.set_title(f"Combined GP Analysis: {date} | $\chi^2_\\nu$: {chi2_red:.3f}")
    ax1.set_ylabel("Normalized Flux")
    ax1.legend(loc="lower right", ncol=2)

    # --- RESIDUAL PLOTS ---
    ax2 = fig.add_subplot(gs[1, 0])
    std_res = residuals / err
    ax2.hist(std_res, bins=25, density=True, color='skyblue', alpha=0.7)
    x_g = np.linspace(-4, 4, 100)
    ax2.plot(x_g, stats.norm.pdf(x_g, 0, 1), 'r--', label='Ideal CLT')
    ax2.set_title("Standardized Residuals (CLT Check)")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(t, residuals, marker='.', color='black', alpha=0.3)
    ax3.axhline(0, color='red', linestyle='--')
    ax3.set_title("Residuals vs. Time")

    plt.tight_layout()
    plt.show()

    # E. Table and Corner
    print(f"\n--- {date} FINAL SUMMARY ---")
    print(f"Fast Period: {p_f:.4f} min")
    print(f"Slow Period: {p_s:.4f} hours")
    print(f"Chi-Squared Red: {chi2_red:.4f} {'(Taller histogram = Overestimated Errors)' if chi2_red < 1 else ''}")
    
    corner.corner(samples, labels=["Amp1", "Gam1", "P_fast", "Amp2", "Gam2", "P_slow"], truths=mcmc_results[1])
    plt.show()

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
# def plot_visit_results(bin_res, mcmc_results, chains, date, mcmc_samples=None):
#     # 1. Extract necessary data from bin_res dictionary
#     t = bin_res['time_hr']
#     y = bin_res['y_raw']
#     t_smooth = bin_res['t_smooth']
#     mu_s_fine = bin_res['mu_slow_fine']
    
#     # 2. Setup Figure and Axes first to avoid UnboundLocalError
#     fig = plt.figure(figsize=(12, 10))
#     gs = plt.GridSpec(3, 2, figure=fig)
    
#     # Explicitly define axes
#     ax1 = fig.add_subplot(gs[0:2, :]) 
#     ax2 = fig.add_subplot(gs[2, 0])
#     ax3 = fig.add_subplot(gs[2, 1])

#     # --- TOP PLOT (ax1) ---
#     # Use the absolute residuals as the "distance-based" error bar
#     residuals = bin_res['residuals_sub']
#     ax1.errorbar(t, y, yerr=np.abs(residuals), fmt=".k", alpha=0.4, label="Data")
    
#     # Spaghetti lines: Must use t_smooth to match the 1000-point dimension
#     if mcmc_samples is not None:
#         for m_sample in mcmc_samples:
#             if len(m_sample) == len(t_smooth):
#                 ax1.plot(t_smooth, m_sample, color="green", alpha=0.1, lw=0.5)

#     ax1.plot(t_smooth, mu_s_fine, "blue", lw=2, ls="--", label="Slow Trend")
#     ax1.set_title(f"Visit Analysis: {date}")
#     ax1.legend()

#     # --- RESIDUAL PLOTS (ax2, ax3) ---
#     std_res = residuals / bin_res['y_sub_err']
#     ax2.hist(std_res, bins=20, density=True, color='skyblue', alpha=0.7)
#     ax2.set_title("Standardized Residuals")

#     ax3.scatter(t, residuals, marker='.', color='black', alpha=0.4)
#     ax3.axhline(0, color='red', linestyle='--')
#     ax3.set_title("Residuals vs. Time")

#     plt.tight_layout()
#     plt.show()