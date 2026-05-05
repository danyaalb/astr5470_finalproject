import numpy as np
from kernels import get_kernel
from modeling import run_mcmc, get_best_params
from file_formatting import load_h5_data, save_corr_data
import h5py
import george
from george import kernels
from scipy.optimize import minimize
from scipy.ndimage import generic_filter

def compute_gp_correction(time, flux, err, gp, mcmc_results, k_fast, k_slow):
    # Update GP to best-fit
    gp.set_parameter_vector(mcmc_results[1])

    # 1000 point grid for smooth plotting
    t_smooth = np.linspace(time.min(), time.max(), 1000)

    # Predictions at data points
    mu_fast = gp.predict(flux, time[:, None], kernel=k_fast, return_cov=False)
    mu_slow = gp.predict(flux, time[:, None], kernel=k_slow, return_cov=False)
    mu_total = mu_fast + mu_slow  # <--- This is the key that was missing!

    # Predictions on smooth grid for the plotter
    mu_total_fine = gp.predict(flux, t_smooth[:, None], return_cov=False)
    mu_fast_fine = gp.predict(flux, t_smooth[:, None], kernel=k_fast, return_cov=False)
    # Reconstruct slow trend for plotting
    mu_slow_fine = (mu_total_fine - mu_fast_fine) - np.mean(mu_total_fine - mu_fast_fine) + np.mean(flux)

    # Calculate scatter (for k-factor scaling)
    # residuals = (Normalized Data) - (GP Model)
    residuals = flux - mu_total
    # empirical_scatter = np.nanstd(residuals)
    y_corrected = flux/(1+(mu_fast - np.mean(mu_fast)))
    correction_factor =  y_corrected/flux 

    # 2. Calculate local standard deviation using a sliding window
    # We use a window of points (e.g., 5 or 7) to estimate local noise
    def local_std(x):
        return np.nanstd(x)
    
    gp_err = generic_filter(residuals, local_std, size=5)
    return {
        "time_hr": time,
        "t_smooth": t_smooth,
        "mu_fast": mu_fast,
        "mu_slow": mu_slow,
        "mu_total": mu_total,         
        "mu_fast_smooth": mu_fast_fine,
        "mu_slow_smooth": mu_slow_fine,
        "flux_factor": correction_factor,
        "flux_raw": flux,
        "eureka_err": err,
        "gp_err": gp_err,
        'flux_corr': y_corrected
    }


def run_binned_analysis(
    time,
    flux_2d,
    err_2d,
    wavelengths,
    gp,
    mcmc_results,
    k_fast,
    k_slow,
    wav_ranges
):
    """
    Compute GP correction factors for each wavelength bin.
    """

    if flux_2d.shape[0] == len(time):
        flux_2d = flux_2d.T
        err_2d = err_2d.T

    results = []

    for w_start, w_end in wav_ranges:

        mask = (wavelengths >= w_start) & (wavelengths < w_end)

        if not np.any(mask):
            continue

        # Bin flux
        bin_flux = np.nansum(flux_2d[mask, :], axis=0)
        bin_err = np.sqrt(np.nansum(err_2d[mask, :]**2, axis=0))

        # Normalize
        norm = np.nanmedian(bin_flux)
        bin_flux /= norm
        bin_err /= norm

        gp_res = compute_gp_correction(
            time,
            bin_flux,
            bin_err,
            gp,
            mcmc_results,
            k_fast,
            k_slow
        )

        gp_res["label"] = f"{w_start:.3f}-{w_end:.3f}"

        results.append(gp_res)

    return results
import h5py


def get_corrected_pixel_data(h5_path, binned_results):
    pixel_results = {}

    with h5py.File(h5_path, "r") as hf:
        waves = hf["wave_1d"][:]
        # These are in physical units (Jy)
        flux_2d = hf["calibrated_optspec"][:]
        err_2d = hf["calibrated_opterr"][:]

        for bin_res in binned_results:
            w_start, w_end = map(float, bin_res["label"].split("-"))
            mask = (waves >= w_start) & (waves < w_end)
            if not np.any(mask): continue

            # --- 1. MULTIPLICATIVE CORRECTION ---
            # correction is dimensionless (centered around 1.0)
            # flux_2d is Jy. Result is Jy.
            # Inside the bin loop in get_corrected_pixel_data
            correction_1d = bin_res["flux_factor"]

            # Apply to all pixels in the bin (Broadcasting)
            corrected_flux_2d = flux_2d[:, mask] * correction_1d[:, None]

            # corrected_flux = flux_2d[:, mask] / correction[:, None]

            # --- TIME-DEPENDENT K-FACTOR RECONCILIATION ---
            # bin_res['k_time_series'] is the fractional empirical noise (n_time,)
            empirical_noise_t = bin_res["gp_err"]

            # Calculate fractional pipeline noise at every time step (n_time,)
            bin_flux_phys = np.nansum(flux_2d[:, mask], axis=1)
            bin_err_phys = np.sqrt(np.nansum(err_2d[:, mask]**2, axis=1))
            pipeline_noise_t = bin_err_phys / bin_flux_phys

            # Calculate the K-factor array (n_time,)
            k_factors = empirical_noise_t / pipeline_noise_t
            
            # Apply to pixels: multiply each time-row by its specific k-factor
            # corrected_err[time, wavelength] = err_2d[time, wavelength] * k_t[time]
            corrected_err = err_2d[:, mask] * k_factors[:, None]

            pixel_results[bin_res["label"]] = {
                "wavelengths": waves[mask],
                "fluxes_corr_jy": corrected_flux_2d,        # In Jy
                "errors_corr_jy": corrected_err, # In Jy
                "err_factors": k_factors
            }
    return pixel_results