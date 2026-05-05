import numpy as np
from .kernels import get_kernel
from .modeling import run_mcmc, get_best_params
from .file_formatting import load_h5_data, save_corr_data
import h5py
import george
from george import kernels
from scipy.optimize import minimize
from scipy.ndimage import generic_filter

def compute_gp_correction(time, flux, err, gp, mcmc_results, k_fast, k_slow):
    '''
    Compute the GP correction factor for a given time series and its associated uncertainties.

    Inputs:
    - time: 1D array of time points
    - flux: 1D array of flux values (normalized)
    - err: 1D array of flux uncertainties (also normalized)
    - gp: george.GP object already fitted to the data
    - mcmc_results: 2D array of MCMC samples (n_samples x n_params)
    - k_fast: The "fast" kernel component
    - k_slow: The "slow" kernel component

    Outputs:
    - A dictionary containing:
        - "time_hr": Original time array
        - "t_smooth": A smooth time grid for plotting (1000 points)
        - "mu_fast": GP prediction for the fast component at original time points
        - "mu_slow": GP prediction for the slow component at original time points
        - "mu_total": Total GP prediction (fast + slow) at original time points
        - "mu_fast_smooth": Fast component on the smooth grid
        - "mu_slow_smooth": Slow component on the smooth grid
        - "flux_factor": The multiplicative correction factor to apply to the flux
        - "flux_raw": The original flux (normalized)
        - "eureka_err": The original errors (normalized)
        - "gp_err": The GP-derived empirical noise estimate (local std of residuals)
        - "flux_corr": The GP-corrected flux (after applying the correction factor)
    '''
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

    Inputs:
    - time: 1D array of time points (in hours)
    - flux_2d: 2D array of flux values (time x wavelength)
    - err_2d: 2D array of flux uncertainties (time x wavelength)
    - wavelengths: 1D array of wavelength values corresponding to the columns of flux_2d
    - gp: george.GP object already fitted to the data
    - mcmc_results: 2D array of MCMC samples (n_samples x n_params)
    - k_fast: The "fast" kernel component
    - k_slow: The "slow" kernel component
    - wav_ranges: List of tuples defining wavelength bins, e.g., [(1.0, 1.2), (1.2, 1.4)]   

    Outputs:
    - A list of dictionaries, each containing the GP correction results for a specific wavelength bin. 
      Each dictionary has the same structure as the output of compute_gp_correction, but with an additional "label" key indicating the wavelength range of the bin.
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
    '''
    Maps the GP correction factors from the binned analysis back to the original pixel-level data.
    Inputs:
    - h5_path: str, path to the original H5 file (to read wavelengths and original flux/error)
    - binned_results: list of dicts, each containing the GP correction results for a specific wavelength bin (output of run_binned_analysis)

    Outputs:
    - pixel_results: dict, mapping each wavelength bin to its corrected pixel-level data.
    '''
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