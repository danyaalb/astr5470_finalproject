import yaml
import os
import numpy as np
import h5py
from pipeline_code.core import JWSTVisit, load_all_visits
from pipeline_code import modeling, correction, plotting, kernels, file_formatting

def main():

    # 1. Load the simple YAML
    with open("set_inputs.yaml", "r") as f:
        config = yaml.safe_load(f)

    #Create the output directory if it doesn't exist
    if not os.path.exists(config['output_directory']):
        os.makedirs(config['output_directory'])

    # 2. Load the data using the direct paths
    # We use the manager function to calculate the global MJD min 
    # and return a list of prepared JWSTVisit objects
    visits = load_all_visits(config)


    print("Loading data and calculating global time references...")

    # --- 3. Execution Loop ---
    for v in visits:
        print(f"\n{'='*30}")
        print(f"Processing Visit: {v.date}")
        print(f"{'='*30}")
        
        # A. Setup GP Kernels
        # Using parameters from your config file for flexibility
        gp_settings = config['gp_settings']
        gp_obj, kf, ks = kernels.get_kernel(
            v.time, 
            v.white_flux, 
            v.white_error, 
            gp_settings['fast_scale'], 
            gp_settings['slow_scale']
        )
        
        # B. Run MCMC Modeling
        print("Running MCMC...")
        samples, mcmc_results, res = modeling.run_mcmc(
            v.time, 
            v.white_flux, 
            gp_obj, 
            config['mcmc_bounds'],  # Points to the list of 6 pairs
            n_walkers=config['mcmc_controls']['n_walkers'],
            n_steps=config['mcmc_controls']['n_steps']
)

        # C. Extract Best-Fit & Shaded Uncertainties
        x_pred, mu, gp_std, mu_f, mu_s, residuals, mcmc_mu_samples = modeling.get_best_params(
            gp_obj, v.time, v.white_flux, samples, mcmc_results, kf, ks
        )

        # D. Binned Spectral Analysis
        print("Performing binned spectral correction...")
        # Defining the range (using full wave range from the object)
        entire_range = [(v.wave_1d[0], v.wave_1d[-1])]
        
        binned_results = correction.run_binned_analysis(
            v.time,
            v.flux_2d,
            v.err_2d,
            v.wave_1d,
            gp_obj,
            mcmc_results,
            kf,
            ks,
            wav_ranges=entire_range
        )

        # E. Correct Jitter and Save
        pixel_data = correction.get_corrected_pixel_data(v.h5_path, binned_results)
        
        output_filename = f"{v.date}_GP_corrected.h5"
        final_output_path = os.path.join(config['output_directory'], output_filename)
        
        file_formatting.save_corr_data(v.h5_path, pixel_data, final_output_path)
        print(f"Saved corrected data to: {final_output_path}")

        # F. Visualization
        if config.get('generate_plots', True):
            plotting.plot_results(
                v.time, 
                v.white_flux, 
                v.white_error, 
                x_pred, 
                mu, 
                gp_std, 
                mu_f, 
                mu_s, 
                residuals, 
                samples, 
                mcmc_results, 
                mcmc_mu_samples, 
                v.date,                # Argument 13
                config['output_directory'] # Pass the directory for saving!
            )
        # G. Binned Diagnostics 
        if config.get('generate_plots', True):
            plotting.plot_binned_results(
                binned_results, 
                v.date, 
                config['output_directory']
            )

    print("\nPipeline execution complete for all visits.")

if __name__ == "__main__":
    main()