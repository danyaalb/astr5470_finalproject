import h5py
import numpy as np
import shutil
from datetime import datetime
import numpy as np
from .kernels import get_kernel
from .modeling import run_mcmc, get_best_params
import h5py
from scipy.ndimage import generic_filter
def load_h5_data(file_path):
     '''
     Loads the necessary datasets from the H5 file and returns them as numpy arrays.

     Inputs:
     - file_path: str, path to the H5 file

     Outputs:    
     - wave: 1D numpy array of wavelengths
     - time_raw: 1D numpy array of time in MJD
     - flux_raw: 2D numpy array of flux (time x wavelength)
     - err_raw: 2D numpy array of errors (time x wavelength)
     '''
     with h5py.File(file_path, 'r') as h5:
        wave = h5['wave_1d'][:]
        time_raw = h5['time'][:]
        flux_raw = h5['calibrated_optspec'][:]
        err_raw = h5['calibrated_errspec'][:]
     return wave, time_raw, flux_raw, err_raw

def save_corr_data(original_h5, pixel_data, output_path):
    """
    Clones the original H5 and adds 'flux_2d_gp_corrected' and 
    'err_2d_gp_corrected' as new datasets.

    Inputs:
    - original_h5: str, path to the original H5 file
    - pixel_data: dict, keys are wavelength bin labels, values are dicts with 'fluxes_corr_jy' and 'errors_corr_jy'
    - output_path: str, path to save the new H5 file with corrections   

    Outputs:
    - None (saves the file to disk)
    """
    import shutil
    from datetime import datetime
    
    # 1. Create a physical copy to preserve original metadata
    shutil.copyfile(original_h5, output_path)
     
    with h5py.File(output_path, 'r+') as hf:
        waves = hf['wave_1d'][:]
        
        # Start with original calibrated data
        corrected_flux = hf['calibrated_optspec'][:].astype(np.float32)
        corrected_err = hf['calibrated_opterr'][:].astype(np.float64)
        
        # 2. Map the GP corrections into these new arrays
        for label, data in pixel_data.items():
            # Clean up the label to get numeric boundaries
            w_bounds = label.replace('$\mu$m', '').strip().split('-')
            w_start, w_end = float(w_bounds[0]), float(w_bounds[1])
            mask = (waves >= w_start) & (waves < w_end)
            
            # Using the names defined in get_corrected_pixel_data
            corrected_flux[:, mask] = data['fluxes_corr_jy']     
            corrected_err[:, mask] = data['errors_corr_jy']  
            
        # 3. Create/Replace the new datasets
        if 'flux_2d_gp_corrected' in hf:
            del hf['flux_2d_gp_corrected']
        hf.create_dataset('flux_2d_gp_corrected', data=corrected_flux, dtype='float32')

        if 'err_2d_gp_corrected' in hf:
            del hf['err_2d_gp_corrected']
        hf.create_dataset('err_2d_gp_corrected', data=corrected_err, dtype='float64')

        # Deleting some old data (Not needed in the package but won't cause it to crash)

        if 'flux_div_norm' in hf:
            del hf['flux_div_norm']

        if 'flux_sub_norm' in hf:
            del hf['flux_sub_norm']

        if 'flux_raw_norm' in hf:
            del hf['flux_raw_norm']

        hf.attrs['gp_correction_applied'] = True
        hf.attrs['gp_correction_date'] = datetime.now().strftime("%Y-%m-%d_%H%M")

    print(f"File updated successfully: {output_path}")
