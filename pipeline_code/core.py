import numpy as np
import h5py
from scipy.ndimage import median_filter

class JWSTVisit:
    """
    A class to handle JWST NIRSpec BOTS data for a single observation epoch.
    
    Attributes:
        date (str): Date of observation (YYYY-MM-DD).
        time (array): Time in hours relative to a global reference.
        white_flux (array): Normalized white light curve flux.
        white_error (array): Empirical noise estimate for the light curve.
    """
    
    def __init__(self, date, h5_path, global_mjd_min):
        """
        Initializes the visit, loads data, and performs normalization.
        
        Args:
            date (str): The date string for the visit.
            h5_path (str): Full path to the Eureka! Stage 3 .h5 file.
            global_mjd_min (float): The earliest MJD across all visits to 
                                    set the 'zero' point for time in hours.
        """
        self.date = date
        self.h5_path = h5_path

        # 1. Load Data
        with h5py.File(h5_path, 'r') as hf:
            wave_raw = hf['wave_1d'][:]
            time_raw = hf['time'][:] # MJD
            flux_2d_raw = hf['calibrated_optspec'][:]
            err_2d_raw = hf['calibrated_opterr'][:]
            
        # 2. Sort by time (ensures GP and MCMC run correctly)
        sort_idx = np.argsort(time_raw)
        time_raw = time_raw[sort_idx]
        flux_2d_raw = flux_2d_raw[sort_idx, :]
        err_2d_raw = err_2d_raw[sort_idx, :]

        # 3. Generate White Light Curve
        w_flux_raw = np.nansum(flux_2d_raw, axis=1)
        w_err_raw = np.sqrt(np.nansum(err_2d_raw**2, axis=1))
        
        # 4. Create Masking
        self.time_mask = np.isfinite(w_flux_raw) & (w_flux_raw > 0)
        
        # 5. Unit Conversion & Normalization
        # Here we use the 'passed in' global_mjd_min instead of a global variable
        self.time = (time_raw[self.time_mask] - global_mjd_min) * 24.0
        
        norm_val = np.nanmedian(w_flux_raw[self.time_mask])
        
        self.wave_1d = wave_raw
        self.white_flux = w_flux_raw[self.time_mask] / norm_val
        self.eureka_error = w_err_raw[self.time_mask] / norm_val
        self.flux_2d = flux_2d_raw[self.time_mask, :]
        self.err_2d = err_2d_raw[self.time_mask, :]

        # 6. Empirical Noise Calculation
        self._calculate_empirical_noise()

    def _calculate_empirical_noise(self):
        """Internal helper to estimate scatter from residuals."""
        lc_trend = median_filter(self.white_flux, size=5)
        residuals = self.white_flux - lc_trend
        scatter = np.nanstd(residuals)
        self.white_error = np.full_like(self.white_flux, scatter)

    def __repr__(self):
        return f"JWSTVisit(Date={self.date}, Points={len(self.time)})"


def load_all_visits(config):
    """
    Directly loads visits using the full paths provided in the config.
    """
    all_start_mjds = []
    
    # First pass: find the global minimum MJD across all files
    for v_info in config['visits']:
        file_path = v_info['path']
        try:
            with h5py.File(file_path, 'r') as f:
                all_start_mjds.append(np.nanmin(f['time'][:]))
        except FileNotFoundError:
            print(f"❌ Error: Could not find file at {file_path}")
            raise 

    global_min = np.min(all_start_mjds)
    
    # Second pass: Create the JWSTVisit objects
    visits = []
    for v_info in config['visits']:
        v_obj = JWSTVisit(v_info['date'], v_info['path'], global_min)
        visits.append(v_obj)
        
    return visits