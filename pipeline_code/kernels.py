import george
from george import kernels
import numpy as np

def get_kernel(time, flux, err, p_fast_guess=0.2, p_slow_guess=8.0):
    """
    Constructs a composite kernel for the GP model, consisting of a "fast" component to capture 
    short-term jitter and a "slow" component for long-term trends.
    
    :param time: 1D array of time points (in hours)
    :type time: np.ndarray
    :param flux: 1D array of flux values (normalized)
    :type flux: np.ndarray
    :param err: 1D array of flux uncertainties (normalized)
    :type err: np.ndarray
    :param p_fast_guess: Initial guess for the period of the fast component (in hours) (default: 0.2)
    :type p_fast_guess: float
    :param p_slow_guess: Initial guess for the period of the slow component (in hours) (default: 8.0)
    :type p_slow_guess: float
    :return: Tuple containing:
        - gp (george.GP): Initialized george.GP object with composite kernel and ready for fitting
        - k_fast: The "fast" kernel component (useful for plotting and analysis)
        - k_slow: The "slow" kernel component (useful for plotting and analysis)
    :rtype: tuple
    """
    y_mean = np.mean(flux)
    y_std = np.std(flux)
    
    # Define Kernels (Strictly Fast + Slow)
    k_fast = (y_std*0.3)**2 * kernels.ExpSine2Kernel(gamma=1.0, log_period=np.log(p_fast_guess), ndim=1)
    k_slow = (y_std*0.7)**2 * kernels.ExpSine2Kernel(gamma=1.0, log_period=np.log(p_slow_guess), ndim=1)
 
    kernel = k_fast + k_slow
    gp = george.GP(kernel, mean=y_mean)
    gp.compute(time, err)

    return gp, k_fast, k_slow