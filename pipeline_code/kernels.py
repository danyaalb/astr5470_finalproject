import george
from george import kernels
import numpy as np

def get_kernel(time, flux, err, p_fast_guess=0.2, p_slow_guess=8.0):
    y_mean = np.mean(flux)
    y_std = np.std(flux)
    
    # Define Kernels (Strictly Fast + Slow)
    k_fast = (y_std*0.3)**2 * kernels.ExpSine2Kernel(gamma=1.0, log_period=np.log(p_fast_guess), ndim=1)
    k_slow = (y_std*0.7)**2 * kernels.ExpSine2Kernel(gamma=1.0, log_period=np.log(p_slow_guess), ndim=1)
 
    kernel = k_fast + k_slow
    gp = george.GP(kernel, mean=y_mean)
    gp.compute(time, err)

    
    return gp, k_fast, k_slow