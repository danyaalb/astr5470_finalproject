import numpy as np
import emcee
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import corner
from scipy.ndimage import generic_filter

def run_mcmc(time, flux, gp, bounds, n_walkers=32, n_steps=600, burn_in=200):
    '''
    Run MCMC sampling for the GP model. Returns the samples, summary statistics, and the optimization result.

    Inputs:
    - time: 1D array of time points
    - flux: 1D array of flux values (normalized)
    - gp: george.GP object already fitted to the data
    - bounds: list of tuples, each tuple is (min, max) for a parameter
    - n_walkers: int, number of MCMC walkers
    - n_steps: int, total number of MCMC steps
    - burn_in: int, number of steps to discard as burn-in   

    Outputs:
    - samples: 2D array of MCMC samples (n_samples x n_params)  
    - mcmc_results: 2D array of percentiles (16th, 50th, 84th) for each parameter
    - res: Optimization result from the initial MLE step (scipy OptimizeResult object)
    '''
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(flux, quiet=True)

    # 1. MLE Optimization
    res = minimize(neg_ln_like, gp.get_parameter_vector(), method="L-BFGS-B", bounds=bounds)
    
    # 2. MCMC Logic
    def log_prob(p):
        for val, b in zip(p, bounds):
            if b[0] is not None and val < b[0]: return -np.inf
            if b[1] is not None and val > b[1]: return -np.inf
        gp.set_parameter_vector(p)
        lp = gp.log_likelihood(flux, quiet=True)
        return lp if np.isfinite(lp) else -np.inf

    n_walkers, n_dim = n_walkers, len(res.x)
    pos = res.x + 1e-5 * np.random.randn(n_walkers, n_dim)
    
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    samples = sampler.get_chain(discard=burn_in, flat=True)
    mcmc_results = np.percentile(samples, [16, 50, 84], axis=0)
    return samples, mcmc_results, res

def get_best_params(gp, time, flux, samples, mcmc_results, k_fast, k_slow):
    '''
    
    After MCMC, we want to extract the best-fit parameters and compute the GP predictions for plotting and residual analysis.

    Inputs:
    - gp: george.GP object (should be set to the median MCMC parameters before calling this function)
    - time: 1D array of time points
    - flux: 1D array of flux values (normalized)
    - samples: 2D array of MCMC samples (n_samples x n_params)
    - mcmc_results: 2D array of percentiles (16th, 50th, 84th) for each parameter
    - k_fast: The "fast" kernel component
    - k_slow: The "slow" kernel component   
    
    Outputs:
    - x_pred: 1D array of time points for smooth plotting (1000 points)
    - mu: 1D array of GP predictions at x_pred (median parameters)
    - gp_std: 1D array of GP standard deviation at x_pred (median parameters)
    - mu_f: 1D array of the "fast" component of the GP at x_pred
    - mu_s: 1D array of the "slow" component of the GP at x_pred
    - residuals: 1D array of (flux - GP prediction) at the original time points
    - mcmc_mu_samples: list of 1D arrays, each being the GP prediction at x_pred for a random sample from the MCMC chain (used for "spaghetti" plots) 
    '''
    # Set GP to the Median (50th percentile)
    gp.set_parameter_vector(mcmc_results[1])
    x_pred = np.linspace(time.min(), time.max(), 1000)
    x_pred_2d = x_pred[:, None]
    
    # 1. GP Uncertainty (Standard "Shaded Region")
    mu_total, var_total = gp.predict(flux, x_pred_2d, return_var=True)
    mu_fast, _ = gp.predict(flux, x_pred_2d, kernel=k_fast)
    mu_slow, _ = gp.predict(flux, x_pred_2d, kernel=k_slow)

    # 2. MCMC Uncertainty (The "Spaghetti" spread)
    # We take 50 random samples from the MCMC chain to show parameter uncertainty
    mcmc_mu_samples = []
    selection = samples[np.random.randint(len(samples), size=50)]
    for s in selection:
        gp.set_parameter_vector(s)
        mcmc_mu_samples.append(gp.predict(flux, x_pred_2d, return_cov=False))
    
    # Reset GP to median for residuals
    gp.set_parameter_vector(mcmc_results[1])
    mu_at_data = gp.predict(flux, time[:, None], return_cov=False)
    residuals = flux - mu_at_data
    
    return x_pred, mu_total, np.sqrt(var_total), mu_fast, mu_slow, residuals, mcmc_mu_samples
