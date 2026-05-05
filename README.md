# Lightcurve Jitter Correction Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Welcome to my final project for ASTR5470: Computational Astronomy! In this package I use Gaussian Process regression along with a Markov Chain Monte Carlo code to find systematic errors in light curves. The systematic error will be removed, correcting the signal from the light curve and propagating this out into the spectrum. We will create a python package with a dedicated GitHub repository and documentation for users to read.

## 🌌 Scientific Background

Brown dwarf time-series observations are often contaminated by instrumental systematics. This pipeline uses **Gaussian Process (GP) Regression** to model the light curve as a combination of:
1.  **Astrophysical Signal:** The "Slow Trend" representing the brown dwarf's intrinsic variability.
2.  **Correlated Noise:** The "Fast Trend" representing periodic detector jitter.
---

## 🚀 Getting Started

### Prerequisites
Ensure you have a Python environment (3.10 or higher) with the following libraries installed:
* `numpy`
* `scipy`
* `matplotlib`
* `george` (Gaussian Process library)
* `emcee` (MCMC library)
* `h5py` (For handling JWST data formats)
* `pytest` (For running the test suite)

### Installation
Clone this repository to your local machine:
```bash
git clone [https://github.com/danyaalb/pipeline_code.git](https://github.com/danyaalb/pipeline_code.git)
cd pipeline_code

### Directory structure
pipeline_code: Contains modeling, correction, and plotting logic.
data: Directory for input .h5 data files.
outputs: Generated diagnostic plots (PNG/PDF).
pipeline_code/run_pipeline.py: File that runs the code.
pipeline_code/test_pipeline.py: Pytest suite for systematic error validation.

### How to run this code
1. Clone repository to local machine
2. Change inputs in pipeline_code/set_inputs.yaml
3. Run in terminal ```pythono run_pipeline.py
4. Check your outputs folder for corrected files and new H5 files (H5 files for the current pdf outputs not shown because it is proprietary data).

## For more information about assumptions for GP fitting and details about the code, read the GitHub Wiki.

I acknowledge AI helped contribute some code for this project with detailed oversight and corrections/re-formatting made by myself.
