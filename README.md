# MP: Martingale Posteriors with Copulas
This repository contains the code for the illustrations in the preprint "Martingale Posterior Distributions" by Edwin Fong, Chris Holmes and Stephen G. Walker, which can be found [here](https://arxiv.org/abs/2103.15671). 

We provide the run scripts for obtaining the numerical results and plots in the paper, and the run scripts use our `pr_copula` package. The `pr_copula` package is a general-purpose Python package built on [JAX](https://github.com/google/jax) for predictive bivariate copula updating with predictive resampling, returning density estimates and uncertainty. The package version here is a snapshot for the experiments in the paper - we will likely continue active development on the package in a separate repository. We may include a vignette on how to use the package at a later time. 


## Installation
The `pr_copula` package can be installed by runnning the following in the main folder:
```
python3 setup.py install
```
or the following if you'd like to make edits to the package:
```
python3 setup.py develop
```
We recommend doing so in a clean virtual environment if reproducing the experimental results is of interest.

Please check the [JAX](https://github.com/google/jax) page for instructions on installing JAX for CPU versus GPU usage - both should work with `pr_copula` for general use. Note that for the numerical values and plots in the paper, the CPU version was used for reproducibility due to the non-determinism of GPU calculations, and timing was carried out on the GPU version. For full reproducibility of the experiments in the paper, please use the versions installed by `setup.py`, that is `jax==0.2.6` and `jaxlib==0.1.57`. 

The suggested version of R is ≥4.0 for the MCMC examples. Please install the `dirichletprocess` package [here](https://cran.r-project.org/web/packages/dirichletprocess/index.html).

## Running Experiments
All experiment scripts can be found in `run_expt`. The scripts are prefixed based on the order in which the experiments appear in the paper, and running the Python scripts should involve entering the following in terminal when in the `run_expt` folder, for example:
```
python3 1_univariate_copula.py
```
The R scripts can be run in RStudio or terminal, and should be run after the corresponding Python scripts, as some datasets are simulated/downloaded into the `R_data` folder by the Python scripts. 

Outputs from the experiments (e.g. martingale posterior samples) are stored in `plot_files`, and all plots in the paper can be produced using the Jupyter notebook `Experiments_Plots.ipynb`. Finally, `Misc_Plots.ipynb` produces the other 2 plots in the paper outside the experiments sections.

## Data
For convenience, some [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) datasets are included in `run_expt/data`. These are:
- [Breast Cancer (wdbc)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Concrete](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
- [Ionosphere](http://archive.ics.uci.edu/ml/datasets/Ionosphere)
- [Parkinsons](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- [Wine](https://archive.ics.uci.edu/ml/datasets/wine)
- [Wine Quality (red)](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [Statlog (German credit data)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

Other datasets are imported from the appropriate R/Python packages, or can also be found here:
- [Galaxy](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/galaxies.html)
- [Air Quality](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/airquality.html)
- [LIDAR](http://www.stat.cmu.edu/~larry/all-of-nonpar/=data/lidar.dat)
- [Moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
