import jax.numpy as jnp
import time
import pandas as pd
import numpy as np

#import copula functions
from pr_copula.main_copula_classification import fit_copula_classification,predict_copula_classification,\
                                        predictive_resample_classification
from jax.random import PRNGKey,split

## MOON ##
#Load moon dataset
print('Dataset: Moon')
from sklearn.datasets import make_moons
seed = 52
n = 100
n_test = 5000
noise = 0.3
x_temp,y_temp= make_moons(n+n_test,noise = noise,random_state = seed)
y = y_temp[0:n]
x = x_temp[0:n]
y_test = y_temp[n:n+n_test]
x_test = x_temp[n:n+n_test]

d = np.shape(x)[1]
d_gp = d

#normalize
mean_norm = np.mean(x,axis = 0)
std_norm = np.std(x,axis = 0)
x = (x - mean_norm)/std_norm
x_test = (x_test - mean_norm)/std_norm

y_plot = np.array([0])
x_plot = np.linspace(-4,4.1,25)
x_meshgrid = np.meshgrid(x_plot,x_plot)
x_plot_grid= jnp.array([x_meshgrid[0].ravel(),x_meshgrid[1].ravel()]).transpose()

#Fit copula obj
copula_classification_obj = fit_copula_classification(jnp.array(y),jnp.array(x),single_x_bandwidth = False,n_perm_optim = 10)
print('Bandwidth is {}'.format(copula_classification_obj.rho_opt))
print('Bandwidth is {}'.format(copula_classification_obj.rho_x_opt))
print('Preq loglik is {}'.format(copula_classification_obj.preq_loglik/n))

#Predict Yplot
logpmf1 = predict_copula_classification(copula_classification_obj,x_plot_grid)
pmf1 = jnp.exp(logpmf1)
jnp.save('plot_files/ccopula_moon_pmf',pmf1)

#Predictive Resample
B = 1000
T = 5000
logpmf_ytest_samp,logpmf_yn_samp,y_samp,x_samp,pdiff = predictive_resample_classification(copula_classification_obj,y,x,x_plot_grid,B,T)

jnp.save('plot_files/ccopula_moon_logpmf_ytest_pr',logpmf_ytest_samp)
jnp.save('plot_files/ccopula_moon_logpmf_yn_pr',logpmf_yn_samp)

#Convergence
T = 10000 #T = 10000, seed = 50 for i = 30
seed = 200
_,_,_,_,pdiff = predictive_resample_classification(copula_classification_obj,y,x,x_test[0:1],1,T,seed = seed)
jnp.save('plot_files/ccopula_moon_pdiff',pdiff)

#Gaussian Process
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,WhiteKernel 
from sklearn.gaussian_process import GaussianProcessClassifier

kernel =ConstantKernel()*RBF() + WhiteKernel()
gp = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=10).fit(x, y.reshape(-1,))
p_pred = gp.predict_proba(np.array([x_meshgrid[0].ravel(),x_meshgrid[1].ravel()]).transpose())
jnp.save('plot_files/gp_moon_pred',p_pred)

