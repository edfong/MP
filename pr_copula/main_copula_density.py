from scipy.optimize import minimize
from collections import namedtuple
import time
import numpy as np
from tqdm import tqdm

#import jax
import jax.numpy as jnp
from jax import vmap
from jax.random import permutation,PRNGKey,split

#import package functions
from . import copula_density_functions as mvcd
from . import sample_copula_density_functions as samp_mvcd

### Fitting ###
#Compute overhead v_{1:n}, return fit copula object for prediction
def fit_copula_density(y,n_perm = 10, seed = 20,n_perm_optim = None, single_bandwidth = True):
    #Set seed for scipy
    np.random.seed(seed)

    #Generate random permutations
    key = PRNGKey(seed)
    key,*subkey = split(key,n_perm +1 )
    subkey = jnp.array(subkey)
    y_perm = vmap(permutation,(0,None))(subkey,y)

    #Initialize parameter and put on correct scale to lie in [0,1]
    d = jnp.shape(y)[1]

    if single_bandwidth == True:
        rho_init = 0.9*jnp.ones(1)
    else:
        rho_init = 0.9*jnp.ones(d) 
    hyperparam_init = jnp.log(1/rho_init - 1) 

    #calculate rho_opt
    #either use all permutations or a selected number to fit bandwidth
    if n_perm_optim is None:
        y_perm_opt = y_perm
    else:
        y_perm_opt = y_perm[0:n_perm_optim]

    #Compiling
    print('Compiling...')
    start = time.time()
    temp = mvcd.fun_jll_perm_sp(hyperparam_init,y_perm_opt)
    temp = mvcd.grad_jll_perm_sp(hyperparam_init,y_perm_opt)
    temp = mvcd.update_pn_loop_perm(rho_init,y_perm)[0].block_until_ready()
    end = time.time()
    print('Compilation time: {}s'.format(round(end-start, 3)))

    print('Optimizing...')
    start = time.time()
    opt = minimize(fun = mvcd.fun_jll_perm_sp, x0= hyperparam_init,\
                     args = (y_perm_opt),jac =mvcd.grad_jll_perm_sp,method = 'SLSQP')

    #check optimization succeeded
    if opt.success == False:
        print('Optimization failed')

    #unscale hyperparameter
    hyperparam_opt = opt.x
    rho_opt = 1/(1+jnp.exp(hyperparam_opt))
    end = time.time()

    print('Optimization time: {}s'.format(round(end-start, 3)))
        
    print('Fitting...')
    start = time.time()
    vn_perm= mvcd.update_pn_loop_perm(rho_opt,y_perm)[0].block_until_ready()
    end = time.time()
    print('Fit time: {}s'.format(round(end-start, 3)))

    copula_density_obj = namedtuple('copula_density_obj',['vn_perm','rho_opt','preq_loglik'])
    return copula_density_obj(vn_perm,rho_opt,-opt.fun)

#Predict on test data using copula object
def predict_copula_density(copula_density_obj,y_test):
    print('Predicting...')
    start = time.time()
    logcdf_conditionals,logpdf_joints = mvcd.update_ptest_loop_perm_av(copula_density_obj.vn_perm,copula_density_obj.rho_opt,y_test)
    logcdf_conditionals = logcdf_conditionals.block_until_ready() #for accurate timing
    end = time.time()
    print('Prediction time: {}s'.format(round(end-start, 3)))
    return logcdf_conditionals,logpdf_joints

#Sample from predcitive density p_n
def sample_copula_density(copula_density_obj,B_samples,seed = 100):
    d = np.shape(copula_density_obj.vn_perm)[2]

    #Compiling
    print('Compiling...')
    start = time.time()
    temp = samp_mvcd.compute_quantile_pn_av(copula_density_obj.vn_perm,copula_density_obj.rho_opt,0.5*np.ones(d))
    end = time.time()
    print('Compilation time: {}s'.format(round(end-start, 3)))

    #Initialize
    y_samp = np.zeros((B_samples,d))
    err = np.zeros(B_samples)
    n_iter = np.zeros(B_samples)

    #Simulate uniform random variables
    np.random.seed(seed)
    un = np.random.rand(B_samples,d)

    #Sampling
    print('Sampling...')
    start = time.time()
    for i in tqdm(range(B_samples)):
        y_samp[i],err[i],n_iter[i] = samp_mvcd.compute_quantile_pn_av(copula_density_obj.vn_perm,copula_density_obj.rho_opt,un[i])
    end = time.time()
    print('Sampling time: {}s'.format(round(end-start, 3)))
    print(f'Max abs error in cdf: {np.sqrt(np.max(err)):.2e}')
    return y_samp,err,n_iter
### ###

### Predictive Resampling ###
#Forward sampling without diagnostics for speed
def predictive_resample_density(copula_density_obj,y_test,B_postsamples, T_fwdsamples = 5000, seed = 100):
    #Fit permutation averaged cdf/pdf
    logcdf_conditionals,logpdf_joints = predict_copula_density(copula_density_obj,y_test)

    #Initialize random seeds
    key = PRNGKey(seed)
    key,*subkey = split(key,B_postsamples+1)
    subkey = jnp.array(subkey)

    #Forward sample
    n = jnp.shape(copula_density_obj.vn_perm)[1] #get original data size
    print('Predictive resampling...')
    start = time.time()
    logcdf_conditionals_pr,logpdf_joints_pr = samp_mvcd.predictive_resample_loop_B(subkey,logcdf_conditionals,logpdf_joints,\
                                                                                      copula_density_obj.rho_opt,n,T_fwdsamples)
    logcdf_conditionals_pr = logcdf_conditionals_pr.block_until_ready() #for accurate timing
    end = time.time()
    print('Predictive resampling time: {}s'.format(round(end-start, 3)))
    return logcdf_conditionals_pr,logpdf_joints_pr

#Check convergence by running 1 long forward sample chain
def check_convergence_pr(copula_density_obj,y_test,B_postsamples,T_fwdsamples = 10000, seed = 100):
    #Fit permutation averaged cdf/pdf
    logcdf_conditionals,logpdf_joints = predict_copula_density(copula_density_obj,y_test)

    # #Initialize random seeds
    key = PRNGKey(seed)
    key,*subkey = split(key,B_postsamples+1)
    subkey = jnp.array(subkey)

    #Forward sample
    n = jnp.shape(copula_density_obj.vn_perm)[1] #get original data size
    print('Predictive resampling...')
    start = time.time()
    logcdf_conditionals_pr,logpdf_joints_pr,pdiff,cdiff = samp_mvcd.pr_loop_conv_B(subkey,logcdf_conditionals,logpdf_joints,\
                                                                                                    copula_density_obj.rho_opt,n,T_fwdsamples)
    logcdf_conditionals_pr = logcdf_conditionals_pr.block_until_ready() #for accurate timing
    end = time.time()
    print('Predictive resampling time: {}s'.format(round(end-start, 3)))
    return logcdf_conditionals_pr,logpdf_joints_pr,pdiff,cdiff
### ###




