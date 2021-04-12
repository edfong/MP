from scipy.optimize import minimize
import numpy as np
from collections import namedtuple
import time
from tqdm import tqdm

#import jax
import jax.numpy as jnp
from jax import vmap
from jax.random import permutation,PRNGKey,split

#import package functions
from . import copula_density_functions as mvcd
from . import copula_regression_functions as mvcr
from . import sample_copula_regression_functions as samp_mvcr
from . import sample_copula_density_functions as samp_mvcd

### Fitting ###
#Compute overhead v_{1:n}, return fit copula object for prediction
def fit_copula_jregression(y,x,n_perm = 10, seed = 20,n_perm_optim = None, single_bandwidth = True):
    #Set seed for scipy
    np.random.seed(seed)
    
    #Combine x,y
    z = jnp.concatenate((x,y.reshape(-1,1)), axis = 1)

    #Generate random permutations
    key = PRNGKey(seed)
    key,*subkey = split(key,n_perm +1 )
    subkey = jnp.array(subkey)
    z_perm = vmap(permutation,(0,None))(subkey,z)

    #Initialize parameter and put on correct scale to lie in [0,1]
    d = jnp.shape(z)[1]

    if single_bandwidth == True:
        rho_init = 0.9*jnp.ones(1)
    else:
        rho_init = 0.9*jnp.ones(d) 
    hyperparam_init = jnp.log(1/rho_init - 1) 


    #calculate rho_opt
    #either use all permutations or a selected number to fit bandwidth
    if n_perm_optim is None:
        z_perm_opt = z_perm
    else:
        z_perm_opt = z_perm[0:n_perm_optim]

    #Compiling
    print('Compiling...')
    start = time.time()

    #Condit
    temp = mvcr.fun_jcll_perm_sp(hyperparam_init,z_perm_opt)
    temp = mvcr.grad_jcll_perm_sp(hyperparam_init,z_perm_opt)

    temp = mvcd.update_pn_loop_perm(rho_init,z_perm)[0].block_until_ready()
    end = time.time()
    print('Compilation time: {}s'.format(round(end-start, 3)))

    print('Optimizing...')
    start = time.time()
    # Condit preq loglik
    opt = minimize(fun = mvcr.fun_jcll_perm_sp, x0= hyperparam_init,\
                     args = (z_perm_opt),jac =mvcr.grad_jcll_perm_sp,method = 'SLSQP') 

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
    vn_perm= mvcd.update_pn_loop_perm(rho_opt,z_perm)[0].block_until_ready()
    end = time.time()
    print('Fit time: {}s'.format(round(end-start, 3)))

    copula_jregression_obj = namedtuple('copula_jregression_obj',['vn_perm','rho_opt','preq_loglik'])
    return copula_jregression_obj(vn_perm,rho_opt,-opt.fun)

#Predict on test data using average permutations
def predict_copula_jregression(copula_jregression_obj,y_test,x_test):
    #Combine x,y
    z_test = jnp.concatenate((x_test,y_test.reshape(-1,1)), axis = 1)

    print('Predicting...')
    start = time.time()
    logcdf_conditionals,logpdf_joints = mvcd.update_ptest_loop_perm_av(copula_jregression_obj.vn_perm,copula_jregression_obj.rho_opt,z_test)
    logcdf_conditionals = logcdf_conditionals.block_until_ready() #for accurate timing
    end = time.time()
    print('Prediction time: {}s'.format(round(end-start, 3)))
    return logcdf_conditionals,logpdf_joints
### ###

### Predictive resampling ###
#Forward sampling without diagnostics for speed
def predictive_resample_jregression(copula_jregression_obj,y_test,x_test,B_postsamples, T_fwdsamples = 5000, seed = 100):
    #Fit permutation averaged cdf/pdf
    logcdf_conditionals,logpdf_joints = predict_copula_jregression(copula_jregression_obj,y_test,x_test)

    #Initialize random seeds
    key = PRNGKey(seed)
    key,*subkey = split(key,B_postsamples+1)
    subkey = jnp.array(subkey)

    #Forward sample
    n = jnp.shape(copula_jregression_obj.vn_perm)[1] #get original data size
    print('Predictive resampling...')
    start = time.time()
    logcdf_conditionals_pr,logpdf_joints_pr = samp_mvcd.predictive_resample_loop_B(subkey,logcdf_conditionals,logpdf_joints,\
                                                                                      copula_jregression_obj.rho_opt,n,T_fwdsamples)
    logcdf_conditionals_pr = logcdf_conditionals_pr.block_until_ready() #for accurate timing
    end = time.time()
    print('Predictive resampling time: {}s'.format(round(end-start, 3)))
    return logcdf_conditionals_pr,logpdf_joints_pr

#Check convergence by running 1 long forward sample chain
def check_convergence_pr_jregression(copula_jregression_obj,y_test,x_test,B_postsamples,T_fwdsamples = 10000, seed = 100):
    #Fit permutation averaged cdf/pdf
    logcdf_conditionals,logpdf_joints = predict_copula_jregression(copula_jregression_obj,y_test,x_test)

    # #Initialize random seeds
    key = PRNGKey(seed)
    key,*subkey = split(key,B_postsamples+1)
    subkey = jnp.array(subkey)

    #Forward sample
    n = jnp.shape(copula_jregression_obj.vn_perm)[1] #get original data size
    print('Predictive resampling...')
    start = time.time()
    logcdf_conditionals_pr,logpdf_joints_pr,pdiff,cdiff = samp_mvcr.pr_loop_conv_jregression_B(subkey,logcdf_conditionals,logpdf_joints,\
                                                                                                    copula_jregression_obj.rho_opt,n,T_fwdsamples)
    logcdf_conditionals_pr = logcdf_conditionals_pr.block_until_ready() #for accurate timing
    end = time.time()
    print('Predictive resampling time: {}s'.format(round(end-start, 3)))
    return logcdf_conditionals_pr,logpdf_joints_pr,pdiff,cdiff
### ###