from scipy.optimize import minimize
from collections import namedtuple
import time
import numpy as np

#import jax
import jax.numpy as jnp
from jax import vmap
from jax.random import permutation,PRNGKey,split

#import package functions
from . import copula_classification_functions as mvcc
from . import sample_copula_classification_functions as samp_mvcc

### Fitting ###
#Compute overhead v_{1:n}, return fit copula object for prediction
def fit_copula_classification(y,x,n_perm = 10, seed = 20,n_perm_optim = None,single_x_bandwidth = True):
    #Set seed for scipy
    np.random.seed(seed)

    #Generate random permutations
    key = PRNGKey(seed)
    key,*subkey = split(key,n_perm +1 )
    subkey = jnp.array(subkey)
    y_perm = vmap(permutation,(0,None))(subkey,y).reshape(n_perm,-1,1)
    x_perm = vmap(permutation,(0,None))(subkey,x)

    #Initialize parameter and put on correct scale to lie in [0,1]
    d = jnp.shape(x)[1]
    if single_x_bandwidth ==True:
        rho_init = 0.8*jnp.ones(2) 
    else: 
        rho_init = 0.8*jnp.ones(d+1)
    hyperparam_init = jnp.log(1/rho_init - 1) 

    #calculate rho_opt
    #either use all permutations or a selected number to fit bandwidth
    if n_perm_optim is None:
        y_perm_opt = y_perm
        x_perm_opt = x_perm
    else:
        y_perm_opt = y_perm[0:n_perm_optim]
        x_perm_opt = x_perm[0:n_perm_optim]

    #Compiling
    print('Compiling...')
    start = time.time()
    #temp = mvcc.fun_grad_ccll_perm_sp(hyperparam_init,y_perm_opt,x_perm_opt) #value and grad is slower for many parameters
    temp = mvcc.fun_ccll_perm_sp(hyperparam_init,y_perm_opt,x_perm_opt)
    temp = mvcc.grad_ccll_perm_sp(hyperparam_init,y_perm_opt,x_perm_opt)
    temp = mvcc.update_pn_loop_perm(hyperparam_init[0],hyperparam_init[1:],y_perm,x_perm)[0].block_until_ready()
    end = time.time()
    print('Compilation time: {}s'.format(round(end-start, 3)))

    print('Optimizing...')
    start = time.time()
    # Condit preq loglik
    opt = minimize(fun = mvcc.fun_ccll_perm_sp, x0= hyperparam_init,\
        args = (y_perm_opt,x_perm_opt),jac = mvcc.grad_ccll_perm_sp,method = 'SLSQP',options={'maxiter':100, 'ftol': 1e-4})

    #check optimization succeeded
    if opt.success == False:
        print('Optimization failed')

    #unscale hyperparameter
    hyperparam_opt = opt.x
    rho_opt = 1/(1+jnp.exp(hyperparam_opt[0]))
    rho_opt_x = 1/(1+jnp.exp(hyperparam_opt[1:]))
    end = time.time()


    print('Optimization time: {}s'.format(round(end-start, 3)))
        
    print('Fitting...')
    start = time.time()
    log_vn,logpmf_yn_perm,*_= mvcc.update_pn_loop_perm(rho_opt,rho_opt_x,y_perm,x_perm)
    log_vn = log_vn.block_until_ready()
    end = time.time()
    print('Fit time: {}s'.format(round(end-start, 3)))

    copula_classification_obj = namedtuple('copula_classification_obj',['log_vn_perm','logpmf_yn_perm','rho_opt','rho_x_opt','preq_loglik','y_perm','x_perm'])
    return copula_classification_obj(log_vn,logpmf_yn_perm,rho_opt,rho_opt_x,-opt.fun,y_perm,x_perm)

#Returns p(y=1 |x)
def predict_copula_classification(copula_classification_obj,x_test):
    #code loop for now, can speed up to use indices
    n_perm = np.shape(copula_classification_obj.x_perm)[0]
    n = np.shape(copula_classification_obj.x_perm)[1]
    n_test = np.shape(x_test)[0]
    logk_xx = np.zeros((n_perm,n,n_test))

    print('Predicting...')
    start = time.time()
    logpmf = mvcc.update_ptest_loop_perm_av(copula_classification_obj.log_vn_perm,copula_classification_obj.rho_opt,copula_classification_obj.rho_x_opt\
                                                                        ,copula_classification_obj.y_perm,copula_classification_obj.x_perm,x_test)
    logpmf = logpmf.block_until_ready() #for accurate timing
    end = time.time()
    print('Prediction time: {}s'.format(round(end-start, 3)))
    return logpmf
###

### Predictive Resampling ###
#Forward sampling: we can draw y,x directly as it is binary
def predictive_resample_classification(copula_classification_obj,y,x,x_test,B_postsamples, T_fwdsamples = 5000, seed = 100):
    #Fit permutation averaged cdf/pdf
    logpmf1_yn_av = predict_copula_classification(copula_classification_obj,x)
    logpmf1_ytest_av = predict_copula_classification(copula_classification_obj,x_test)

    #Initialize random seeds
    key = PRNGKey(seed)
    key,*subkey = split(key,B_postsamples+1)
    subkey = jnp.array(subkey)

    #Forward sample
    print('Predictive resampling...')
    start = time.time()
    logpmf_ytest_samp, logpmf_yn_samp,y_samp,x_samp,pdiff = samp_mvcc.forward_sample_y_samp_B(subkey,logpmf1_ytest_av,logpmf1_yn_av,y,x,x_test,\
                               copula_classification_obj.rho_opt, copula_classification_obj.rho_x_opt,T_fwdsamples)
    y_samp = y_samp.block_until_ready()
    end = time.time()
    print('Predictive resampling time: {}s'.format(round(end-start, 3)))
    return logpmf_ytest_samp,logpmf_yn_samp,y_samp,x_samp,pdiff
### ###

