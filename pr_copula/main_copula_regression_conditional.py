from scipy.optimize import minimize
from collections import namedtuple
import time
import numpy as np

#import jax
import jax.numpy as jnp
from jax import vmap
from jax.random import permutation,PRNGKey,split

#import package functions
from . import mv_copula_regression as mvcr
from . import sample_mv_copula_regression as samp_mvcr

def fit_copula_cregression(y,x,n_perm = 10, seed = 20,n_perm_optim = None,single_x_bandwidth = True):
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
    temp = mvcr.fun_ccll_perm_sp(hyperparam_init,y_perm_opt,x_perm_opt)
    temp = mvcr.grad_ccll_perm_sp(hyperparam_init,y_perm_opt,x_perm_opt)
    temp = mvcr.update_pn_loop_perm(hyperparam_init[0],hyperparam_init[1:],y_perm,x_perm)[0].block_until_ready()
    end = time.time()
    print('Compilation time: {}s'.format(round(end-start, 3)))

    print('Optimizing...')
    start = time.time()
    # Condit preq loglik
    opt = minimize(fun = mvcr.fun_ccll_perm_sp, x0= hyperparam_init,\
             args = (y_perm_opt,x_perm_opt),jac =mvcr.grad_ccll_perm_sp,method = 'SLSQP',options={'maxiter':100, 'ftol': 1e-4})
    #check optimization succeeded
    if opt.success == False:
        print('Optimization failed')

    #unscale hyperparameter
    hyperparam_opt = opt.x
    rho_opt = 1/(1+jnp.exp(hyperparam_opt[0]))
    #l_scale_opt = jnp.exp(hyperparam_opt[1:])
    rho_opt_x = 1/(1+jnp.exp(hyperparam_opt[1:]))
    end = time.time()


    print('Optimization time: {}s'.format(round(end-start, 3)))
        
    print('Fitting...')
    start = time.time()
    vn_perm= mvcr.update_pn_loop_perm(rho_opt,rho_opt_x,y_perm,x_perm)[0].block_until_ready()
    end = time.time()
    print('Fit time: {}s'.format(round(end-start, 3)))

    copula_cregression_obj = namedtuple('copula_cregression_obj',['vn_perm','rho_opt','rho_x_opt','preq_loglik','x_perm'])
    return copula_cregression_obj(vn_perm,rho_opt,rho_opt_x,-opt.fun,x_perm)


#Predict on test data using average permutations
def predict_copula_cregression(copula_cregression_obj,y_test,x_test):
    #code loop for now, can speed up to use indices
    n_perm = np.shape(copula_cregression_obj.x_perm)[0]
    n = np.shape(copula_cregression_obj.x_perm)[1]
    n_test = np.shape(x_test)[0]
    logk_xx = np.zeros((n_perm,n,n_test))

    print('Predicting...')
    start = time.time()
    logcdf_conditionals,logpdf_joints = mvcr.update_ptest_loop_perm_av(copula_cregression_obj.vn_perm,copula_cregression_obj.rho_opt\
                                                                         ,copula_cregression_obj.rho_x_opt,copula_cregression_obj.x_perm, y_test.reshape(-1,1),x_test)
    logcdf_conditionals = logcdf_conditionals.block_until_ready() #for accurate timing
    end = time.time()
    print('Prediction time: {}s'.format(round(end-start, 3)))
    return logcdf_conditionals,logpdf_joints

#Forward sampling without diagnostics for speed
def predictive_resample_cregression(copula_cregression_obj,x,y_test,x_test,B_postsamples, T_fwdsamples = 5000, seed = 100):
    #Fit permutation averaged cdf/pdf
    logcdf_conditionals,logpdf_joints = predict_copula_cregression(copula_cregression_obj,y_test,x_test)

    #Initialize random seeds
    key = PRNGKey(seed)
    key,*subkey = split(key,B_postsamples+1)
    subkey = jnp.array(subkey)

    #Forward sample
    n = jnp.shape(copula_cregression_obj.vn_perm)[1] #get original data size
    print('Predictive resampling...')
    start = time.time()
    logcdf_conditionals_pr,logpdf_joints_pr = samp_mvcr.predictive_resample_loop_cregression_B(subkey,logcdf_conditionals,logpdf_joints,x,x_test,\
                                                                                                copula_cregression_obj.rho_opt,copula_cregression_obj.rho_x_opt,n,T_fwdsamples)
    logcdf_conditionals_pr = logcdf_conditionals_pr.block_until_ready() #for accurate timing
    end = time.time()
    print('Predictive resampling time: {}s'.format(round(end-start, 3)))
    return logcdf_conditionals_pr,logpdf_joints_pr

#Check convergence by running 1 long forward sample chain
def check_convergence_pr_cregression(copula_cregression_obj,x,y_test,x_test,B_postsamples,T_fwdsamples = 10000, seed = 100):
    #Fit permutation averaged cdf/pdf
    logcdf_conditionals,logpdf_joints = predict_copula_cregression(copula_cregression_obj,y_test,x_test)

    # #Initialize random seeds
    key = PRNGKey(seed)
    key,*subkey = split(key,B_postsamples+1)
    subkey = jnp.array(subkey)

    #Forward sample
    n = jnp.shape(copula_cregression_obj.vn_perm)[1] #get original data size
    print('Predictive resampling...')
    start = time.time()
    logcdf_conditionals_pr,logpdf_joints_pr,pdiff,cdiff = samp_mvcr.pr_loop_conv_cregression_B(subkey,logcdf_conditionals,logpdf_joints,x,x_test,\
                                                                                                copula_cregression_obj.rho_opt,copula_cregression_obj.rho_x_opt,n,T_fwdsamples)
    logcdf_conditionals_pr = logcdf_conditionals_pr.block_until_ready() #for accurate timing
    end = time.time()
    print('Predictive resampling time: {}s'.format(round(end-start, 3)))
    return logcdf_conditionals_pr,logpdf_joints_pr,pdiff,cdiff
