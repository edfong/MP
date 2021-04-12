import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap,jacfwd,jacrev,random,remat,value_and_grad
from jax.scipy.special import ndtri,erfc,logsumexp
from jax.scipy.stats import norm
from jax import random
from jax.lax import fori_loop,scan
from jax.ops import index, index_add, index_update
import scipy as osp
from functools import partial
from scipy.optimize import minimize,root

from tqdm import tqdm_notebook

from . import copula_density_functions as mvcd
from . import sample_copula_density_functions as samp_mvcd
from . import copula_regression_functions as mvcr
from .utils.BFGS import minimize_BFGS
from .utils.bivariate_copula import ndtri_

## Joint method regression ##

### Predictive resampling functions are same as samp_mvcd ###

#### Convergence checks for PR ####
# Update p(y|x) in forward sampling, while keeping a track of change in p(y|x) for convergence check
def pr_1step_conv_jregression(i,inputs):  #t = n+i
    logcdf_conditionals,logpdf_joints,logcdf_conditionals_init,logpdf_joints_init,pdiff,cdiff,rho,n,a_rand = inputs #a is d-dimensional uniform rv
    n_test = jnp.shape(logcdf_conditionals)[0]
    d = jnp.shape(logcdf_conditionals)[1]

    #update pdf/cdf
    logalpha = jnp.log(2- (1/(n+i+1)))-jnp.log(n+i+2)

    u = jnp.exp(logcdf_conditionals)
    v = a_rand[i] #cdf of rv is uniformly distributed

    logcdf_conditionals_new,logpdf_joints_new= mvcd.update_copula(logcdf_conditionals,logpdf_joints,u,v,logalpha,rho)

    #conditional density
    pdiff = index_update(pdiff,i,jnp.mean(jnp.abs(jnp.exp(logpdf_joints_new[:,-1] - logpdf_joints_new[:,-2])- jnp.exp(logpdf_joints_init[:,-1]-logpdf_joints_init[:,-2])))) #mean density diff from initial
    cdiff = index_update(cdiff,i,jnp.mean(jnp.abs(jnp.exp(logcdf_conditionals_new[:,-1])- jnp.exp(logcdf_conditionals_init[:,-1])))) #mean condit cdf diff from initial

    outputs = logcdf_conditionals_new,logpdf_joints_new,logcdf_conditionals_init,logpdf_joints_init,pdiff,cdiff,rho,n,a_rand 
    return outputs

#Loop through forward sampling, starting with average p_n
@partial(jit,static_argnums = (4,5))
def pr_loop_conv_jregression(key,logcdf_conditionals,logpdf_joints,rho,n,T):
    d = jnp.shape(logcdf_conditionals)[1]

    #generate random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (T,d))

    #Track difference
    pdiff = jnp.zeros(T)
    cdiff = jnp.zeros(T)

    inputs = logcdf_conditionals,logpdf_joints,logcdf_conditionals,logpdf_joints,pdiff,cdiff,rho,n,a_rand 

    #run loop
    outputs = fori_loop(0,T,pr_1step_conv_jregression,inputs)
    logcdf_conditionals,logpdf_joints,logcdf_conditionals_init,logpdf_joints_init,pdiff,cdiff,rho,n,a_rand = outputs

    return logcdf_conditionals,logpdf_joints,pdiff,cdiff

## Vmap over random seed to check convergence for multiple samples
pr_loop_conv_jregression_B =jit(vmap(pr_loop_conv_jregression,(0,None,None,None,None,None)),static_argnums=(4,5))
#### ####
### ###
## ##

## Conditional method regression ###

### Predictive resampling functions ###

#### Main function ####
# Loop through forward sampling; generate uniform random variables, then use p(y|x) update from mvcr
@partial(jit,static_argnums = (7,8))
def predictive_resample_single_loop_cregression(key,logcdf_conditionals,logpdf_joints,x,x_test,rho,rho_x,n,T):

    #generate uniform random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (T,1))

    #Draw random x_samp from BB
    key, subkey = random.split(key) #split key
    n = jnp.shape(x)[0]
    w = random.dirichlet(subkey, jnp.ones(n)) #single set of dirichlet weights
    key, subkey = random.split(key) #split key
    ind_new = random.choice(key,a = jnp.arange(n),p = w,shape = (1,T))[0]
    x_new = x[ind_new] 

    # #Draw from KDE (this is experimental)
    # key, subkey = random.split(key) #split key
    # d = jnp.shape(x)[1]
    # x_new = x_new + 0.5*random.normal(key, shape = (T,d))

    #Append a_rand to empty vn (for correct array size)
    vT = jnp.concatenate((jnp.zeros((n,1)),a_rand),axis = 0)
    x_samp = jnp.concatenate((x,x_new),axis = 0)
    

    #run forward loop
    inputs = vT,logcdf_conditionals,logpdf_joints,x_samp,x_test,rho,rho_x
    rng = jnp.arange(n,n+T)
    outputs,rng = mvcr.update_ptest_single_scan(inputs,rng)
    _,logcdf_conditionals,logpdf_joints,*_ = outputs

    return logcdf_conditionals,logpdf_joints

## Vmap over multiple test points, then over multiple seeds
predictive_resample_loop_cregression = jit(vmap(predictive_resample_single_loop_cregression,(None,0,0,None,0,None,None,None,None)),static_argnums = (7,8)) #vmap across y_test
predictive_resample_loop_cregression_B =jit(vmap(predictive_resample_loop_cregression,(0,None,None,None,None,None,None,None,None)),static_argnums = (7,8)) #vmap across B posterior samples
#### ####

#### Convergence checks ####

# Update p(y|x) in forward sampling, while keeping a track of change in p(y|x) for convergence check
def pr_1step_conv_cregression(i,inputs):  #t = n+i
    logcdf_conditionals,logpdf_joints,x,x_test,rho,rho_x,n,a_rand,\
    logcdf_conditionals_init,logpdf_joints_init,pdiff,cdiff = inputs #a is d-dimensional uniform rv

    n_test = jnp.shape(logcdf_conditionals)[0]
    d = jnp.shape(logcdf_conditionals)[1]

    #update pdf/cdf
    x_new = x[i]
    logalpha = jnp.log(2- (1/(n+i+1)))-jnp.log(n+i+2)

    #compute x rhos/alphas
    logk_xx = mvcr.calc_logkxx(x_test,x_new,rho_x)
    logalphak_xx = logalpha + logk_xx
    log1alpha = jnp.log1p(-jnp.exp(logalpha))
    logalpha_x = (logalphak_xx) - (jnp.logaddexp(log1alpha,logalphak_xx))

    u = jnp.exp(logcdf_conditionals)
    v = a_rand[i] #cdf of rv is uniformly distributed

    logcdf_conditionals_new,logpdf_joints_new= mvcr.update_copula(logcdf_conditionals,logpdf_joints,u,v,logalpha_x,rho)

    #conditional density
    pdiff = index_update(pdiff,i,jnp.mean(jnp.abs(jnp.exp(logpdf_joints_new[:,-1])- jnp.exp(logpdf_joints_init[:,-1])))) #mean density diff from initial
    cdiff = index_update(cdiff,i,jnp.mean(jnp.abs(jnp.exp(logcdf_conditionals_new[:,-1])- jnp.exp(logcdf_conditionals_init[:,-1])))) #mean condit cdf diff from initial

    outputs =  logcdf_conditionals_new,logpdf_joints_new,x,x_test,rho,rho_x,n,a_rand,\
                logcdf_conditionals_init,logpdf_joints_init,pdiff,cdiff
    return outputs

#Loop through forward sampling, starting with average p_n
@partial(jit,static_argnums = (7,8))
def pr_loop_conv_cregression(key,logcdf_conditionals,logpdf_joints,x,x_test,rho,rho_x,n,T):
    d = jnp.shape(logcdf_conditionals)[1]

    #generate uniform random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (T,1))

    #Draw random x_samp from BB
    key, subkey = random.split(key) #split key
    n = jnp.shape(x)[0]
    w = random.dirichlet(subkey, jnp.ones(n)) #single set of dirichlet weights
    key, subkey = random.split(key) #split key
    ind_new = random.choice(key,a = jnp.arange(n),p = w,shape = (1,T))[0]
    x_new = x[ind_new]

    #Append a_rand to empty vn (for correct array size)
    x_samp = jnp.concatenate((x,x_new),axis = 0)

    #Track difference
    pdiff = jnp.zeros(T)
    cdiff = jnp.zeros(T)

    inputs = logcdf_conditionals,logpdf_joints,x_samp,x_test,rho,rho_x,n,a_rand,logcdf_conditionals,logpdf_joints,pdiff,cdiff

    #run loop
    outputs = fori_loop(0,T,pr_1step_conv_cregression,inputs)
    logcdf_conditionals,logpdf_joints,x_samp,x_test,rho,rho_x,n,a_rand,logcdf_conditionals,logpdf_joints,pdiff,cdiff = outputs

    return logcdf_conditionals,logpdf_joints,pdiff,cdiff

## Vmap over random seed to check convergence for multiple samples
pr_loop_conv_cregression_B =jit(vmap(pr_loop_conv_cregression,(0,None,None,None,None,None,None,None,None)),static_argnums=(7,8))
#### ####
### ###
## ##