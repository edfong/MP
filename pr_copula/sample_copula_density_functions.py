import numpy as np
import scipy as sp
from functools import partial

#import jax functions
import jax.numpy as jnp
from jax import grad,value_and_grad, jit, vmap,jacfwd,jacrev,random
from jax.scipy.stats import norm
from jax.lax import fori_loop
from jax.ops import index_update

#import package functions
from . import copula_density_functions as mvcd
from .utils.BFGS import minimize_BFGS
from .utils.bivariate_copula import ndtri_

### Predictive resampling functions ###

#### Main function ####
# Loop through forward sampling; generate uniform random variables, then use p(y) update from mvcd
@partial(jit,static_argnums = (4,5))
def predictive_resample_single_loop(key,logcdf_conditionals,logpdf_joints,rho,n,T):
    d = jnp.shape(logcdf_conditionals)[0]

    #generate uniform random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (T,d))

    #Append a_rand to empty vn (for correct array size)
    vT = jnp.concatenate((jnp.zeros((n,d)),a_rand),axis = 0)

    #run forward loop
    inputs = vT,logcdf_conditionals,logpdf_joints,rho
    rng = jnp.arange(n,n+T)
    outputs,rng = mvcd.update_ptest_single_scan(inputs,rng)
    vT,logcdf_conditionals,logpdf_joints,rho = outputs

    return logcdf_conditionals,logpdf_joints

## Vmap over multiple test points, then over multiple seeds
predictive_resample_loop = jit(vmap(predictive_resample_single_loop,(None,0,0,None,None,None)),static_argnums = (4,5)) #vmap across y_test
predictive_resample_loop_B =jit(vmap(predictive_resample_loop,(0,None,None,None,None,None)),static_argnums = (4,5)) #vmap across B posterior samples
#### ####

#### Convergence checks ####

# Update p(y) in forward sampling, while keeping a track of change in p(y) for convergence check
def pr_1step_conv(i,inputs):  #t = n+i
    logcdf_conditionals,logpdf_joints,logcdf_conditionals_init,logpdf_joints_init,pdiff,cdiff,rho,n,a_rand = inputs #a is d-dimensional uniform rv
    n_test = jnp.shape(logcdf_conditionals)[0]
    d = jnp.shape(logcdf_conditionals)[1]

    #update pdf/cdf
    logalpha = jnp.log(2- (1/(n+i+1)))-jnp.log(n+i+2)

    u = jnp.exp(logcdf_conditionals)
    v = a_rand[i] #cdf of rv is uniformly distributed

    logcdf_conditionals_new,logpdf_joints_new= mvcd.update_copula(logcdf_conditionals,logpdf_joints,u,v,logalpha,rho)

    #joint density
    pdiff = index_update(pdiff,i,jnp.mean(jnp.abs(jnp.exp(logpdf_joints_new[:,-1])- jnp.exp(logpdf_joints_init[:,-1])))) #mean density diff from initial
    cdiff = index_update(cdiff,i,jnp.mean(jnp.abs(jnp.exp(logcdf_conditionals_new[:,0])- jnp.exp(logcdf_conditionals_init[:,0])))) #mean cdf diff from initial (only univariate)

    outputs = logcdf_conditionals_new,logpdf_joints_new,logcdf_conditionals_init,logpdf_joints_init,pdiff,cdiff,rho,n,a_rand 
    return outputs

#Loop through forward sampling, starting with average p_n
@partial(jit,static_argnums = (4,5))
def pr_loop_conv(key,logcdf_conditionals,logpdf_joints,rho,n,T):
    d = jnp.shape(logcdf_conditionals)[1]

    #generate random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (T,d))

    #Track difference
    pdiff = jnp.zeros(T)
    cdiff = jnp.zeros(T)

    inputs = logcdf_conditionals,logpdf_joints,logcdf_conditionals,logpdf_joints,pdiff,cdiff,rho,n,a_rand 

    #run loop
    outputs = fori_loop(0,T,pr_1step_conv,inputs)
    logcdf_conditionals,logpdf_joints,logcdf_conditionals_init,logpdf_joints_init,pdiff,cdiff,rho,n,a_rand = outputs

    return logcdf_conditionals,logpdf_joints,pdiff,cdiff

## Vmap over random seed to check convergence for multiple samples
pr_loop_conv_B =jit(vmap(pr_loop_conv,(0,None,None,None,None,None)),static_argnums=(4,5))
#### ####
### ###


### Additional utility functions ###

#Compute error between average P_n and quantile
@jit
def calc_pn_av_err2(y0,vn_perm,rho,quantile):
    n = jnp.shape(vn_perm)[1]
    d = jnp.shape(vn_perm)[2]
    
    quantile = quantile.reshape(1,d)

    #compute p_n(y0) through perm avg
    y_test = y0.reshape(1,d)
    logcdf_conditionals_ytest,logpdf_joints_ytest = mvcd.update_ptest_loop_perm_av(vn_perm,rho,y_test) #can sample from each permutation independently
    err2 = jnp.sum((jnp.exp(logcdf_conditionals_ytest)- quantile)**2)
    return err2

grad_pn_av_err2 = jit(grad(calc_pn_av_err2))

#Find quantile P_n^{-1}(u), which can be used for sampling
@jit
def compute_quantile_pn_av(vn_perm,rho,quantile): #delta = 0.5 works well!
    d = jnp.shape(vn_perm)[2] 
    
    #unif rv
    y0_init = ndtri_(quantile)

    #function wrappers for BFGS
    #@jit
    def fun(y0): #wrapper around function evaluation and grad
        return(calc_pn_av_err2(y0,vn_perm,rho,quantile))

    y_samp,err2,n_iter,_ = minimize_BFGS(fun,y0_init,delta_B_init = 0.5)

    return y_samp,err2,n_iter

### ###


