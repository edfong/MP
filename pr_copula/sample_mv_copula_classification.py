import jax.numpy as jnp
import numpy as onp
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

from . import mv_copula_classification as mvcc
from .utils.BFGS import minimize_BFGS


### CONDITIONAL REGRESSION ###
### PREDICTIVE RESAMPLING ###
@jit
def update_pn_forward(carry,i):
    logpmf_ytest,logpmf_yn,y_samp,pdiff,y,x,x_test,rho,rho_x,ind_new,vT,logpmf_init = carry

    #Sample new x
    x_new = x[ind_new[i]]

    #Sample new y based on unif rv
    y_new = jnp.where((jnp.log(vT[i])<= logpmf_yn[ind_new[i]]),x = 1,y =0)
    log_vn = logpmf_yn[ind_new[i]]
    y_samp = index_update(y_samp,index[i],y_new)
    
    #Update pmf_yn
    #compute x rhos/alphas
    logalpha = jnp.log(2.- (1/(i+1)))-jnp.log(i+2)
    logk_xx = mvcc.calc_logkxx(x,x_new,rho_x)
    logalphak_xx = logalpha + logk_xx
    log1alpha = jnp.log1p(-jnp.exp(logalpha))
    logalpha_x = (logalphak_xx) - (jnp.logaddexp(log1alpha,logalphak_xx)) #alpha*k_xx /(1-alpha + alpha*k_xx)
    #clip for numerical stability to prevent NaNs
    eps = 1e-4 #1e-6 causes optimization to fail
    logalpha_x = jnp.clip(logalpha_x,jnp.log(eps),jnp.log(1-eps))

    logpmf_yn= mvcc.update_copula(logpmf_yn,log_vn,y_new,logalpha_x,rho)

    #Compute pdiff
    pdiff = index_update(pdiff,index[i,:],jnp.abs(jnp.exp(logpmf_yn[:,0]) - jnp.exp(logpmf_init[:,0])))

    #Update pmf_ytest
    #compute x rhos/alphas
    logalpha = jnp.log(2.- (1/(i+1)))-jnp.log(i+2)
    logk_xx = mvcc.calc_logkxx(x_test,x_new,rho_x)
    logalphak_xx = logalpha + logk_xx
    log1alpha = jnp.log1p(-jnp.exp(logalpha))
    logalpha_x = (logalphak_xx) - (jnp.logaddexp(log1alpha,logalphak_xx)) #alpha*k_xx /(1-alpha + alpha*k_xx)
    #clip for numerical stability to prevent NaNs
    eps = 1e-4 #1e-6 causes optimization to fail
    logalpha_x = jnp.clip(logalpha_x,jnp.log(eps),jnp.log(1-eps))

    #Update pmf_ytest
    logpmf_ytest= mvcc.update_copula(logpmf_ytest,log_vn,y_new,logalpha_x,rho)

    carry = logpmf_ytest,logpmf_yn,y_samp,pdiff,y,x,x_test,rho,rho_x,ind_new,vT,logpmf_init
    return carry,i

@jit
def update_pn_scan_forward(carry,rng):
    return scan(update_pn_forward,carry,rng)

#Forward sample x_{n+1:N} and y_{n+1:N}
@partial(jit,static_argnums = (8))
def forward_sample_y_samp(key,logpmf_ytest,logpmf_yn,y,x,x_test,rho,rho_x,T):
    n = jnp.shape(y)[0]

    #generate uniform random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (T,1))
    vT = jnp.append(jnp.zeros(n),a_rand) #uniform rv for sampling 
    y_samp = jnp.concatenate((y.reshape(-1,1),jnp.zeros((T,1))),axis = 0) #remember y

    #Draw random x_samp from BB
    key, subkey = random.split(key) #split key
    n = jnp.shape(x)[0]
    w = random.dirichlet(subkey, jnp.ones(n)) #single set of dirichlet weights
    key, subkey = random.split(key) #split key
    ind_new = random.choice(key,a = jnp.arange(n),p = w,shape = (1,T))[0]
    x_samp = jnp.concatenate((x,x[ind_new]),axis = 0)

    # #Draw random x_samp from KDE (this is experimental, may delete?)
    # key, subkey = random.split(key) #split key
    # n = jnp.shape(x)[0]
    # w = random.dirichlet(subkey, jnp.ones(n)) #single set of dirichlet weights
    # key, subkey = random.split(key) #split key
    # ind_new = random.choice(key,a = jnp.arange(n),p = w,shape = (1,T))[0]
    # x_samp = jnp.concatenate((x,x[ind_new]+random.normal(key,shape = (1,T,d))),axis = 0)

    #Track changes
    pdiff = jnp.zeros((n+T,n))

    #run forward loop
    carry = logpmf_ytest,logpmf_yn,y_samp,pdiff,y,x,x_test,rho,rho_x,ind_new,vT,logpmf_yn
    rng = jnp.arange(n,n+T)
    carry,rng = update_pn_scan_forward(carry,rng)
    logpmf_ytest,logpmf_yn,y_samp,pdiff,*_ = carry
    return logpmf_ytest,logpmf_yn, y_samp,x_samp,pdiff

forward_sample_y_samp_B = jit(vmap(forward_sample_y_samp,(0,None,None,None,None,None,None,None,None)),static_argnums = (8))


#### CONVERGENCE CHECKS FOR PR (Not implemented yet) ####
def pr_1step_conv_classification(i,inputs):  #t = n+i
    logpmf_ytest,logpmf_ytest_init,vT,x,y_test,x_test,rho,rho_x = inputs

    #update pdf/cdf
    y_new = y[i]
    x_new = x[i]
    logalpha = jnp.log(2- (1/(n+i+1)))-jnp.log(n+i+2)

    #compute x rhos/alphas
    logk_xx = mvcc.calc_logkxx(x_test,x_new,rho_x)
    logalphak_xx = logalpha + logk_xx
    log1alpha = jnp.log1p(-jnp.exp(logalpha))
    logalpha_x = (logalphak_xx) - (jnp.logaddexp(log1alpha,logalphak_xx))


    logpmf_ytest= update_copula_single(logpmf_ytest,y_test,y_new,logalpha_x,rho)

    #conditional density
    pdiff = index_update(pdiff,i,jnp.mean(jnp.abs(jnp.exp(logpmf_ytest)- jnp.exp(logpmf_ytest_init)))) #mean density diff from initial

    outputs =  logcdf_conditionals_new,logpdf_joints_new,x,x_test,rho,rho_x,n,a_rand,\
                logcdf_conditionals_init,logpdf_joints_init,pdiff,cdiff
    return outputs

#loop update for pdf/cdf for xn, starting with average p_n
@partial(jit,static_argnums = (7,8))
def pr_loop_conv_classification(key,logcdf_conditionals,logpdf_joints,x,x_test,rho,rho_x,n,T):
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
    outputs = fori_loop(0,T,pr_1step_conv_classification,inputs)
    logcdf_conditionals,logpdf_joints,x_samp,x_test,rho,rho_x,n,a_rand,logcdf_conditionals,logpdf_joints,pdiff,cdiff = outputs

    return logcdf_conditionals,logpdf_joints,pdiff,cdiff
##Multiple samples for convergence checks
pr_loop_conv_classification_B =jit(vmap(pr_loop_conv_classification,(0,None,None,None,None,None,None,None,None)),static_argnums=(7,8))
