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

from . import mv_copula_density as mvcd
from .utils.BFGS import minimize_BFGS

## Conditional method ###
#Initialize joint/conditional cdf/pmfs to 0.5
def init_marginals_single(y):
    n = jnp.shape(y)[0]

    ##discrete
    p_yplot = 0.5
    logpmf_init_marginals = jnp.array([jnp.log(p_yplot)])

    #clip outliers
    eps = 1e-6
    logpmf_init_marginals =jnp.clip(logpmf_init_marginals, jnp.log(eps),jnp.log(1-eps))
    ##
    return  logpmf_init_marginals
init_marginals = jit(vmap(init_marginals_single,(0)))

# Works only with p(y=1 | x) 
#Bernoulli 'copula'
def update_copula_single(logpmf1,log_v,y_new,logalpha,rho):
    eps = 5e-5
    logpmf1 = jnp.clip(logpmf1,jnp.log(eps),jnp.log(1-eps)) #clip u before passing to bicop
    log_v = jnp.clip(log_v,jnp.log(eps),jnp.log(1-eps)) #clip u before passing to bicop

    log1alpha = jnp.log1p(jnp.clip(-jnp.exp(logalpha),-1+eps,jnp.inf))
    log1_v = jnp.log1p(jnp.clip(-jnp.exp(log_v),-1+eps,jnp.inf))

    min_logu1v1 = jnp.min(jnp.array([logpmf1,log_v]))

    ##Bernoulli update
    frac = y_new*jnp.exp(min_logu1v1 - logpmf1 -log_v) + (1-y_new)*(1/jnp.exp(log1_v) - jnp.exp(min_logu1v1 - logpmf1 - log1_v)) #make this more accurate?
    kyy_ = 1-rho + rho*frac
    kyy_ = jnp.clip(kyy_,eps,jnp.inf)

    logkyy_ = jnp.log(kyy_)
    logpmf1_new =  jnp.logaddexp(log1alpha, (logalpha+logkyy_))+logpmf1

    return logpmf1_new

#vmap over rho/alpha as well
update_copula = jit(vmap(update_copula_single,(0,None,None,0,None))) 

@jit
def calc_logkxx_single(x,x_new,rho_x):
    logk_xx = -0.5*jnp.sum(jnp.log(1-rho_x**2)) -jnp.sum((0.5/(1-rho_x**2))*(((rho_x**2)*(x**2 + x_new**2) - 2*rho_x*x*x_new)))
    return logk_xx
calc_logkxx = jit(vmap(calc_logkxx_single,(0,None,None)))
calc_logkxx_test = jit(vmap(calc_logkxx,(None,0,None)))

@jit
def calc_rhokxx_single(x,x_new,rho_0,l_scale):
    rho_xx = rho_0*jnp.exp(-0.5*np.sum((x-x_new)**2/l_scale))
    return rho_xx
calc_rhoxx = jit(vmap(calc_rhokxx_single,(0,None,None)))
calc_rhokxx_test = jit(vmap(calc_rhoxx,(None,0,None)))

#overhead to calculate p_{1:n}(y_{1:n})
@jit
def update_pn(carry,i):
    log_vn,logpmf1_yn,preq_loglik,y,x,rho,rho_x = carry

    #Compute new x
    y_new = y[i]
    x_new = x[i]
    logalpha = jnp.log(2.- (1/(i+1)))-jnp.log(i+2)

    #compute x rhos/alphas
    eps = 5e-5
    logk_xx = calc_logkxx(x,x_new,rho_x)
    logalphak_xx = logalpha + logk_xx
    log1alpha = jnp.log1p(jnp.clip(-jnp.exp(logalpha),-1 + eps,jnp.inf))
    logalpha_x = (logalphak_xx) - (jnp.logaddexp(log1alpha,logalphak_xx)) #alpha*k_xx /(1-alpha + alpha*k_xx)

    #clip for numerical stability to prevent NaNs
    logalpha_x = jnp.clip(logalpha_x,jnp.log(eps),jnp.log(1-eps))
 
    #add p1 or (1-p1) depending on what y_new is
    temp = y_new*logpmf1_yn[i,-1] +(1-y_new)*jnp.log1p(jnp.clip(-jnp.exp(logpmf1_yn[i,-1]),-1+eps,jnp.inf))
    preq_loglik = index_update(preq_loglik,i,temp)

    #
    log_v = logpmf1_yn[i]
    log_vn = index_update(log_vn,i,log_v)

    logpmf1_yn= update_copula(logpmf1_yn,log_v,y_new,logalpha_x,rho)
    carry = log_vn,logpmf1_yn,preq_loglik,y,x,rho,rho_x
    return carry,i

@jit
def update_pn_scan(carry,rng):
    return scan(update_pn,carry,rng)

#loop update for pdf/cdf for xn
@jit
def update_pn_loop(rho,rho_x,y,x):
    n = jnp.shape(y)[0]
    preq_loglik = jnp.zeros((n,1)) #p_n(y_{n+1}=1 | x_{n+1})

    #initialize cdf/pdf
    logpmf1_yn= init_marginals(y)
    log_vn = jnp.zeros((n,1))

    carry = log_vn,logpmf1_yn,preq_loglik,y,x,rho,rho_x
    rng = jnp.arange(n)
    carry,rng = update_pn_scan(carry,rng)

    log_vn,logpmf1_yn,preq_loglik,*_ = carry

    return log_vn,logpmf1_yn,preq_loglik
update_pn_loop_perm = jit(vmap(update_pn_loop,(None,None,0,0)))


##OPTIMIZING MARGINAL LIKELIHOOD
#joint marginal likelihood
@jit
def negpreq_cconditloglik_perm(hyperparam,y_perm,x_perm):
    rho = 1/(1+jnp.exp(hyperparam[0])) #force 0 <rho<1
    rho_x = 1/(1+jnp.exp(hyperparam[1:]))#force 0<rho_x < 1

    n = jnp.shape(y_perm)[1]

    #For non autoregressive models, set rho_x -> jnp.inf?
    _,_,preq_loglik = update_pn_loop_perm(rho,rho_x,y_perm,x_perm)

    #Average over permutations
    preq_loglik = jnp.mean(preq_loglik,axis = 0)

    #Marg
    preq_jointloglik = jnp.sum(preq_loglik[:,-1]) #only look at joint pdf
    return -preq_jointloglik
    
#Derivative
fun_grad_ccll_perm = jit(value_and_grad(negpreq_cconditloglik_perm))
grad_ccll_perm = jit(grad(negpreq_cconditloglik_perm))


####
#Functions for scipy (convert to numpy array)
def fun_ccll_perm_sp(hyperparam,y_perm,x_perm):
    return np.array(negpreq_cconditloglik_perm(hyperparam,y_perm,x_perm))
def grad_ccll_perm_sp(hyperparam,y_perm,x_perm):
    return np.array(grad_ccll_perm(hyperparam,y_perm,x_perm)) ####

def fun_grad_ccll_perm_sp(hyperparam,y_perm,x_perm):
    value,grad = fun_grad_ccll_perm(hyperparam,y_perm,x_perm)
    return (np.array(value),np.array(grad))
####

### Predicting on Test ###
@jit
def update_ptest_single(carry,i):
    log_vn,logpmf_ytest,y,x,x_test,rho,rho_x = carry

    y_new = y[i]
    x_new = x[i]
    logalpha = jnp.log(2.- (1/(i+1)))-jnp.log(i+2)

    #compute x rhos/alphas
    eps = 5e-5 #1e-6 causes optimization to fail
    logk_xx = calc_logkxx_single(x_test,x_new,rho_x)
    logalphak_xx = logalpha + logk_xx
    log1alpha = jnp.log1p(jnp.clip(-jnp.exp(logalpha),-1+eps,jnp.inf))
    logalpha_x = (logalphak_xx) - (jnp.logaddexp(log1alpha,logalphak_xx))

    #clip for numerical stability to prevent NaNs
    logalpha_x = jnp.clip(logalpha_x,jnp.log(eps),jnp.log(1-eps))

    logpmf_ytest= update_copula_single(logpmf_ytest,log_vn[i],y_new,logalpha_x,rho)

    carry = log_vn,logpmf_ytest,y,x,x_test,rho,rho_x
    return carry,i

@jit
def update_ptest_single_scan(carry,rng):
    return scan(update_ptest_single,carry,rng)

@jit
def update_ptest_single_loop(log_vn,rho,rho_x,y,x,x_test):
    n = jnp.shape(y)[0]
    n_test = jnp.shape(x_test)[0]

    logpmf1_ytest= init_marginals_single(np.zeros((n_test,1)))

    carry = log_vn,logpmf1_ytest,y,x,x_test,rho,rho_x
    rng = jnp.arange(n)
    carry,rng = update_ptest_single_scan(carry,rng)
    log_vn,logpmf_ytest,y,x,x_test,rho,rho_x = carry

    return logpmf_ytest
update_ptest_single_loop_perm = jit(vmap(update_ptest_single_loop,(0,None,None,0,0,None))) #vmap over vn_perm

#Average p(y) over permutations
@jit
def update_ptest_single_loop_perm_av(log_vn_perm,rho,rho_x,y_perm,x_perm,x_test):
    n_perm = np.shape(y_perm)[0]
    logpmf_ytest= update_ptest_single_loop_perm(log_vn_perm,rho,rho_x,y_perm,x_perm,x_test)
    logpmf_ytest = logsumexp(logpmf_ytest,axis = 0) - jnp.log(n_perm)
    return logpmf_ytest

#No jit so can adapt to number of test points
update_ptest_loop_perm_av = jit(vmap(update_ptest_single_loop_perm_av,(None,None,None,None,None,0)))




