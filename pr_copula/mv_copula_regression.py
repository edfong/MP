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

## Joint method ###
#condit marginal likelihood
@jit
def negpreq_jconditloglik_perm(hyperparam,z_perm):
    rho = 1/(1+jnp.exp(hyperparam)) #force 0 <rho<1

    n = jnp.shape(z_perm)[1]
    d = jnp.shape(z_perm)[2]

    #For non autoregressive models, set rho_x -> jnp.inf?
    vn,logcdf_conditionals_yn,logpdf_joints_yn,preq_loglik = mvcd.update_pn_loop_perm(rho,z_perm)

    #Average over permutations
    preq_conditloglik = preq_loglik[:,:,-1]-preq_loglik[:,:,-2]
    preq_conditloglik = jnp.mean(preq_conditloglik,axis = 0)

    #Marg
    preq_conditloglik = jnp.sum(preq_conditloglik) # look at conditional pdf
    return -preq_conditloglik/n


#Derivative
grad_jcll_perm = jit(grad(negpreq_jconditloglik_perm))
fun_grad_jcll_perm = jit(value_and_grad(negpreq_jconditloglik_perm))

#Functions for scipy (convert to numpy array)
def fun_jcll_perm_sp(hyperparam,z):
    return np.array(negpreq_jconditloglik_perm(hyperparam,z))
def grad_jcll_perm_sp(hyperparam,z):
    return np.array(grad_jcll_perm(hyperparam,z)) ####

def fun_grad_jcll_perm_sp(hyperparam,z):
    value,grad = fun_grad_jcll_perm(hyperparam,z)
    return (np.array(value),np.array(grad))
### ###

## Conditional method ###
#vmap over rho/alpha as well
update_copula = jit(vmap(mvcd.update_copula_single,(0,0,0,None,0,None))) 
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
    vn,logcdf_conditionals_yn,logpdf_joints_yn,preq_loglik,x,rho,rho_x = carry

    #Compute new x
    x_new = x[i]
    logalpha = jnp.log(2.- (1/(i+1)))-jnp.log(i+2)

    #compute x rhos/alphas
    logk_xx = calc_logkxx(x,x_new,rho_x)
    logalphak_xx = logalpha + logk_xx
    log1alpha = jnp.log1p(-jnp.exp(logalpha))
    logalpha_x = (logalphak_xx) - (jnp.logaddexp(log1alpha,logalphak_xx)) #alpha*k_xx /(1-alpha + alpha*k_xx)

    #clip for numerical stability to prevent NaNs
    eps = 1e-5 #1e-6 causes optimization to fail
    logalpha_x = jnp.clip(logalpha_x,jnp.log(eps),jnp.log(1-eps))

    u = jnp.exp(logcdf_conditionals_yn)
    v = jnp.exp(logcdf_conditionals_yn[i])

    vn = index_update(vn,i,v) #remember history of vn
 
    preq_loglik = index_update(preq_loglik,i,logpdf_joints_yn[i,-1])
    logcdf_conditionals_yn,logpdf_joints_yn= update_copula(logcdf_conditionals_yn,logpdf_joints_yn,u,v,logalpha_x,rho)
    carry = vn,logcdf_conditionals_yn,logpdf_joints_yn,preq_loglik,x,rho,rho_x
    return carry,i

@jit
def update_pn_scan(carry,rng):
    return scan(update_pn,carry,rng)

#loop update for pdf/cdf for xn
@jit
def update_pn_loop(rho,rho_x,y,x):
    n = jnp.shape(y)[0]
    preq_loglik = jnp.zeros((n,1)) #prequential joint loglik for y | x
    vn = jnp.zeros((n,1)) #conditional cdf history of yn, no need to differentiate wrt

    #initialize cdf/pdf
    logcdf_conditionals_yn, logpdf_joints_yn= mvcd.init_marginals(y)

    carry = vn,logcdf_conditionals_yn,logpdf_joints_yn,preq_loglik,x,rho,rho_x
    rng = jnp.arange(n)
    carry,rng = update_pn_scan(carry,rng)

    vn,logcdf_conditionals_yn,logpdf_joints_yn,preq_loglik,*_ = carry

    return vn,logcdf_conditionals_yn,logpdf_joints_yn,preq_loglik
update_pn_loop_perm = jit(vmap(update_pn_loop,(None,None,0,0)))


##OPTIMIZING MARGINAL LIKELIHOOD
#joint marginal likelihood
@jit
def negpreq_cconditloglik_perm(hyperparam,y_perm,x_perm):
    rho = 1/(1+jnp.exp(hyperparam[0])) #force 0 <rho<1
    rho_x = 1/(1+jnp.exp(hyperparam[1:]))#force 0<rho_x < 1

    n = jnp.shape(y_perm)[1]

    #For non autoregressive models, set rho_x -> jnp.inf?
    vn,logcdf_conditionals_yn,logpdf_joints_yn,preq_loglik = update_pn_loop_perm(rho,rho_x,y_perm,x_perm)

    #Average over permutations
    preq_loglik = jnp.mean(preq_loglik,axis = 0)

    #Marg
    preq_jointloglik = jnp.sum(preq_loglik[:,-1]) #only look at joint pdf
    return -preq_jointloglik/n
    
#Derivative
grad_ccll_perm = jit(grad(negpreq_cconditloglik_perm))
fun_grad_ccll_perm = jit(value_and_grad(negpreq_cconditloglik_perm))

####
#Functions for scipy (convert to numpy array)
def fun_ccll_perm_sp(hyperparam,y_perm,x_perm):
    return np.array(negpreq_cconditloglik_perm(hyperparam,y_perm,x_perm))
def grad_ccll_perm_sp(hyperparam,y_perm,x_perm):
    return np.array(grad_ccll_perm(hyperparam,y_perm,x_perm)) 

def fun_grad_ccll_perm_sp(hyperparam,y_perm,x_perm):
    value,grad = fun_grad_ccll_perm(hyperparam,y_perm,x_perm)
    return (np.array(value),np.array(grad))
####

### Predicting on Test ###
@jit
def update_ptest_single(carry,i):
    vn,logcdf_conditionals_ytest,logpdf_joints_ytest,x,x_test,rho,rho_x = carry

    x_new = x[i]
    logalpha = jnp.log(2.- (1/(i+1)))-jnp.log(i+2)

    #compute x rhos/alphas
    logk_xx = calc_logkxx_single(x_test,x_new,rho_x)
    
    logalphak_xx = logalpha + logk_xx
    log1alpha = jnp.log1p(-jnp.exp(logalpha))
    logalpha_x = (logalphak_xx) - (jnp.logaddexp(log1alpha,logalphak_xx))

    #clip for numerical stability to prevent NaNs
    eps = 1e-5 #1e-6 causes optimization to fail
    logalpha_x = jnp.clip(logalpha_x,jnp.log(eps),jnp.log(1-eps))

    u = jnp.exp(logcdf_conditionals_ytest)
    v = vn[i]

    logcdf_conditionals_ytest,logpdf_joints_ytest= mvcd.update_copula_single(logcdf_conditionals_ytest,logpdf_joints_ytest,u,v,logalpha_x,rho)

    carry = vn,logcdf_conditionals_ytest,logpdf_joints_ytest,x,x_test,rho,rho_x
    return carry,i

@jit
def update_ptest_single_scan(carry,rng):
    return scan(update_ptest_single,carry,rng)

@jit
def update_ptest_single_loop(vn,rho,rho_x,x,y_test,x_test):
    n = jnp.shape(vn)[0]

    logcdf_conditionals_ytest, logpdf_joints_ytest= mvcd.init_marginals_single(y_test)

    carry = vn,logcdf_conditionals_ytest,logpdf_joints_ytest,x,x_test,rho,rho_x
    rng = jnp.arange(n)
    carry,rng = update_ptest_single_scan(carry,rng)
    vn,logcdf_conditionals_ytest,logpdf_joints_ytest,x,x_test,rho,rho_x = carry

    return logcdf_conditionals_ytest,logpdf_joints_ytest

update_ptest_single_loop_perm = jit(vmap(update_ptest_single_loop,(0,None,None,0,None,None))) #vmap over vn_perm

#Average p(y) over permutations
@jit
def update_ptest_single_loop_perm_av(vn_perm,rho,rho_x,x_perm,y_test,x_test):
    n_perm = jnp.shape(vn_perm)[0]
    logcdf_conditionals, logpdf_joints = update_ptest_single_loop_perm(vn_perm,rho,rho_x,x_perm,y_test,x_test)
    logcdf_conditionals = logsumexp(logcdf_conditionals,axis = 0) - jnp.log(n_perm)
    logpdf_joints = logsumexp(logpdf_joints,axis = 0) - jnp.log(n_perm)
    return logcdf_conditionals,logpdf_joints

#No jit so can adapt to number of test points
update_ptest_loop_perm_av = jit(vmap(update_ptest_single_loop_perm_av,(None,None,None,None,0,0)))




