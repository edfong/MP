import jax.numpy as jnp
import numpy as np

#delete
import time

from jax import grad, jit, vmap,jacfwd,jacrev,random,remat,value_and_grad
from jax.scipy.special import ndtri,erfc,logsumexp
from jax.scipy.stats import norm
from jax import random
from jax.lax import fori_loop,scan
#from jax.config import config; config.update("jax_enable_x64", True)
from jax.ops import index, index_add, index_update
import scipy as osp
from functools import partial
from scipy.optimize import minimize,root

from tqdm import tqdm_notebook

from . import mv_copula_density as mvcd
from . import sample_mv_copula_density as samp_mvcd
from . import mv_copula_regression as mvcr
from .utils.BFGS import minimize_BFGS
from .utils.bivariate_copula import ndtri_,norm_logcdf, norm_copula_logdistribution_logdensity

#Compute sequence alpha(x,x_i) for conditional method for single x_plot
@jit
def compute_logalpha_x_single(x_plot,xn,rho_x):
    d = jnp.shape(xn)[1]
    n = jnp.shape(xn)[0]

    #compute alpha_n
    n_range = jnp.arange(n)+1
    logalpha_seq = (jnp.log(2.- (1/(n_range))) - jnp.log(n_range +1))
    log1alpha_seq = jnp.log1p(-jnp.exp(logalpha_seq))

    #compute cop_dens
    logk_xx= mvcr.calc_logkxx_test(x_plot.reshape(1,d),xn.reshape(-1,d),rho_x)[:,0]

    #compute alpha_x
    logalpha_x = (logalpha_seq + logk_xx) -jnp.logaddexp(log1alpha_seq,(logalpha_seq+logk_xx))
    return logalpha_x

### Predicting on Test ###
@jit
def update_ptest_single(carry,i):
    vn,logcdf_conditionals_ytest,logpdf_joints_ytest,logalpha_x,rho,rho_x = carry

    u = jnp.exp(logcdf_conditionals_ytest)
    v = vn[i]

    logcdf_conditionals_ytest,logpdf_joints_ytest= mvcd.update_copula_single(logcdf_conditionals_ytest,logpdf_joints_ytest,u,v,logalpha_x[i],rho)

    carry = vn,logcdf_conditionals_ytest,logpdf_joints_ytest,logalpha_x,rho,rho_x
    return carry,i

@jit
def update_ptest_single_scan(carry,rng):
    return scan(update_ptest_single,carry,rng)

#single x_test, single y_test, single perm
@jit
def update_ptest_single_loop(vn,rho,rho_x,y_test,logalpha_x):
    n = jnp.shape(vn)[0]
    logcdf_conditionals_ytest, logpdf_joints_ytest= mvcd.init_marginals_single(jnp.array([y_test]))
    carry = vn,logcdf_conditionals_ytest,logpdf_joints_ytest,logalpha_x,rho,rho_x
    rng = jnp.arange(n)
    carry,rng = update_ptest_single_scan(carry,rng)
    vn,logcdf_conditionals_ytest,logpdf_joints_ytest,x,logalpha_x,rho_x = carry

    return logcdf_conditionals_ytest,logpdf_joints_ytest


#single  x_test, multiple y_test, single perm
@jit
def update_ptest_singlex_loop(vn,rho,rho_x,x,y_test,x_test):
    #compute logalpha_x
    logalpha_x = compute_logalpha_x_single(x_test,x,rho_x)
    temp = jit(vmap(update_ptest_single_loop,(None,None,None,0,None)))
    return temp(vn,rho,rho_x,y_test,logalpha_x)

#multiple  x_test, multiple y_test, single perm
update_ptest_loop = jit(vmap(update_ptest_singlex_loop,(None,None,None,None,0,0)))

#Average p(y) over permutations
@jit
def update_ptest_loop_perm_av(vn_perm,rho,rho_x,x_perm,y_test,x_test):
    n_perm = jnp.shape(vn_perm)[0]
    temp = jit(vmap(update_ptest_loop,(0,None,None,0,None,None)))
    logcdf_conditionals, logpdf_joints = temp(vn_perm,rho,rho_x,x_perm,y_test,x_test)
    logcdf_conditionals = logsumexp(logcdf_conditionals,axis = 0) - jnp.log(n_perm)
    logpdf_joints = logsumexp(logpdf_joints,axis = 0) - jnp.log(n_perm)
    return logcdf_conditionals,logpdf_joints



#Predict on test data using average permutations
def predict_copula_cregression(copula_cregression_obj,y_test,x_test):
    #code loop for now, can speed up to use indices
    n_perm = np.shape(copula_cregression_obj.x_perm)[0]
    n = np.shape(copula_cregression_obj.x_perm)[1]
    n_test = np.shape(x_test)[0]

    print('Predicting...')
    start = time.time()
    logcdf_conditionals,logpdf_joints = update_ptest_loop_perm_av(copula_cregression_obj.vn_perm,copula_cregression_obj.rho_opt\
                                                                         ,copula_cregression_obj.rho_x_opt,copula_cregression_obj.x_perm, y_test,x_test)
    logcdf_conditionals = logcdf_conditionals.block_until_ready() #for accurate timing
    end = time.time()
    print('Prediction time: {}s'.format(round(end-start, 3)))
    return logcdf_conditionals,logpdf_joints