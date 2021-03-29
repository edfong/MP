###JAX implementation of ADAM
import jax.numpy as jnp
from jax.experimental.optimizers import adam
from jax.lax import while_loop,cond
from jax import grad,value_and_grad, jit
from functools import partial

### Optimizer loop and termination
@partial(jit,static_argnums = (0))
def minimize_ADAM(fun,x0,data,step_size = 0.01,f_tol = 1e-5,n_iter_max = 1000): 
    """Minimize function with ADAM.
    Args:
    fun: function to minimize, which takes (x,data) as input
    x0: ndarray: initial solution 
    data: ndarray: full data to subsample
    
    #Optimizer hyperparams
    step_size: positive scalar for step-size to pass into minimize_adam
    f_tol: positive scalar for value of norm of f change before termination
    n_iter_max: maximum number of sgd steps before termination

    Returns:
    x_opt: Optimized x
    loss: Minimized loss value
    n_iter: Number of iterations at termination
    delta_f: Value of change of f at termination
    """

    #function wrappers
    value_and_grad_fun = jit(value_and_grad(fun))

    @jit #wrapper around step to allow for termination checks
    def step(carry): 
        i,loss,g,opt_state,key,n = carry
        key,*subkey = random.split(key)
        ind = random.shuffle(key,jnp.arange(n))
        params = get_params(opt_state)
        loss,g = value_and_grad_fun(params,data[ind])
        i=i+1
        carry = i,loss,g,opt_update(i, g, opt_state)
        return carry

    @jit #termination condition on gradient norm or number of iterations
    def converged(carry): 
        i,loss,g,opt_state,key,n = carry
        delta_f = 1
        return jnp.logical_and(norm_g>g_tol,i<n_iter_max) 
    
    #check dimensions and initialize
    d = jnp.shape(x0)[0]
    key = random.PRNGKey(0)
    n = jnp.shape(data)[0]

    #initialize optimzer
    opt_init, opt_update, get_params = adam(step_size = step_size)
    opt_state = opt_init(x0)

    #run optimizer until termination
    carry = 0,1.,jnp.ones(d), opt_state,key,n
    carry = while_loop(converged,step,carry)

    #extract values from carry
    n_iter,loss,g,opt_state,_,_ = carry

    x_opt = get_params(opt_state)

    return x_opt,loss,n_iter,delta_f
###
