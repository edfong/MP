###JAX implementation of BFGS with armijo back-tracking line search
import jax.numpy as jnp
from jax.experimental.optimizers import optimizer
from jax.lax import while_loop,cond
from jax import grad,value_and_grad, jit
from functools import partial

### Optimizer loop and termination
@partial(jit,static_argnums = (0))
def minimize_BFGS(fun,x0,g_tol = 1e-5,n_iter_max = 100,delta_B_init = 1.,c1_ls = 1e-4,c2_ls = 0.9,n_max_ls = 10, tau_ls  = 0.5): 
    """Minimize function with BFGS.
    Args:
    fun: function to minimize, which only takes x as input
    x0: ndarray: initial solution 
    
    #Optimizer hyperparams
    g_tol: positive scalar for value of norm of gradient before termination
    n_iter_max: maximum number of BFGS steps before termination
    delta_B_init: positive scalar for expected norm of initial step to scale initial Hessian inverse

    #Line search hyperparams
    c1_ls: positive scalar in Armijo sufficient decrease condition
    c2_ls: positive scalar in weak Wolfe curvature condition
    n_max_ls: maximum number of back-tracking steps 
    tau_ls: positive scalar less than 1 for scaling factor step size for back-tracking

    Returns:
    x_opt: Optimized x
    loss: Minimized loss value
    n_iter: Number of iterations at termination
    norm_g: Norm of gradient at termination
    """

    #function wrappers
    value_and_grad_fun = jit(value_and_grad(fun))

    @jit #wrapper around step to allow for termination checks
    def step(carry): 
        i,loss,g,opt_state = carry
        params = get_params(opt_state)
        loss,g = value_and_grad_fun(params)
        i=i+1
        carry = i,loss,g,opt_update(i, g, opt_state)
        return carry

    @jit #termination condition on gradient norm or number of iterations
    def converged(carry): 
        i,loss,g,opt_state = carry
        norm_g = jnp.sqrt(jnp.sum(g**2))
        return jnp.logical_and(norm_g>g_tol,i<n_iter_max) 
    
    #check dimensions
    d = jnp.shape(x0)[0]

    #initialize optimzer
    opt_init, opt_update, get_params = BFGS(fun,delta_B_init = delta_B_init,c1_ls = c1_ls,c2_ls = c2_ls,n_max_ls = n_max_ls, tau_ls  = tau_ls)
    opt_state = opt_init(x0)

    #run optimizer until termination
    carry = 0,1.,jnp.ones(d), opt_state
    carry = while_loop(converged,step,carry)

    #extract values from carry
    n_iter,loss,g,opt_state = carry
    norm_g = jnp.sqrt(jnp.sum(g**2))
    x_opt = get_params(opt_state)

    return x_opt,loss,n_iter,norm_g
###

### Optimizer triple
@optimizer
def BFGS(fun,delta_B_init = 1.,c1_ls = 1e-4,c2_ls = 0.9,n_max_ls = 10, tau_ls  = 0.5):
    """Construct optimizer triple for BFGS.
    Args:
    fun: function to minimize, which only takes x as input
    delta_B_init: expected norm of initial step to scale initial Hessian inverse

    #Armijo backtracking line search hyperparams
    c1_ls: positive scalar in Armijo sufficient decrease condition
    c2_ls: positive scalar in weak Wolfe curvature condition
    n_max_ls: maximum number of back-tracking steps 
    tau_ls: positive scalar less than 1 for scaling factor step size for back-tracking

    Returns:
    An (init_fun, update_fun, get_params) triple.
    """

    #define gradient (user does not need to pass)
    grad_fun= jit(grad(fun))

    #update BFGS matrix
    def update_B_inv(input_update): 
        B_inv,sk,yk = input_update
        sk_t_yk = jnp.dot(sk,yk)
        yk_sk_t = jnp.outer(yk,sk)
        yk_t_B_inv_yk = jnp.dot(yk.transpose(),jnp.dot(B_inv,yk))
        
        B_inv = B_inv + ((sk_t_yk+yk_t_B_inv_yk)*jnp.outer(sk,sk))/(sk_t_yk**2)\
                -(jnp.dot(B_inv,yk_sk_t) + jnp.dot(yk_sk_t.transpose(),B_inv))/sk_t_yk
        return B_inv

    #Line search wrappers
    armijo_linesearch_=  partial(armijo_linesearch,fun,c1_ls,n_max_ls,tau_ls) #wrapper around line search
    curvature_check_ = partial(curvature_check,grad_fun,c2_ls) #wrapper around weak Wolfe curvature check

    def init(x0):
        d = jnp.shape(x0)[0]
        g = grad_fun(x0)
        B0_inv = delta_B_init*jnp.eye(d)/(jnp.sqrt(jnp.sum(g**2))) #expect norm to be about 1?
        return x0, B0_inv
    def update(i, g, state):
        x, B_inv = state

        #Step direction
        step_dir = -jnp.dot(B_inv,g)

        #Back-tracking line search for step_size with Armijo termination
        step_size = 1.
        n_iter_ls = 0
        f = fun(x)
        carry_ls = x,step_size,step_dir,n_iter_ls,f,g
        step_size = armijo_linesearch_(carry_ls)
        
        #update x
        sk = step_size * step_dir
        x_new = x + sk
        
        #check weak Wolfe curvature conditions 
        carry_ls = x,step_size,step_dir,n_iter_ls,f,g
        curvature,g2 = curvature_check_(carry_ls)
        
        #update BFGS matrix if weak Wolfe conditions satisfied
        yk = g2 - g
        B_inv = cond(curvature,(B_inv,sk,yk),update_B_inv,B_inv,lambda x: x)
        return x_new, B_inv

    def get_params(state):
        x, _ = state
        return x
    return init, update, get_params
###

### Armijo backtracking line search
def armijo_linesearch(fun,c1_ls,n_max_ls,tau_ls,carry): 
    armijo_check_ = partial(armijo_check,fun,c1_ls,n_max_ls) #wrapper for sufficient decrease check
    armijo_step_ = partial(armijo_step,tau_ls) #wrapper for sufficient decrease check

    x,step_size,step_dir,n_iter_ls,f,g = carry
    carry = while_loop(armijo_check_,armijo_step_,carry) #while loop for backtracking
    _,step_size,*_= carry
    return step_size

#decrease step size and increment counter
def armijo_step(tau_ls,carry): 
    x,step_size,step_dir,n_iter_ls,f,g = carry
    n_iter_ls = n_iter_ls +1
    step_size = step_size*tau_ls
    carry = x,step_size,step_dir,n_iter_ls,f,g
    return carry

#check if armijo sufficient decrease condition is satisfied
def armijo_check(fun,c1_ls,n_max_ls,carry):
    x,step_size,step_dir,n_iter_ls,f,g = carry
    f2 = fun(x+step_size*step_dir)
    k = -c1_ls*jnp.dot(g.transpose(),step_dir)
    return jnp.logical_and(f-f2<k,n_iter_ls <n_max_ls)

#check if (weak) Wolfe curvature condition is satisfied
def curvature_check(grad_fun,c2_ls,carry):
    x,step_size,step_dir,n_iter_ls,f,g = carry
    pg1 = jnp.dot(g.transpose(),step_dir)
    g2 = grad_fun(x+step_size*step_dir)
    pg2 = jnp.dot(g2.transpose(),step_dir)
    return -0.9*pg1>-pg2,g2
### 