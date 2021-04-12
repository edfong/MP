import jax.numpy as jnp
from jax import custom_jvp,jit
from jax.scipy.stats import norm,t
from jax.scipy.special import ndtri

### Functions for normal copula ###
#Custom derivatives for Phi^{-1} to speed up autograd
@custom_jvp 
def ndtri_(u):
    return ndtri(u)
@ndtri_.defjvp
def f_jvp(primals, tangents):
    u, = primals
    u_dot, = tangents
    primal_out = ndtri_(u)
    tangent_out = (1/norm.pdf(primal_out))*u_dot
    return primal_out, tangent_out
ndtri_ = jit(ndtri_)

#Custom derivatives for logPhi to speed up autograd
@custom_jvp
def norm_logcdf(z):
    return norm.logcdf(z)
@norm_logcdf.defjvp
def f_jvp(primals, tangents):
    z, = primals
    z_dot, = tangents
    primal_out = norm_logcdf(z)
    tangent_out = jnp.exp(norm.logpdf(z)-primal_out)*z_dot
    return primal_out, tangent_out
norm_logcdf = jit(norm_logcdf)

#Calculate bivariate normal copula log H_uv and log c_uv
@jit
def norm_copula_logdistribution_logdensity(u,v,rho):
    #clip to prevent 0s and 1s in CDF, needed for numerical stability in high d.
    eps = 1e-6
    u = jnp.clip(u,eps,1-eps) 
    v = jnp.clip(v,eps,1-eps)
    
    #for reverse mode
    pu = ndtri_(u)
    pv = ndtri_(v)
    z = (pu - rho*pv)/jnp.sqrt(1- rho**2)

    logcop_dist = norm_logcdf(z)
    logcop_dist = jnp.clip(logcop_dist,jnp.log(eps),jnp.log(1-eps))
    logcop_dens = -0.5*jnp.log(1-rho**2) + (0.5/(1-rho**2))*(-(rho**2)*(pu**2 + pv**2)+ 2*rho*pu*pv)

    return logcop_dist,logcop_dens
### ###

## Functions for Student t copula (experimental) ##
#Custom derivatives for T^{-1} to speed up autograd
@custom_jvp 
def arctan_(u):
    return jnp.arctan(u)
@arctan_.defjvp
def f_jvp(primals, tangents):
    u, = primals
    u_dot, = tangents
    primal_out = arctan_    (u)
    tangent_out = (1/(1+ u**2))*u_dot
    return primal_out, tangent_out
arctan_ = jit(arctan_)

### Student-t copula, df = 1 ###
@jit
def t1_logcdf(x,loc = 0,scale = 1):
    z = (x-loc)/scale
    cdf = 0.5+(1/jnp.pi)*arctan_(z)
    return jnp.log(cdf)

@jit
def t1_logpdf(x,loc = 0,scale = 1):
    z = (x-loc)/scale
    logpdf = -jnp.log(jnp.pi*(1+z**2)*scale)
    return logpdf

@jit 
def t1_invcdf(u):
    z = jnp.tan(jnp.pi*(u-0.5))
    return z

@jit
def t2_logcdf(x,loc = 0,scale = 1):
    z = (x-loc)/scale
    cdf = 0.5+z/(2*jnp.sqrt(2)*jnp.sqrt(1+0.5*z**2))
    return jnp.log(cdf)

@jit #Calculate normal copula cdf and logpdf for f32
def t1_copula_logdistribution_logdensity(u,v,rho):
    #clip to prevent 0s and 1s in CDF, bigger clip than float64
    eps = 1e-4
    u = jnp.clip(u,eps,1-eps)
    v = jnp.clip(v,eps,1-eps)
    
    #for reverse mode
    pu = t1_invcdf(u)
    pv = t1_invcdf(v)
    mu = rho*pv
    sigma = jnp.sqrt((1-rho**2)*(1+pv**2)/2)
    z = (pu - mu)/sigma

    logcop_dist = t2_logcdf(z)
    logcop_dist = jnp.clip(logcop_dist,jnp.log(eps),jnp.log(1-eps))

    logcop_dens = -jnp.log(2*jnp.pi) - 0.5*jnp.log(1-rho**2) \
    - 1.5*jnp.log(1 + (1/(1-rho**2))*(pu**2 + pv**2 - 2*rho*pu*pv)) #numerator
    logcop_dens = logcop_dens - t1_logpdf(pu) - t1_logpdf(pv)

    return logcop_dist,logcop_dens
### ###

### Student-t copula, df = 2 ###
@jit
def t_logpdf(x,df = 2,loc = 0,scale = 1):
    return t.logpdf(x,df = df,loc = loc, scale = scale)

@jit
def t_logcdf(x,df = 2,loc = 0,scale = 1):
    z = (x-loc)/scale
    ind = jnp.where(z > 0, x = 1,y = 0)
    betainc_z = betainc(df/2,0.5,df/(df + z**2))
    cdf = ind*(1- 0.5*betainc_z) + (1-ind)*0.5*betainc_z
    return jnp.log(cdf)

@jit
def t3_logcdf(x,loc = 0,scale = 1):
    z = (x-loc)/scale
    cdf = 0.5+(1/jnp.pi)*( (1/jnp.sqrt(3))*z/(1+z**2/3) + arctan_(z/jnp.sqrt(3)))
    return jnp.log(cdf)

@jit
def t2_logpdf(x,loc = 0,scale = 1):
    z = (x-loc)/scale
    pdf = 1/(2*jnp.sqrt(2)*(1+0.5*z**2)**(1.5))
    pdf = pdf*scale
    return jnp.log(pdf)

@jit 
def t2_invcdf(u):
    a = 4*u*(1-u)
    z =2*(u-0.5)*jnp.sqrt(2/a)
    return z

@jit #t2 bivariate copula
def t2_copula_logdistribution_logdensity(u,v,rho):
    #clip to prevent 0s and 1s in CDF, bigger clip than float64
    eps = 1e-6
    u = jnp.clip(u,eps,1-eps)
    v = jnp.clip(v,eps,1-eps)
    
    #for reverse mode
    pu = t2_invcdf(u)
    pv = t2_invcdf(v)
    mu = rho*pv
    sigma = jnp.sqrt((1-rho**2)*(2+pv**2)/3)
    z = (pu - mu)/sigma
    
    logcop_dist = t3_logcdf(z)
    logcop_dist = jnp.clip(logcop_dist,jnp.log(eps),jnp.log(1-eps))

    logcop_dens = -jnp.log(2*jnp.pi) - 0.5*jnp.log(1-rho**2) \
    - 2*jnp.log(1 + (0.5/(1-rho**2))*(pu**2 + pv**2 - 2*rho*pu*pv)) #numerator
    logcop_dens = logcop_dens - t2_logpdf(pu) - t2_logpdf(pv)

    return logcop_dist,logcop_dens
### ###




