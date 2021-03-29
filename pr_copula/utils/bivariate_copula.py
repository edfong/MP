import jax.numpy as jnp
from jax import custom_jvp,jit
from jax.scipy.stats import norm,t
from jax.scipy.special import ndtri

### NORMAL COPULA ###
@custom_jvp #forward diff (define jvp for faster derivatives)
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

@custom_jvp #forward diff (define jvp for faster derivatives)
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

@jit #Calculate normal copula cdf and logpdf for f32
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

##Probit##
#Approximations for computing bivariate copula distribution
# @jit
# def log_g_cop(u,rho):
#     eps = 1e-6
#     u = jnp.clip(u,eps,1-eps)
#     pu = ndtri_(u)
#     log_g = norm_logcdf(jnp.sqrt((1-rho)/(1+rho))*pu)
#     return log_g

# #Approximate C(u,u)
# @jit
# def norm_logbicop_diag_approx(log_u,rho):
#     eps = 1e-6
#     log_u = jnp.clip(log_u,jnp.log(eps),jnp.log(1-eps))
#     ind_true = jnp.where(log_u<=jnp.log(0.5),x = 1,y = 0) #check if u <0.5
#     log_u = ind_true*log_u + (1-ind_true)*jnp.log1p(-jnp.exp(log_u)) #replaces log(u) with log(1-u) if less than 0.5

#     u = jnp.exp(log_u)
#     log_g = log_g_cop(u,rho) #for u<0.5
#     log_interp = jnp.log((1+(rho/2)+(1/jnp.pi)*jnp.arcsin(rho)) + u*((2/jnp.pi)*jnp.arcsin(rho)- rho))
#     logbicop = log_u + log_g +log_interp

#     #add 2u-1 if u >0.5
#     logbicop = jnp.log(ind_true*jnp.exp(logbicop)+ (1-ind_true)*((1-2*u)+jnp.exp(logbicop)))

#     return logbicop

# #Approximate C(u,v), add custom deriv (copula density)
# @custom_jvp
# def norm_logbicop_approx(u,v,rho):
#     pu = jnp.where(u==0.5,x =0.5,y = ndtri_(u))
#     pv = jnp.where(v==0.5,x =0.5,y = ndtri_(v))
#     alpha_u = jnp.where(u==0.5,x =0,y = (1/jnp.sqrt(1-rho**2))*((pv/pu)-rho))
#     alpha_v = jnp.where(v==0.5,x =0, y = (1/jnp.sqrt(1-rho**2))*((pu/pv)-rho))
#     rho_u = -alpha_u /(jnp.sqrt(1+alpha_u**2))
#     rho_v = -alpha_v /(jnp.sqrt(1+alpha_v**2))

#     C_uu = jnp.exp(norm_logbicop_diag_approx(jnp.log(u),1-(2*rho_u**2)))
#     C_vv = jnp.exp(norm_logbicop_diag_approx(jnp.log(v),1-(2*rho_v**2)))

#     ind_rho_u = jnp.where(rho_u <0,x = 1,y = 0)
#     C_half_u = 0.5*ind_rho_u*C_uu + (1-ind_rho_u)*(u-0.5*C_uu)

#     ind_rho_v = jnp.where(rho_v <0,x = 1,y = 0)
#     C_half_v = 0.5*ind_rho_v*C_vv + (1-ind_rho_v)*(v-0.5*C_vv)

#     delta_uv = jnp.where(jnp.logical_or(jnp.logical_and(u<0.5,v>=0.5), jnp.logical_and(u>=0.5, v<0.5)), x = 0.5, y = 0)
#     C_uv = C_half_u + C_half_v -delta_uv
    
#     ind_v_half = jnp.where(u==0.5, x = 1, y = 0)
#     ind_u_half = jnp.where(v==0.5, x = 1, y = 0)
#     ind_uv_half_or = jnp.logical_or(u==0.5,v==0.5)
#     ind_uv_half_and = jnp.logical_and(u==0.5,v==0.5)
#     C_uv = (1-ind_uv_half_and)*ind_u_half*(u-0.5*jnp.exp(norm_logbicop_diag_approx(jnp.log(u),1-2*rho**2)))+ \
#             (1-ind_uv_half_and)*ind_v_half*(v-0.5*jnp.exp(norm_logbicop_diag_approx(jnp.log(v),1-2*rho**2)))+\
#             ind_uv_half_and*jnp.exp(norm_logbicop_diag_approx(jnp.log(u),rho)) +\
#             (1-ind_uv_half_or)*C_uv
#     return jnp.log(C_uv)
# @norm_logbicop_approx.defjvp
# def f_jvp(primals, tangents):
#     u,v,rho = primals
#     u_dot,v_dot,rho_dot = tangents
#     primal_out = norm_logbicop_approx(u,v,rho)
#     #Compute derivatives by hand
#     pu = ndtri_(u)
#     pv = ndtri_(v)
#     zu = (pv - rho*pu)/jnp.sqrt(1- rho**2)
#     u_dot_new = jnp.exp(norm_logcdf(zu))
#     zv = (pu - rho*pv)/jnp.sqrt(1- rho**2)
#     v_dot_new = jnp.exp(norm_logcdf(zv))
#     rho_dot_new = (1/(2*jnp.pi*jnp.sqrt(1-rho**2)))*jnp.exp(-(0.5/(1-rho**2))*(pu**2 + pv**2 - 2*pu*pv*rho))
#     tangent_out = (u_dot*u_dot_new + v_dot*v_dot_new + jnp.array([rho_dot])*rho_dot_new)/jnp.exp(primal_out) #divide due to log(Cuu); rho_dot is 0-d
#     return primal_out, tangent_out
# norm_logbicop_approx = jit(norm_logbicop_approx)

### ###

### Student t ###
@custom_jvp #forward diff (define jvp for faster derivatives)
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

### STUDENT-T COPULA, df = 1 ###
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

### STUDENT-T COPULA, df = 2 ###
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




