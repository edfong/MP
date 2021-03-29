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
Approximations for computing bivariate copula distribution
@jit
def log_g_cop(u,rho):
    eps = 1e-6
    u = jnp.clip(u,eps,1-eps)
    pu = ndtri_(u)
    log_g = norm_logcdf(jnp.sqrt((1-rho)/(1+rho))*pu)
    return log_g

#Approximate C(u,u)
@jit
def norm_logbicop_diag_approx(log_u,rho):
    eps = 1e-6
    log_u = jnp.clip(log_u,jnp.log(eps),jnp.log(1-eps))
    ind_true = jnp.where(log_u<=jnp.log(0.5),x = 1,y = 0) #check if u <0.5
    log_u = ind_true*log_u + (1-ind_true)*jnp.log1p(-jnp.exp(log_u)) #replaces log(u) with log(1-u) if less than 0.5

    u = jnp.exp(log_u)
    log_g = log_g_cop(u,rho) #for u<0.5
    log_interp = jnp.log((1+(rho/2)+(1/jnp.pi)*jnp.arcsin(rho)) + u*((2/jnp.pi)*jnp.arcsin(rho)- rho))
    logbicop = log_u + log_g +log_interp

    #add 2u-1 if u >0.5
    logbicop = jnp.log(ind_true*jnp.exp(logbicop)+ (1-ind_true)*((1-2*u)+jnp.exp(logbicop)))

    return logbicop

#Approximate C(u,v), add custom deriv (copula density)
@custom_jvp
def norm_logbicop_approx(u,v,rho):
    pu = jnp.where(u==0.5,x =0.5,y = ndtri_(u))
    pv = jnp.where(v==0.5,x =0.5,y = ndtri_(v))
    alpha_u = jnp.where(u==0.5,x =0,y = (1/jnp.sqrt(1-rho**2))*((pv/pu)-rho))
    alpha_v = jnp.where(v==0.5,x =0, y = (1/jnp.sqrt(1-rho**2))*((pu/pv)-rho))
    rho_u = -alpha_u /(jnp.sqrt(1+alpha_u**2))
    rho_v = -alpha_v /(jnp.sqrt(1+alpha_v**2))

    C_uu = jnp.exp(norm_logbicop_diag_approx(jnp.log(u),1-(2*rho_u**2)))
    C_vv = jnp.exp(norm_logbicop_diag_approx(jnp.log(v),1-(2*rho_v**2)))

    ind_rho_u = jnp.where(rho_u <0,x = 1,y = 0)
    C_half_u = 0.5*ind_rho_u*C_uu + (1-ind_rho_u)*(u-0.5*C_uu)

    ind_rho_v = jnp.where(rho_v <0,x = 1,y = 0)
    C_half_v = 0.5*ind_rho_v*C_vv + (1-ind_rho_v)*(v-0.5*C_vv)

    delta_uv = jnp.where(jnp.logical_or(jnp.logical_and(u<0.5,v>=0.5), jnp.logical_and(u>=0.5, v<0.5)), x = 0.5, y = 0)
    C_uv = C_half_u + C_half_v -delta_uv
    
    ind_v_half = jnp.where(u==0.5, x = 1, y = 0)
    ind_u_half = jnp.where(v==0.5, x = 1, y = 0)
    ind_uv_half_or = jnp.logical_or(u==0.5,v==0.5)
    ind_uv_half_and = jnp.logical_and(u==0.5,v==0.5)
    C_uv = (1-ind_uv_half_and)*ind_u_half*(u-0.5*jnp.exp(norm_logbicop_diag_approx(jnp.log(u),1-2*rho**2)))+ \
            (1-ind_uv_half_and)*ind_v_half*(v-0.5*jnp.exp(norm_logbicop_diag_approx(jnp.log(v),1-2*rho**2)))+\
            ind_uv_half_and*jnp.exp(norm_logbicop_diag_approx(jnp.log(u),rho)) +\
            (1-ind_uv_half_or)*C_uv
    return jnp.log(C_uv)
@norm_logbicop_approx.defjvp
def f_jvp(primals, tangents):
    u,v,rho = primals
    u_dot,v_dot,rho_dot = tangents
    primal_out = norm_logbicop_approx(u,v,rho)
    #Compute derivatives by hand
    pu = ndtri_(u)
    pv = ndtri_(v)
    zu = (pv - rho*pu)/jnp.sqrt(1- rho**2)
    u_dot_new = jnp.exp(norm_logcdf(zu))
    zv = (pu - rho*pv)/jnp.sqrt(1- rho**2)
    v_dot_new = jnp.exp(norm_logcdf(zv))
    rho_dot_new = (1/(2*jnp.pi*jnp.sqrt(1-rho**2)))*jnp.exp(-(0.5/(1-rho**2))*(pu**2 + pv**2 - 2*pu*pv*rho))
    tangent_out = (u_dot*u_dot_new + v_dot*v_dot_new + jnp.array([rho_dot])*rho_dot_new)/jnp.exp(primal_out) #divide due to log(Cuu); rho_dot is 0-d
    return primal_out, tangent_out
norm_logbicop_approx = jit(norm_logbicop_approx)

### ###
##UPDATE FOR MVCC###
#probit
def update_copula_single(logpmf1,log_v,y_new,logalpha,rho):
    eps = 1e-5

    ##PROBIT UPDATE
    logpmf1 = jnp.clip(logpmf1,jnp.log(eps),jnp.log(1-eps)) #clip u before passing to bicop
    log_v = jnp.clip(log_v,jnp.log(eps),jnp.log(1-eps)) #clip u before passing to bicop

    #Compute 
    #logbicop = norm_logbicop_diag_approx(logpmf1,rho) #log C(u,u)
    # logbicop = norm_logbicop_approx(jnp.exp(logpmf1),jnp.exp(log_v),rho) #log C(u,v)
    
    # log1v = jnp.log1p(-jnp.exp(log_v)) #log(1-v)
    # logpmf1_bicop = logpmf1 + jnp.log1p(-jnp.exp(logbicop-logpmf1)) #log(pmf - bicop)

    # logkyy_ =y_new*(logbicop -logpmf1 - log_v)+ (1-y_new)*((logpmf1_bicop)-(logpmf1+log1v))

    # log1alpha = jnp.log1p(-jnp.exp(logalpha))
    # logpmf1_new =  jnp.logaddexp(log1alpha, (logalpha+logkyy_))+logpmf1

    # #clip outliers
    # logpmf1_new =jnp.clip(logpmf1_new, jnp.log(eps),jnp.log(1-eps))

    u = jnp.exp(logpmf1)
    v = jnp.exp(log_v)
    C_uv = jnp.exp(norm_logbicop_approx(jnp.exp(logpmf1),jnp.exp(log_v),rho)) #C(u,v)
    
    kyy_ =y_new*(C_uv/(u*v))+ (1-y_new)*((u-C_uv)/(u*(1-v)))

    alpha = jnp.exp(logalpha)
    logpmf1_new =  jnp.log((1-alpha + alpha*kyy_)*u)

    #clip outliers
    logpmf1_new =jnp.clip(logpmf1_new, jnp.log(eps),jnp.log(1-eps))

    return logpmf1_new




