import jax.numpy as jnp
import numpy as np
from scipy.stats import bernoulli,norm

#import copula functions
from pr_copula import mv_copula_density as mvcd
from pr_copula import sample_mv_copula_density as samp_mvcd
from pr_copula.main_copula_density import fit_copula_density,predict_copula_density,predictive_resample_density,check_convergence_pr,sample_pr_quantiles_density

### TOY GMM ###
#Simulate toy data
def gen_data(seed,n):
    np.random.seed(seed)
    p = bernoulli.rvs(0.2,size = n)
    y = np.random.randn(n)
    y[p==0] +=-2
    y[p==1]+=2
    return y.reshape(-1,1)

#Consider 2 dataset sizes
for n in [50,200]:
    print('Dataset: GMM with n={}'.format(n))
    #Load data
    seed = 102
    d = 1
    y = gen_data(seed,n)
    mean_y = jnp.mean(y)
    std_y = jnp.std(y)
    y = (y-mean_y)/std_y

    #Save data for R
    np.savetxt("R_data/gmm_n{}.csv".format(n),y,delimiter = ',')

    #Plot of true pdf
    y_plot = jnp.arange(-4,4,0.05).reshape(-1,1)
    dy = y_plot[1]-y_plot[0]
    true_pdf = (0.8*norm.pdf(std_y*y_plot+mean_y,loc = -2) + 0.2*norm.pdf(std_y*y_plot+mean_y,loc = 2))*std_y

    #Fit copula obj
    copula_density_obj = fit_copula_density(y,seed = 200)
    print('Bandwidth is {}'.format(copula_density_obj.rho_opt))
    print('Preq loglik is {}'.format(copula_density_obj.preq_loglik))

    #Predict on yplot
    logcdf_conditionals,logpdf_joints = predict_copula_density(copula_density_obj,y_plot)
    pdf_cop = jnp.exp(logpdf_joints[:,-1])
    cdf_cop = jnp.exp(logcdf_conditionals[:,-1])

    #Predictive resample
    T_fwdsamples = 5000 #T = N-n
    B_postsamples = 1000
    logcdf_conditionals_pr,logpdf_joints_pr= predictive_resample_density(copula_density_obj,y_plot,B_postsamples, T_fwdsamples, seed = 200) #(seed = 200)

    jnp.save('plot_files/copula_gmm_logpdf_pr_n{}'.format(n),logpdf_joints_pr)
    jnp.save('plot_files/copula_gmm_logcdf_pr_n{}'.format(n),logcdf_conditionals_pr)

    #Convergence plots
    seed = 200 
    T_fwdsamples = 10000
    _,_,pdiff,cdiff =check_convergence_pr(copula_density_obj,y_plot,1, T_fwdsamples, seed)
    jnp.save('plot_files/copula_gmm_pr_pdiff_n{}'.format(n),pdiff)
    jnp.save('plot_files/copula_gmm_pr_cdiff_n{}'.format(n),cdiff)
### ###

### GALAXY ###
#Load data
print('Dataset: Galaxy')
from pydataset import data
y = data('galaxies').values/1000
mean_y = jnp.mean(y)
std_y = jnp.std(y)
y = (y-mean_y) /std_y
n = jnp.shape(y)[0]
d = 1
n_plot = 200
y_plot = (jnp.linspace(5,40,n_plot).reshape(-1,1) - mean_y)/std_y

#Fit copula obj
copula_density_obj = fit_copula_density(y,seed = 200,single_bandwidth = False)
print('Bandwidth is {}'.format(copula_density_obj.rho_opt))
print('Preq loglik is {}'.format(copula_density_obj.preq_loglik))

#Predict on yplot
logcdf_conditionals,logpdf_joints = predict_copula_density(copula_density_obj,y_plot)

#Predictive resample
T_fwdsamples = 5000 #T = N-n
B_postsamples =1000

logcdf_conditionals_pr,logpdf_joints_pr= predictive_resample_density(copula_density_obj,y_plot,B_postsamples, T_fwdsamples, seed = 200)

jnp.save('plot_files/copula_galaxy_logpdf_pr',logpdf_joints_pr)
jnp.save('plot_files/copula_galaxy_logcdf_pr',logcdf_conditionals_pr)


#Convergence plots
seed = 200 
T_fwdsamples = 10000
_,_,pdiff,cdiff =check_convergence_pr(copula_density_obj,y_plot,1, T_fwdsamples, seed)
jnp.save('plot_files/copula_galaxy_pr_pdiff',pdiff)
jnp.save('plot_files/copula_galaxy_pr_cdiff',cdiff)

