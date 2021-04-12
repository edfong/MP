import jax.numpy as jnp
import numpy as np

#import copula functions
from pr_copula.main_copula_density import fit_copula_density,predict_copula_density,predictive_resample_density, check_convergence_pr

### OZONE ###
#load function
def load_ozone(n_plot_marg):
    from pydataset import data
    data = data('airquality')
    var = ['Ozone','Solar.R']
    y = data[var].dropna().values
    y[:,0] = y[:,0]**(1/3)
    y = (y-np.mean(y,axis =0))/np.std(y,axis = 0)
    n = np.shape(y)[0]
    d = np.shape(y)[1]

    ylim = (-2.5,2.25)
    xlim = (-2.75,2.75)

    y_plot_marg = np.zeros((n_plot_marg,d))
    y_plot_marg[:,0] = np.linspace(-2.75,2.75,num = n_plot_marg)
    y_plot_marg[:,1] = np.linspace(-2.5,2.25,num = n_plot_marg)

    dy1 = y_plot_marg[1,0]- y_plot_marg[0,0]
    dy2 = y_plot_marg[1,1]- y_plot_marg[0,1]

    y_plot_grid = np.meshgrid(y_plot_marg[:,0],y_plot_marg[:,1])
    y_plot = np.vstack((y_plot_grid[0].ravel(), y_plot_grid[1].ravel())).transpose()
    n_plot_y = np.shape(y_plot)[0]
    return y,y_plot,y_plot_grid,n,d,ylim,xlim,y_plot_marg

#Load Ozone dataset
print('Dataset: Ozone')
n_plot_marg = 25
y,y_plot,y_plot_grid,n,d,ylim,xlim,y_plot_marg =load_ozone(n_plot_marg)
y = jnp.array([y])[0]

#Fit copula obj
copula_density_obj = fit_copula_density(y,seed = 50,single_bandwidth = False) #or seed = 200?
print('Bandwidth is {}'.format(copula_density_obj.rho_opt))
print('Preq loglik is {}'.format(copula_density_obj.preq_loglik))

#Predict on yplot
logcdf_conditionals,logpdf_joints = predict_copula_density(copula_density_obj,y_plot)

#Predictive resample
T_fwdsamples = 5000 #T = N-n
B_postsamples =1000

logcdf_conditionals_pr,logpdf_joints_pr= predictive_resample_density(copula_density_obj,y_plot,B_postsamples, T_fwdsamples, seed = 50) 

jnp.save('plot_files/copula_ozone_logpdf_pr',logpdf_joints_pr)
jnp.save('plot_files/copula_ozone_logcdf_pr',logcdf_conditionals_pr)

#Convergence plots
seed = 20
T_fwdsamples = 10000
logcdf_pr_conv,logpdf_pr_conv,pdiff,cdiff =check_convergence_pr(copula_density_obj,y_plot,1, T_fwdsamples, seed)
pdf_pr_conv = jnp.exp(logpdf_pr_conv[0,:,-1])
jnp.save('plot_files/copula_ozone_pr_pdf_samp',pdf_pr_conv)
jnp.save('plot_files/copula_ozone_pr_pdiff',pdiff)
jnp.save('plot_files/copula_ozone_pr_cdiff',cdiff)