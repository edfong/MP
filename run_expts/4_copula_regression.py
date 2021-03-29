import jax.numpy as jnp
import time
import pandas as pd
import numpy as np

#import copula functions
from pr_copula.main_copula_regression_joint import fit_copula_jregression,predict_copula_jregression,predictive_resample_jregression,check_convergence_pr_jregression
from pr_copula.main_copula_regression_conditional import fit_copula_cregression,predict_copula_cregression,predictive_resample_cregression,check_convergence_pr_cregression


## LIDAR ##
#Load data
print('Dataset: LIDAR')
DATA_URI = 'http://www.stat.cmu.edu/~larry/all-of-nonpar/=data/lidar.dat'

df = pd.read_csv(DATA_URI,delim_whitespace = True)
y = df['logratio'].values
x = df['range'].values.reshape(-1,1)

y =(y-np.mean(y))/jnp.std(y)
x = (x - np.mean(x,axis = 0))/np.std(x,axis = 0)

#Save for R
np.savetxt("R_data/lidar_x.csv",x,delimiter = ',')
np.savetxt("R_data/lidar_y.csv",y,delimiter = ',')


y_plot = np.linspace(-3.0,2.0,num = 200)
n_plot_y = np.shape(y_plot)[0]
dy = y_plot[1] - y_plot[0]

x_plot = np.linspace(jnp.min(x),jnp.max(x),num = 40)
n_plot_x = np.shape(x_plot)[0]
dx = x_plot[1] - x_plot[0]

n = jnp.shape(y)[0]
n_plot_marg = np.array([jnp.shape(x_plot)[0],jnp.shape(y_plot)[0]])

xlim = (-2.0,2.0)
ylim = (-2.5,1.7)

z_plot_grid = np.meshgrid(x_plot,y_plot)
x_plot_ravel = z_plot_grid[0].ravel().reshape(-1,1)
y_plot_ravel = z_plot_grid[1].ravel()


### JOINT COPULA METHOD ###
print('Method: Joint Copula')
#fit copula obj
copula_jregression_obj = fit_copula_jregression(y,x,seed = 200,single_bandwidth = False,n_perm_optim = 10)
print('Bandwidth is {}'.format(copula_jregression_obj.rho_opt))
print('Preq loglik is {}'.format(copula_jregression_obj.preq_loglik))

#predict yplot
logcdf_conditionals,logpdf_joints = predict_copula_jregression(copula_jregression_obj,y_plot_ravel,x_plot_ravel)

## Compute predictive means and quantiles
pdf_cop_condj = jnp.exp(logpdf_joints[:,-1]- logpdf_joints[:,-2])
cdf_cop_condj = jnp.exp(logcdf_conditionals[:,-1])
cdf_condj_plot = cdf_cop_condj.reshape(n_plot_marg[1],n_plot_marg[0])
pdf_condj_plot = pdf_cop_condj.reshape(n_plot_marg[1],n_plot_marg[0])

jnp.save('plot_files/jcopula_lidar_pdf_plot',pdf_condj_plot)
jnp.save('plot_files/jcopula_lidar_cdf_plot',cdf_condj_plot)


#predictive resample (single x)
#Setup y grid
for x_pr_val in [0,-3]:
    y_pr = y_plot
    x_pr = jnp.array([x_pr_val]) #plot for x = 0
    z_pr_grid = np.meshgrid(x_pr,y_pr)
    x_pr_grid = z_pr_grid[0].ravel().reshape(-1,1)
    y_pr_grid = z_pr_grid[1].ravel()

    T_fwdsamples = 5000 #T = N-n
    B_postsamples =1000
    logcdf_conditionals_pr,logpdf_joints_pr= predictive_resample_jregression(copula_jregression_obj,y_pr_grid,x_pr_grid,B_postsamples, T_fwdsamples, seed = 20)

    jnp.save('plot_files/jcopula_lidar_logpdf_pr{}'.format(x_pr_val),logpdf_joints_pr)
    jnp.save('plot_files/jcopula_lidar_logcdf_pr{}'.format(x_pr_val),logcdf_conditionals_pr)

    #Convergence plot
    seed = 200
    T_fwdsamples = 10000
    logcdf_pr_conv,logpdf_pr_conv,pdiff,cdiff =check_convergence_pr_jregression(copula_jregression_obj,y_pr_grid,x_pr_grid,1, T_fwdsamples, seed)
    jnp.save('plot_files/jcopula_lidar_pr_pdiff{}'.format(x_pr_val),pdiff)


#Predictive resample for median (multiple x)
#Setup y grid for each x based on 95% credible interval
bot25_joint = np.zeros(n_plot_x)
top25_joint = np.zeros(n_plot_x)
mean_cop_joint = np.zeros(n_plot_x)

for j in range(n_plot_x):
    bot25_joint[j] =y_plot[np.searchsorted(cdf_condj_plot[:,j],0.025)-1]
    top25_joint[j] =y_plot[np.searchsorted(cdf_condj_plot[:,j],0.975)-1]
    mean_cop_joint[j] = jnp.sum(pdf_condj_plot[:,j]*y_plot*dy)

n_grid_x = jnp.shape(x_plot)[0]
n_grid_y = 40
y_plot_median = np.linspace(bot25_joint,top25_joint,n_grid_y,axis = -1)

y_pr = y_plot_median.flatten().reshape(n_grid_y * n_grid_x,1)
x_pr = jnp.repeat(x_plot.reshape(-1,1),n_grid_y, axis = 1).reshape(n_grid_y * n_grid_x,1)

T_fwdsamples = 5000 #T = N-n
B_postsamples =1000
logcdf_conditionals_pr,logpdf_joints_pr= predictive_resample_jregression(copula_jregression_obj,y_pr,x_pr,B_postsamples, T_fwdsamples, seed = 20)

cdf_condj_pr = jnp.exp(logcdf_conditionals_pr[:,:,-1])
cdf_condj_pr = cdf_condj_pr.reshape(B_postsamples,n_grid_x,n_grid_y)
jnp.save('./plot_files/jcopula_lidar_cdf_median',cdf_condj_pr)




### CONDITIONAL COPULA METHOD ###
print('Method: Conditional Copula')
from pr_copula.main_copula_regression_conditional import fit_copula_cregression,predict_copula_cregression,predictive_resample_cregression,check_convergence_pr_cregression

copula_cregression_obj = fit_copula_cregression(jnp.array(y),jnp.array(x),single_x_bandwidth = False,n_perm_optim = 10)
print(copula_cregression_obj.rho_opt)
print(copula_cregression_obj.rho_x_opt)
print(copula_cregression_obj.preq_loglik)

logcdf_conditionals,logpdf_joints = predict_copula_cregression(copula_cregression_obj,y_plot_ravel,x_plot_ravel)


## Compute predictive means and quantiles
pdf_cop_condc = jnp.exp(logpdf_joints)
cdf_cop_condc = jnp.exp(logcdf_conditionals)
cdf_condc_plot = cdf_cop_condc.reshape(n_plot_marg[1],n_plot_marg[0])
pdf_condc_plot = pdf_cop_condc.reshape(n_plot_marg[1],n_plot_marg[0])

n_plot_x = np.shape(x_plot)[0]

jnp.save('plot_files/ccopula_lidar_pdf_plot',pdf_condc_plot)
jnp.save('plot_files/ccopula_lidar_cdf_plot',cdf_condc_plot)

#Setup y grid
for x_pr_val in [0,-3]:
    y_pr = y_plot
    x_pr = jnp.array([x_pr_val]) #plot for x = 0
    z_pr_grid = np.meshgrid(x_pr,y_pr)
    x_pr_grid = z_pr_grid[0].ravel().reshape(-1,1)
    y_pr_grid = z_pr_grid[1].ravel()

    #predictive resample (single x)
    T_fwdsamples = 5000 #T = N-n
    B_postsamples =1000
    logcdf_pr,logpdf_pr= predictive_resample_cregression(copula_cregression_obj,x,y_pr_grid,x_pr_grid,B_postsamples, T_fwdsamples, seed = 200) 

    jnp.save('plot_files/ccopula_lidar_logpdf_pr{}'.format(x_pr_val),logpdf_pr)
    jnp.save('plot_files/ccopula_lidar_logcdf_pr{}'.format(x_pr_val),logcdf_pr)


    #Convergence plot
    seed = 200
    T_fwdsamples = 10000
    logcdf_pr_conv,logpdf_pr_conv,pdiff,cdiff =check_convergence_pr_cregression(copula_cregression_obj,x,y_pr_grid,x_pr_grid,1, T_fwdsamples, seed)
    jnp.save('plot_files/ccopula_lidar_pr_pdiff_pr{}'.format(x_pr_val),pdiff)

#Gaussian Process
print('Method: GP')
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,WhiteKernel 
from sklearn.gaussian_process import GaussianProcessRegressor
kernel =ConstantKernel()*RBF() + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,normalize_y = True)
gp.fit(x, y)
mean_gp,std_gp = gp.predict(x_plot.reshape(-1,1),return_std = True) 
jnp.save('plot_files/gp_lidar_mean',mean_gp)
jnp.save('plot_files/gp_lidar_std',std_gp)
