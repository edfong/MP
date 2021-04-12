import jax.numpy as jnp
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time

#import copula functions
from pr_copula.main_copula_density import fit_copula_density,predict_copula_density

#Evaluate models
#Copula
def evaluate_mv_copula(y,y_test):
    #fit copula obj
    copula_density_obj = fit_copula_density(y,seed = 50,n_perm_optim = 10, single_bandwidth = True)

    #predict ytest loglik
    logcdf_conditionals,logpdf_joints = predict_copula_density(copula_density_obj,y_test)
    test_loglik = np.mean(logpdf_joints[:,-1])
    print(test_loglik)
    print(copula_density_obj.rho_opt)
    return(test_loglik)

#KDE
def evaluate_KDE(y,y_test):
    np.random.seed(200)

    #Fit KDE_bandwidth
    start = time.time()
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    params = {'bandwidth': np.logspace(-1, 1, 40)}
    grid = GridSearchCV(KernelDensity(), params,cv = 10)
    grid.fit(y)
    #print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    kde = grid.best_estimator_
    end =time.time()
    print('KDE fitting time is {}'.format(end-start))
    
    #Predict
    test_loglik = jnp.mean(kde.score_samples(y_test))
    
    return(test_loglik)

#DPMM with VB       
def evaluate_DPMM(y,y_test):
    from sklearn.mixture import BayesianGaussianMixture

    np.random.seed(300)
    d = np.shape(y)[1]
    start = time.time()
    #Fit GMM
    DPMM = BayesianGaussianMixture(n_components=30,covariance_type = 'diag',n_init =100,\
                             weight_concentration_prior_type='dirichlet_process',
                                  covariance_prior = np.ones(d),degrees_of_freedom_prior = d,mean_precision_prior = 1, mean_prior = np.zeros(d))

    DPMM.fit(y)
    end =time.time()
    print('DPMM fitting time is {}'.format(end-start))
      
    #Predict
    test_loglik = DPMM.score(y_test)
    
    return(test_loglik)

#Simulate from d-dimensional diagonal GMM
def simulate_GMM(d):
    np.random.seed(100)
    K = 2
    n = 100
    n_test = 1000
    mu = np.array([[2]*d,[-1]*d])
    sigma2 = np.ones((K,d))
    z = sp.stats.bernoulli.rvs(p = 0.5,size = n)
    y = np.random.randn(n,d)*np.sqrt(sigma2[z,:]) + mu[z,:]
    mean_norm = np.mean(y,axis = 0)
    std_norm = np.std(y,axis = 0)
    y = (y- mean_norm)/std_norm

    z_test = sp.stats.bernoulli.rvs(p = 0.5,size = n_test)
    y_test = np.random.randn(n_test,d)*np.sqrt(sigma2[z_test,:]) + mu[z_test,:]

    y_test = (y_test - mean_norm)/std_norm #normalize test data to have 0 mean, 1 std

    y = jnp.array(y)
    y_test = jnp.array(y_test)

    return y,y_test

#Change d and record test loglik
print('Dataset: High-d GMM')
d_range = np.array([1,2,10,20,40,60,80,100])
test_loglik_kde = np.zeros(np.size(d_range))
test_loglik_dpmm = np.zeros(np.size(d_range))
test_loglik_copula = np.zeros(np.size(d_range))
for j in tqdm(range(np.size(d_range))):
    y,y_test = simulate_GMM(d_range[j])
    test_loglik_kde[j]= evaluate_KDE(y,y_test)
    test_loglik_copula[j] = evaluate_mv_copula(y,y_test)
    test_loglik_dpmm[j] = evaluate_DPMM(y,y_test)
    
#Save test log-likelihoods
np.save('plot_files/copula_mv_loglik{}'.format("gmm"),test_loglik_copula)
np.save('plot_files/kde_mv_loglik{}'.format("gmm"),test_loglik_kde)
np.save('plot_files/dpmm_mv_loglik{}'.format("gmm"),test_loglik_dpmm)
