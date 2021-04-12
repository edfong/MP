import jax.numpy as jnp
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time

#import copula functions
from pr_copula.main_copula_density import fit_copula_density,predict_copula_density


#Drop highly correlated variables
def drop_corr(y,threshold= 0.98): 
    
    data = pd.DataFrame(y)
    # Create correlation matrix
    corr_matrix = data.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    y = data.drop(columns = to_drop).values
    
    return(y)

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

#Gaussian
def evaluate_Gaussian(y,y_test):
    from sklearn.mixture import BayesianGaussianMixture

    np.random.seed(300)
    
    start = time.time()
    #Fit Gaussian
    Gaussian = BayesianGaussianMixture(n_components=1,covariance_type = 'full',n_init =1,\
                             weight_concentration_prior_type='dirichlet_process')
    Gaussian.fit(y)
    
    #Predict
    test_loglik = Gaussian.score(y_test)
    test_loglik = jnp.mean(sp.stats.multivariate_normal.logpdf(y_test,mean = jnp.mean(y,axis= 0),cov = jnp.cov(jnp.transpose(y))))
    
    return(test_loglik)

#Run 10 fold CV:
def mv_copula_cv(dataset,frac_train =0.5, split_repeats = 10, seed = 100):

    #Load data
    if dataset=="Breast":
        #Breast
        data = pd.read_csv('data/wdbc.data', header = None)
        data[data == '?']= np.nan
        data.dropna(axis = 0,inplace = True)
        data = data.iloc[:,2:]
    elif dataset == "Ionosphere":
        #Ionosphere
        data = pd.read_csv('data/ionosphere.data', header = None)
        data = data.iloc[:,2:34]
    elif  dataset =="Parkinsons":
        #Parksinsons
        data = pd.read_csv('data/parkinsons.data')
        data[data == '?']= np.nan
        data.dropna(axis = 0,inplace = True)
        data = data.drop(columns=['status','name']).values 
    elif dataset == "Wine":
        #Wine
        data = pd.read_csv('data/wine.data',header = None)
        data[data == '?']= np.nan
        data.dropna(axis = 0,inplace = True)
        data = data.iloc[:,1:]
    else:
        print("Dataset doesn't exist") 
        return -1

    #Start fitting
    print("Dataset: {}".format(dataset))
    #Drop correlated features
    data = drop_corr(data)

    #Initialize
    n_tot = np.shape(data)[0]
    n = np.int(frac_train*n_tot)

    test_loglik_cop = np.zeros(split_repeats)
    test_loglik_kde = np.zeros(split_repeats)
    test_loglik_dpmm = np.zeros(split_repeats)
    test_loglik_gaussian = np.zeros(split_repeats)

    #Train-test split
    for k in tqdm(range(split_repeats)):

        #Split + standardize
        train_ind,test_ind = train_test_split(np.arange(n_tot),test_size = n_tot - n,train_size = n,random_state = seed+k)
        
        y = data[train_ind]
        mean_y = np.mean(y,axis = 0)
        std_y = np.std(y,axis = 0)
        y = (y-mean_y)/std_y

        y_test = data[test_ind]
        y_test = (y_test-mean_y)/std_y

        #convert to Jax arary
        y = jnp.array(y)
        y_test = jnp.array(y_test)

        #Fit copula method
        test_loglik_cop[k] = evaluate_mv_copula(y,y_test)

        #Fit baseline methods
        test_loglik_kde[k] = evaluate_KDE(y,y_test)
        test_loglik_dpmm[k] = evaluate_DPMM(y,y_test)
        test_loglik_gaussian[k] = evaluate_Gaussian(y,y_test)

    #Save test log-likelihoods
    np.save('plot_files/copula_mv_loglik{}'.format(dataset),test_loglik_cop)
    np.save('plot_files/kde_mv_loglik{}'.format(dataset),test_loglik_kde)
    np.save('plot_files/dpmm_mv_loglik{}'.format(dataset),test_loglik_dpmm)
    np.save('plot_files/gaussian_mv_loglik{}'.format(dataset),test_loglik_gaussian)

mv_copula_cv("Breast")
mv_copula_cv("Ionosphere")
mv_copula_cv("Parkinsons")
mv_copula_cv("Wine")



