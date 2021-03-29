import jax.numpy as jnp
import numpy as np
import scipy as sp
import time
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import load_boston,load_diabetes
from sklearn.model_selection import train_test_split

#import copula functions
from pr_copula import mv_copula_regression as mvcr
from pr_copula.main_copula_regression_conditional import fit_copula_cregression,predict_copula_cregression

#Evaluate Models
#Copula
def evaluate_creg_copula(y,x,y_test,x_test):
    #fit copula obj
    copula_cregression_obj = fit_copula_cregression(y,x,seed = 200,n_perm_optim = 10, single_x_bandwidth = False)

    #predict ytest loglik
    logcdf_conditionals,logpdf_joints = predict_copula_cregression(copula_cregression_obj,y_test,x_test)
    test_loglik = np.mean(logpdf_joints[:,-1])
    print(test_loglik)
    print(copula_cregression_obj.rho_opt)
    print(copula_cregression_obj.rho_x_opt)
    return(test_loglik)

#GP
def evaluate_gp(y,x,y_test,x_test):
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel,WhiteKernel 
    from sklearn.gaussian_process import GaussianProcessRegressor
    np.random.seed(200)
    kernel =ConstantKernel()*RBF() + WhiteKernel()
    start = time.time()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,normalize_y= True).fit(x, y)
    mean_gp_test,std_gp_test = gp.predict(x_test,return_std = True)
    end = time.time()
    print('Gp took {}s'.format(end - start))
    
    test_loglik = np.mean(sp.stats.norm.logpdf(y_test, loc = mean_gp_test,scale = std_gp_test))
    return(test_loglik)

#Bayesian linear ridge regression
def evaluate_bayeslin(y,x,y_test,x_test):
    from sklearn.linear_model import BayesianRidge
    np.random.seed(200)
    linreg = BayesianRidge()
    linreg.fit(x,y)
    mean_lr,std_lr = linreg.predict(x_test,return_std = True)
    test_loglik = np.mean(sp.stats.norm.logpdf(y_test,loc = mean_lr,scale = std_lr))
    return(test_loglik)

#Run 10 fold CV:
def creg_copula_cv(dataset,frac_train =0.5, split_repeats = 10, seed = 100):

    #Load data
    if dataset=="Concrete":
        data = pd.read_excel('./data/Concrete_Data.xls')
        y_data = data.iloc[:,8].values
        x_data = data.iloc[:,0:8].values
    elif dataset == "Wine":
        data = pd.read_csv('./data/winequality-red.csv',sep =';')
        y_data= data.iloc[:,11].values #convert strings to integer
        x_data =  data.iloc[:,0:11].values
    elif  dataset =="Boston":
        x_data,y_data = load_boston(return_X_y = True)
    elif dataset == "Diabetes":
        x_data,y_data= load_diabetes(return_X_y = True)
    else:
        print("Dataset doesn't exist") 
        return -1

    #Start fitting
    print("Dataset: {}".format(dataset))

    #Initialize
    n_tot = np.shape(y_data)[0]
    n = np.int(frac_train*n_tot)

    test_loglik_cop = np.zeros(split_repeats)
    test_loglik_gp = np.zeros(split_repeats)
    test_loglik_bayeslin = np.zeros(split_repeats)

    #Train-test split
    for k in tqdm(range(split_repeats)):

        #Split + standardize
        train_ind,test_ind = train_test_split(np.arange(n_tot),test_size = n_tot - n,train_size = n,random_state = seed+k)
        
        y = y_data[train_ind]
        mean_y = np.mean(y,axis = 0)
        std_y = np.std(y,axis = 0)
        y = (y-mean_y)/std_y

        x = x_data[train_ind]
        mean_x = np.mean(x,axis = 0)
        std_x = np.std(x,axis = 0)
        x = (x-mean_x)/std_x

        y_test = y_data[test_ind]
        y_test = (y_test-mean_y)/std_y
        x_test = x_data[test_ind]
        x_test = (x_test-mean_x)/std_x

        #convert to Jax arary
        y = jnp.array(y)
        y_test = jnp.array(y_test)
        x = jnp.array(x)
        x_test = jnp.array(x_test)

        #Fit copula method
        test_loglik_cop[k] = evaluate_creg_copula(y,x,y_test,x_test)
        test_loglik_gp[k] = evaluate_gp(y,x,y_test,x_test)
        test_loglik_bayeslin[k] = evaluate_bayeslin(y,x,y_test,x_test)

    #Save test logliks
    np.save('plot_files/copula_regression_loglik{}'.format(dataset),test_loglik_cop)
    np.save('plot_files/bayeslin_regression_loglik{}'.format(dataset),test_loglik_bayeslin)
    np.save('plot_files/gp_regression_loglik{}'.format(dataset),test_loglik_gp)

creg_copula_cv("Concrete")
creg_copula_cv("Wine")
creg_copula_cv("Boston")
creg_copula_cv("Diabetes")
