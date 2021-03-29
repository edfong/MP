import jax.numpy as jnp
import numpy as np
import scipy as sp
import time
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#import copula functions
from pr_copula import mv_copula_classification as mvcc
from pr_copula.main_copula_classification import fit_copula_classification,predict_copula_classification

#Fit models
#Copula
def evaluate_classification_copula(y,x,y_test,x_test):
    #fit copula obj
    copula_classification_obj = fit_copula_classification(y,x,seed = 200,n_perm_optim = 10, single_x_bandwidth = False) 

    #predict ytest loglik
    logpmf1 = predict_copula_classification(copula_classification_obj,x_test)
    test_loglik = np.mean(y_test*logpmf1[:,-1] + (1-y_test)*np.log(1-jnp.exp(logpmf1[:,-1])))
    print(test_loglik)
    print(copula_classification_obj.rho_opt)
    print(copula_classification_obj.rho_x_opt)
    return(test_loglik)

#GP
def evaluate_gp(y,x,y_test,x_test):
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel,WhiteKernel 
    from sklearn.gaussian_process import GaussianProcessClassifier
    np.random.seed(200)
    kernel =ConstantKernel()*RBF() + WhiteKernel()
    start = time.time()
    gp = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=10).fit(x, y)
    logp = np.log(gp.predict_proba(x_test))
    end = time.time()
    print('Gp took {}s'.format(end - start))
    
    test_loglik = np.mean(y_test.reshape(-1)*logp[:,1] + (1-y_test.reshape(-1))*logp[:,0])
    return(test_loglik)

#Logistic Regression
def evaluate_logistic(y,x,y_test,x_test):
    from sklearn.linear_model import LogisticRegression
    np.random.seed(200)
    logreg = LogisticRegression(C= 0.1)
    logreg.fit(x,y)
    logp = logreg.predict_log_proba(x_test)
    test_loglik = np.mean(y_test*logp[:,1] + (1-y_test)*logp[:,0])
    return(test_loglik)

#Run 10 fold CV:
def classification_copula_cv(dataset,frac_train =0.5, split_repeats = 10, seed = 100):

    #Load data
    if dataset=="Breast":
        data = pd.read_csv('data/wdbc.data', header = None)

        data[data == '?']= np.nan
        data.dropna(axis = 0,inplace = True)
        y_data= data.iloc[:,1].values #convert strings to integer
        x_data =  data.iloc[:,2:,].values

        #set to binary
        y_data[y_data=='B']=0 #benign
        y_data[y_data=='M'] = 1 #malignant
        y_data = y_data.astype('int')

    elif dataset == "Statlog":
        data = pd.read_csv('data/german.data', header = None,delim_whitespace = True)
        data[data == '?']= np.nan
        data.dropna(axis = 0,inplace = True)
        y_data = data.iloc[:,20].values #convert strings to integer
        x_data =  data.iloc[:,0:20]

        #set to binary
        y_data[y_data==1]=0 #good
        y_data[y_data==2] = 1 #bad
        y_data = y_data.astype('int')

        #convert to dummy 
        d = np.shape(x_data)[1]
        types = np.array([1.,0.,1.,1.,0.,1.,1.,0.,1.,1.,0.,1.,0.,1.,1.,0.,1.,0.,1.,1.,])
        for j in range(d):
            if types[j] ==1.:
                x_data.iloc[:,j] = pd.factorize(x_data.iloc[:,j])[0]
        x_data = x_data.values

    elif  dataset =="Ionosphere":
        data = pd.read_csv('data/ionosphere.data', header = None)
        y_data= data.iloc[:,34].values #convert strings to integer
        x_data =  data.iloc[:,0:34]
        x_data=x_data.drop(1,axis = 1).values #drop constant columns

        #set to binary
        y_data[y_data=='g'] = 1 #good
        y_data[y_data=='b'] = 0 #bad
        y_data = y_data.astype('int')

    elif dataset == "Parkinsons":
        data = pd.read_csv('data/parkinsons.data')
        data[data == '?']= np.nan
        data.dropna(axis = 0,inplace = True)
        y_data = data['status'].values #convert strings to integer
        x_data = data.drop(columns = ['name','status']).values
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
    test_loglik_logistic = np.zeros(split_repeats)

    #Train-test split
    for k in tqdm(range(split_repeats)):

        #Split + standardize
        train_ind,test_ind = train_test_split(np.arange(n_tot),test_size = n_tot - n,train_size = n,random_state = seed+k)
        
        y = y_data[train_ind]

        x = x_data[train_ind]
        mean_x = np.mean(x,axis = 0)
        std_x = np.std(x,axis = 0)
        x = (x-mean_x)/std_x

        y_test = y_data[test_ind]

        x_test = x_data[test_ind]
        x_test = (x_test-mean_x)/std_x

        #convert to Jax arary
        y = jnp.array(y)
        y_test = jnp.array(y_test)
        x = jnp.array(x)
        x_test = jnp.array(x_test)

        #Fit copula method
        test_loglik_cop[k] = evaluate_classification_copula(y,x,y_test,x_test)
        test_loglik_gp[k] = evaluate_gp(y,x,y_test,x_test)
        test_loglik_logistic[k] =evaluate_logistic(y,x,y_test,x_test)

    #save test logliks
    np.save('plot_files/copula_classification_loglik{}'.format(dataset),test_loglik_cop)
    np.save('plot_files/logistic_classification_loglik{}'.format(dataset),test_loglik_logistic)
    np.save('plot_files/gp_classification_loglik{}'.format(dataset),test_loglik_gp)

classification_copula_cv("Breast")
classification_copula_cv("Statlog")
classification_copula_cv("Ionosphere")
classification_copula_cv("Parkinsons")
