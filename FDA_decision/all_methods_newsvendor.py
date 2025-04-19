import numpy as np
from sklearn.cluster import KMeans
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV,LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import datetime
import multiprocessing
import scipy.linalg
from scipy.optimize import minimize
from tqdm import tqdm
from kernel_opt_speed import KernelOptimization
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
import os
import pandas as pd

# suppress all warnings
import warnings
warnings.filterwarnings("ignore")


def get_decision1(x,X,Y,h,b):
    decision = (X@x.reshape(-1,1)).flatten()
    cost = np.sum(h*np.maximum(0,decision - Y) + b*np.maximum(0,Y - decision))
    return cost

#paramter training for newsvendor cost
def fun(k,X,Y,h,b,feature_number):
    res = minimize(get_decision1,np.array([0 for i in range(feature_number)]),args = (X,Y,h,b),tol = 0.01)
    return k, res.x


#Decouple OLS
def decentralized_centralised_ols(X_hats, y_hats,h,b):
    K = len(X_hats)
    feature_num = X_hats[0].shape[1]
    model_newsvendor = np.zeros((K,feature_num ))
    #model_dict = {}
    saa_decision = np.zeros(K)
    tasks = []

    data = []
    label = []
     
    for k in range(K):
        X_k = X_hats[k]
        y_k = y_hats[k]
        if len(y_k) == 0:
            saa_decision[k] = -1
        else:
            if X_k.shape[0] > X_k.shape[1]:
                # model = sm.OLS(y_k, X_k).fit()
                # model_dict[k] = model
                data.append(X_k)
                label += list(y_k)
                tasks.append((k,X_k,y_k,h,b,feature_num))
            else:
                saa_decision[k] = -1
    pool = multiprocessing.Pool(processes = 64)
    results = pool.starmap(fun,tasks)
    for i in results:
        model_newsvendor[i[0],:] = i[1]
    pool.close()

    data_array = np.concatenate(data,axis = 0)
    label_array = np.array(label)

    res = minimize(get_decision1,np.array([0 for i in range(feature_num)]),args = (data_array,label_array,h,b),tol = 0.01)
    model_prior = res.x

    return model_newsvendor, model_prior, saa_decision

#for FDA (linear + linear) cross-validation
def decentralized_shared_ols_newsvendor(X_hats, y_hats,h,b):
    K = len(X_hats)
    feature_num = X_hats[0].shape[1]
    #cov_matrix = np.zeros((K,feature_num,feature_num))
    model_array = np.zeros((K,feature_num))
    data = [X_hats[k] for k in range(K)]
    label = [y_hats[k] for k in range(K)]

    #decoupled linear newsvendor solution
    tasks = []
    for k in range(K):
        tasks.append((k,X_hats[k],y_hats[k],h,b,feature_num))
    pool = multiprocessing.Pool(processes = 64)
    results = pool.starmap(fun,tasks)
    for i in results:
        model_array[i[0],:] = i[1]
    pool.close()

    data = np.concatenate(data, axis = 0)
    label = np.concatenate(label, axis = 0)
    

    
    res = minimize(get_decision1,np.array([0 for i in range(feature_num)]),args = (data,label,h,b),tol = 0.01)
    model_prior = res.x
    
    return model_array,model_prior


#Shared Random Forest
def random_forest(X_hats,y_hats,saa_decision,h,b):
    K = len(X_hats)
    data = []
    label = []
    for k in range(K):
        if saa_decision[k] != -1:
            X_k = X_hats[k]
            y_k = y_hats[k]
            data.append(X_k)
            label += list(y_k)
    data_array = np.concatenate(data,axis = 0)
    label_array = np.array(label)
    
    
    cost = {}
    for max_depth in [1,2,3,4,5,6,7,8,9]:
        test_performance = 0
        for i in range(3):
            X_train, X_test, y_train, y_test = train_test_split(data_array, label_array, test_size=0.1)
            model_temp = RandomForestRegressor(n_estimators= 20, max_depth = max_depth,max_features = 'sqrt').fit(X_train,y_train)
            test_performance += np.mean((y_test - model_temp.predict(X_test))**2)
        cost[max_depth] = test_performance/3
    a = sorted(cost.items(), key=lambda x: x[1])
    model = RandomForestRegressor(n_estimators= 100,max_depth= a[0][0], max_features = 'sqrt').fit(data_array,label_array)
    residual_array = label_array - model.predict(data_array)
    error_critical = np.sort(residual_array)[int(residual_array.shape[0]*b)]
    return model,error_critical,a[0][0]




#select suitable alpha for  FDA (linear + linear)
def cross_validation(X_hats, y_hats,saa_decision,h,b,cv = 3):
    K = len(X_hats)
    feature_num = X_hats[0].shape[1]
    #X_hats = np.array(X_hats)

    selected_list = []
    for k in range(K):
        if saa_decision[k] != -1:
            selected_list.append(k)
    alpha_list = [0.01*i for i in range(101)]
    

    data = [[] for i in range(cv)]
    label = [[] for i in range(cv)]
    for k in selected_list:
        N =  X_hats[k].shape[0]
        index_list = np.array([i for i in range(N)])
        np.random.shuffle(index_list)
        number_per_part = int(N/cv)
        for i in range(cv):
            if i < cv - 1:
                data[i].append(X_hats[k][index_list[i*number_per_part:(i+1)*number_per_part],:])
                label[i].append(y_hats[k][index_list[i*number_per_part:(i+1)*number_per_part]])
            else:
                data[i].append(X_hats[k][index_list[i*number_per_part:],:])
                label[i].append(y_hats[k][index_list[i*number_per_part:]])

    
    cost = [0 for i in range(len(alpha_list))]
    for i in range(cv):
        valid_data = data[i]
        valid_label = label[i]
        data_temp = []
        label_temp = []
        for j in range(cv):
            if j != i:
                data_temp.append(data[j])
                label_temp.append(label[j])
        train_data = []
        train_label = []
        for k in range(len( selected_list)):
            temp_data = [data_temp[i][k] for i in range(cv - 1)]
            temp_label = [label_temp[i][k] for i in range(cv - 1)]
            train_data.append( np.concatenate(temp_data, axis = 0))
            train_label.append(np.concatenate(temp_label, axis = 0))
        

        model_ols, model_prior = decentralized_shared_ols_newsvendor(train_data, train_label,h,b)
        N1 = np.max([valid_data[k].shape[0] for k in range(len(selected_list))])
        model_ols = np.tile(model_ols[:,np.newaxis,:],(1,N1,1))
        model_prior = np.tile(model_prior[np.newaxis,np.newaxis,:],(model_ols.shape[0],N1,1))

        data_valid_matrix = np.zeros((len(selected_list),N1,feature_num))
        label_valid_matrix = np.zeros((len(selected_list),N1))
        for k in range(len(selected_list)):
            data_valid_matrix[k,0:valid_data[k].shape[0],:] = valid_data[k]
            label_valid_matrix[k,0:valid_data[k].shape[0]] = valid_label[k]
        for idx,alpha in enumerate(alpha_list):
            decision_shrunken = alpha*np.maximum(np.sum(model_ols*data_valid_matrix,axis = 2),0) + (1-alpha)*np.maximum(np.sum(model_prior*data_valid_matrix,axis = 2),0)
            cost[idx] += np.sum(h*np.maximum(0,decision_shrunken - label_valid_matrix) + b*np.maximum(0,label_valid_matrix -  decision_shrunken))
    

    return alpha_list[np.argmin(cost)]


#select suitable alpha for  FDA (linear + random forest)
def shrunken_non_linear(X_hats,y_hats,saa_decision,max_depth,h,b,cv = 3,alphas = [0.01*i for i in range(101)]):
    K = len(X_hats)
    f = X_hats[0].shape[1]
    selected_list = []
    for k in range(K):
        if saa_decision[k] != -1:
            selected_list.append(k)

    X_train = {}
    y_train = {}
    for k in selected_list:
        length = X_hats[k].shape[0]
        random_set = np.random.choice(range(length),size = 5,replace=False)
        #random_set = [length - 1]
        X_train[k] = []
        y_train[k] = []
        for i in random_set:
            list1 = [j for j in range(length) if j != i]
            X_train[k].append([X_hats[k][list1,:],X_hats[k][i,:]])
            y_train[k].append([y_hats[k][list1],y_hats[k][i]])
    models = {}
    for i in range(cv):
        total_data = []
        total_label = []

        test_data_list = {}
        xgboost_dict = {}
        tasks = []
        for k in selected_list:
            data = X_train[k][i][0]
            label = y_train[k][i][0]
            total_data.append(data)
            total_label.append(label)

            test_data = X_train[k][i][1]
            test_label = y_train[k][i][1]

            test_data_list[k] = [test_data,test_label]
            tasks.append((k,data,label,h,b,f))
            
        pool = multiprocessing.Pool(processes = 64)
        results = pool.starmap(fun,tasks)
        for j in results:
            xgboost_dict[j[0]] = j[1]
        pool.close()
        total_data = np.concatenate(total_data,axis=0)
        total_label = np.concatenate(total_label,axis=0)
        rf = RandomForestRegressor(n_estimators= 100, max_depth = max_depth,max_features = 'sqrt').fit(total_data,total_label)
        residual_array = total_label - rf.predict(total_data)
        error_critical = np.sort(residual_array)[int(residual_array.shape[0]*b)]
        pred_ols = []
        pred_rf = []
        true = []
        for k in selected_list:
            data = test_data_list[k][0]
            label = test_data_list[k][1]
            pred_ols.append(max(xgboost_dict[k] @ data,0))
            pred_rf.append(max(rf.predict(data.reshape(-1,f))[0] + error_critical,0))
            true.append(label)
        models[i] = [pred_ols,pred_rf,true]
    alphas_dict = {}
    for alpha in alphas:
        #test the alpha performance
        test_perfomance = 0
        for i in range(cv):
            y_pred = alpha*np.array(models[i][0]) + (1-alpha)*np.array(models[i][1])
            y_true = np.array(models[i][2])
            test_perfomance += np.sum(h*np.maximum(0,y_pred - y_true) + b*np.maximum(0,y_true -  y_pred))
        alphas_dict[alpha] = test_perfomance/cv
    a = sorted(alphas_dict.items(), key=lambda x: x[1])
    #print(a)
    alpha_best = a[0][0]

    # # xgoost_dict = decentralised_Xgboost(X_hats,y_hats,saa_decision)
    # # rf = random_forest(X_hats,y_hats,saa_decision)
    return alpha_best
    # #splict the all data

#select suitable alpha for  FDA (saa + linear)
def cv_saa_ols(X_hats,y_hats,saa_decision,h,b,cv = 5, alphas = [0.01*i for i in range(101)]):
    K = len(X_hats)
    f = X_hats[0].shape[1]
    selected_list = []
    for k in range(K):
        if saa_decision[k] != -1:
            selected_list.append(k)

    X_train = {}
    y_train = {}
    for k in selected_list:
        length = X_hats[k].shape[0]
        random_set = np.random.choice(range(length),size = 5,replace=False)
        #random_set = [length - 1]
        X_train[k] = []
        y_train[k] = []
        for i in random_set:
            list1 = [j for j in range(length) if j != i]
            X_train[k].append([X_hats[k][list1,:],X_hats[k][i,:]])
            y_train[k].append([y_hats[k][list1],y_hats[k][i]])
    models = {}
    for i in range(cv):
        total_data = []
        total_label = []

        test_data_list = {}
        xgboost_dict = {}
        for k in selected_list:
            data = X_train[k][i][0]
            label = y_train[k][i][0]
            total_data.append(data)
            total_label.append(label)

            test_data = X_train[k][i][1]
            test_label = y_train[k][i][1]

            test_data_list[k] = [test_data,test_label]

            bst = np.sort(label)[int(label.shape[0]*b)]
            xgboost_dict[k] = bst
        total_data = np.concatenate(total_data,axis=0)
        total_label = np.concatenate(total_label,axis=0)


        res = minimize(get_decision1,np.array([0 for i in range(f)]),args = (total_data,total_label,h,b),tol = 0.01)
        rf = res.x
        pred_ols = []
        pred_rf = []
        true = []
        for k in selected_list:
            data = test_data_list[k][0]
            label = test_data_list[k][1]
            pred_ols.append(max(xgboost_dict[k],0))
            pred_rf.append(max(data@rf,0))
            true.append(label)
        models[i] = [pred_ols,pred_rf,true]
    alphas_dict = {}
    for alpha in alphas:
        #test the alpha performance
        test_perfomance = 0
        for i in range(cv):
            y_pred = alpha*np.array(models[i][0]) + (1-alpha)*np.array(models[i][1])
            y_true = np.array(models[i][2])
            test_perfomance +=  np.sum(h*np.maximum(0,y_pred - y_true) + b*np.maximum(0,y_true -  y_pred))
        alphas_dict[alpha] = test_perfomance/cv
    a = sorted(alphas_dict.items(), key=lambda x: x[1])
    #print(a)
    alpha_best = a[0][0]
    return alpha_best

#Poold KO
def pooled_KO(X_train, d_train, X_test,  h,b,selected_list):

    K = len(X_train)
    decision_array = np.zeros(K)

    tasks = []
    data = []
    label = []
    for k in selected_list:
        data.append(X_train[k])
        label.append(d_train[k])
    X_hat = np.concatenate(data,axis = 0)
    d_hat = np.concatenate(label,axis = 0)

    kf = KFold(n_splits=3, shuffle=True)
    k_fold_sets = []
    
    for train_idx, val_idx in kf.split(X_hat):
        X_train_fold = X_hat[train_idx]
        d_train_fold = d_hat[train_idx]
        X_val_fold = X_hat[val_idx]
        d_val_fold = d_hat[val_idx]
        k_fold_sets.append(((X_train_fold, d_train_fold), (X_val_fold, d_val_fold)))

    dist_matrix = cdist(X_hat, X_hat, 'euclidean')
    b_max = np.max(dist_matrix)
    np.fill_diagonal(dist_matrix, np.inf)
    b_min = np.min(dist_matrix)

    optimizer = KernelOptimization(b, h)
    optimal_w = optimizer.find_optimal_bandwidth(k_fold_sets, b_min/2, b_max*2)

    print("ok")

    #for k in selected_list:
    q_optimal = optimizer.kernel_optimization(X_hat, d_hat, X_test[selected_list,:], optimal_w)
    decision_array[selected_list] = q_optimal

    return decision_array

# Decoupled KO
def decoupled_KO(X_train, d_train, X_test,  h,b,selected_list):

    K = len(X_train)
    decision_array = np.zeros(K)

    #tasks = []
    for k in selected_list:
        
        X_hat= X_train[k]
        d_hat = d_train[k]
        X_test1 = X_test[k]
        kf = KFold(n_splits=3, shuffle=True)
        k_fold_sets = []

        for train_idx, val_idx in kf.split(X_hat):
            X_train_fold = X_hat[train_idx]
            d_train_fold = d_hat[train_idx]
            X_val_fold = X_hat[val_idx]
            d_val_fold = d_hat[val_idx]
            k_fold_sets.append(((X_train_fold, d_train_fold), (X_val_fold, d_val_fold)))

        dist_matrix = cdist(X_hat, X_hat, 'euclidean')
        b_max = np.max(dist_matrix)
        np.fill_diagonal(dist_matrix, np.inf)
        b_min = np.min(dist_matrix)

        optimizer = KernelOptimization(b, h)
        optimal_w = optimizer.find_optimal_bandwidth(k_fold_sets, b_min/2, b_max*2)

        decision_array[k] = optimizer.kernel_optimization(X_hat, d_hat, X_test1, optimal_w)


    return decision_array

#create the decision and FDA Linear
def main(X_hats,y_hats, Xs,ys,X_PAB,y_PAB,h,b,data_size,index):
    Xs = np.array(Xs)
    ys = np.array(ys)
    #print(Xs.shape,ys.shape)
    K = len(X_hats)
    beta_0_hat = 0
    ys = np.array(ys)
    
    s1 = datetime.datetime.now()

    #Decouple OLS
    model_newsvendor,  beta_0_hat, saa_decision = decentralized_centralised_ols(X_hats, y_hats,h,b)

    saa_alpha = cv_saa_ols(X_hats,y_hats,saa_decision,h,b)

    

    alpha_hat = cross_validation(X_hats, y_hats,saa_decision,h,b)
    
    
    Gupta_decision = np.zeros(K)
    #Gupta_decision = Gupta(saa_decision, y_hats)
    
 
    rf,rf_constant, max_depth = random_forest(X_hats,y_hats,saa_decision,h,b)

   
    alpha_non_linear = shrunken_non_linear(X_hats,y_hats,saa_decision,max_depth,h,b)

    


   
    #S11 = datetime.datetime.now()
    #Calculate the decision for all methods
   
    shrunken_decision = np.zeros(K)
    shrunken_non_linear_decision = np.zeros(K)
    shruken_saa_linear = np.zeros(K)
  

    selected_list = []
    selected_non_list = []
    saa_newsvendor_decision = np.zeros(K)
    for i in range(K):
        if ys[i] != 0 and saa_decision[i] != -1:
            selected_list.append(i)
            saa_newsvendor_decision[i] = np.sort(y_hats[i])[int(y_hats[i].shape[0]*b)]
        else:
            selected_non_list.append(i)
    
    
    
    pooled_ko_decision = pooled_KO(X_hats, y_hats, Xs,  h,b,selected_list)
    decoupled_ko_decision = decoupled_KO(X_hats, y_hats, Xs,  h,b,selected_list)

    shrunken_decision[selected_list] = alpha_hat*np.maximum(np.sum(model_newsvendor[selected_list,:] * Xs[selected_list,:],axis = 1),0)+ (1-alpha_hat)*np.maximum( (Xs[selected_list,:] @ beta_0_hat.reshape(-1,1)).flatten(),0)
    shrunken_non_linear_decision[selected_list] = alpha_non_linear*np.maximum(np.sum(model_newsvendor[selected_list,:] * Xs[selected_list,:],axis = 1),0)+(1-alpha_non_linear)*np.maximum(rf.predict(Xs[selected_list,:]) + rf_constant,0)
    shruken_saa_linear[selected_list] = saa_alpha*saa_newsvendor_decision[selected_list] + (1-saa_alpha)*np.maximum( (Xs[selected_list,:] @ beta_0_hat.reshape(-1,1)).flatten(),0)
    

  
    shrunken_decision[selected_non_list] = ys[selected_non_list]
    shrunken_non_linear_decision[selected_non_list] = ys[selected_non_list]
    shruken_saa_linear[selected_non_list] = ys[selected_non_list]
    pooled_ko_decision[selected_non_list] = ys[selected_non_list]
    decoupled_ko_decision[selected_non_list] = ys[selected_non_list]
    
    Gupta_decision[selected_non_list] = ys[selected_non_list]
  
  
    
    
    
    gupta_cost =  np.mean(h*np.maximum(0,Gupta_decision - ys) + b*np.maximum(0,ys -  Gupta_decision))
    shrunken_cost = np.mean(h*np.maximum(0,shrunken_decision - ys) + b*np.maximum(0,ys -shrunken_decision))
    shrunken_non_linear_cost = np.mean(h*np.maximum(0,shrunken_non_linear_decision - ys) + b*np.maximum(0,ys - shrunken_non_linear_decision))
    shrunken_saa_linear_cost = np.mean(h*np.maximum(0,shruken_saa_linear - ys) + b*np.maximum(0,ys -shruken_saa_linear))
    pooled_ko_cost = np.mean(h*np.maximum(0,pooled_ko_decision - ys) + b*np.maximum(0,ys -pooled_ko_decision))
    decoupled_ko_cost = np.mean(h*np.maximum(0,decoupled_ko_decision - ys) + b*np.maximum(0,ys -decoupled_ko_decision))
    

 
    cost_list = [shrunken_cost,shrunken_non_linear_cost,shrunken_saa_linear_cost,pooled_ko_cost,decoupled_ko_cost]
    return cost_list

