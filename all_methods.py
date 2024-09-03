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
import scipy.linalg
from tqdm import tqdm
import pandas as pd

# suppress all warnings
import warnings
warnings.filterwarnings("ignore")



#Decouple OLS
def decentralized_ols(X_hats, y_hats):
    K = len(X_hats)
    model_dict = {}
    saa_decision = np.zeros(K)
    
    for k in range(K):
        X_k = X_hats[k]
        y_k = y_hats[k]
        if len(y_k) == 0:
            saa_decision[k] = -1
        else:
            if X_k.shape[0] > X_k.shape[1]:
                model = sm.OLS(y_k, X_k).fit()
                model_dict[k] = model
            else:
                saa_decision[k] = -1
    
    return model_dict, saa_decision

#Shared Random Forest
def random_forest(X_hats,y_hats,saa_decision):
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
    for max_depth in [1,2,3,4,5]:
        test_performance = 0
        for i in range(3):
            X_train, X_test, y_train, y_test = train_test_split(data_array, label_array, test_size=0.1)
            model_temp = RandomForestRegressor(n_estimators= 20, max_depth = max_depth,max_features = 'sqrt').fit(X_train,y_train)
            test_performance += np.mean((y_test - model_temp.predict(X_test))**2)
        cost[max_depth] = test_performance/3
    a = sorted(cost.items(), key=lambda x: x[1])
    model = RandomForestRegressor(n_estimators= 20,max_depth= a[0][0], max_features = 'sqrt').fit(data_array,label_array)
    return model,a[0][0]

#Shared OLS
def centalised_ols(X_hats,y_hats,saa_decision):
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
    model = sm.OLS(label_array, data_array).fit()

    return model

#PAB linear
def PAB_linear(X_PAB,y_PAB,saa_decision):
    K,N,f = X_PAB.shape
    selected_list = []
    weight_list = []

    for k in range(K):
        if saa_decision[k] != -1:
            selected_list.append(k)
            weight_list.append(np.mean([i for i in y_PAB[k,-12:] if i!=0]))

    data = []
    y_proxy = []
    y_true = []
    for i in range(N):
        total_demand = 0
        total_weight_demand = 0
        for idx,k in enumerate(selected_list):
            if y_PAB[k,i] != 0:
                total_demand += y_PAB[k,i]
                total_weight_demand += weight_list[idx]
        for idx,k in enumerate(selected_list):
            if y_PAB[k,i] != 0:
                y_true.append(y_PAB[k,i])
                y_proxy.append(total_demand*weight_list[idx]/total_weight_demand)
                data.append(X_PAB[k,i,:])

    y_proxy = np.array(y_proxy)
    data = np.array(data)
    y_true = np.array(y_true)
    #proxy beta
    ols_proxy = sm.OLS(np.array(y_proxy), np.array(data)).fit()

    #estimated beta
    y_adjust = y_true - ols_proxy.predict(data)
    ols_joint = LassoCV(fit_intercept = False).fit(data,y_adjust)

    return ols_joint.coef_ + ols_proxy.params

#PAB tree, Detailed process can be found in the paper: Pooling and Boosting for Demand Prediction in Retail:A Transfer Learning Approach

#Shrunken SAA
def Gupta(saa_decision, y_hats):
    K = saa_decision.shape[0]
    Gupta_decision = np.zeros(K)
    selected_list = {}
    anchor_mean = []
    for k in range(K):
        if saa_decision[k] != -1:
            selected_list[k] = y_hats[k]
            anchor_mean.append(np.mean(y_hats[k]))

    anchor_mean = np.mean(anchor_mean)
    munerator = 0
    denominator = 0
    for i in selected_list.keys():
        his_data_k = selected_list[i]
        if his_data_k.shape[0] > 1:
            munerator += np.sum((his_data_k - anchor_mean)**2)/(his_data_k.shape[0]-1)
            denominator += (np.mean(his_data_k) - anchor_mean)**2 - np.sum((his_data_k - anchor_mean)**2)/((his_data_k.shape[0]-1)*his_data_k.shape[0])
    

    Gupta_decision[:] = -1
    if denominator != 0:
        alpha = munerator/denominator
        for k in selected_list.keys():
            his_data_k = selected_list[k]
            Gupta_decision[k] = (his_data_k.shape[0]/(alpha + his_data_k.shape[0]))*np.mean(his_data_k) + (alpha/(alpha + his_data_k.shape[0]))*anchor_mean
    
    
    return Gupta_decision

#DAC
def DAC(ols_models,saa_decision,upper_ratio,lower_ratio,X_hats, y_hats,cluster_num = 2):

    K = len(X_hats)
    d = X_hats[0].shape[1]
    z = cluster_num
    #DAC = np.zeros(beta_hats.shape)
    selected_list = []
    DAC = np.zeros((K,d))
    for k in range(K):
        if saa_decision[k] != -1:
            selected_list.append(k)


    #index1 = np.random.choice(selected_list,size = 1)[0]
    index1 = selected_list[0]
    n = len(selected_list)
    # model fitting - our method
    aggre_level = []
    clus_columns = []
    all_coeff = np.zeros((K,d))
    all_coeff[index1,:] = ols_models[index1].params
    n_cols_alg = 0

    
    for j in range(d):

        # a n-1 vector recording if two betas have the same mean
        test_j = np.zeros(K)
        

        for i in selected_list:
            if i != index1:
                all_coeff[i,j] = ols_models[i].params[j]

                z_stat = ( np.abs(ols_models[index1].params[j] - ols_models[i].params[j]) / 
                            np.sqrt(np.square(ols_models[index1].bse[j]) + np.square(ols_models[i].bse[j])) )
                p_value = 1 - norm.cdf(z_stat)
                if p_value >= 0.05:  #P-value ->0.05
                    test_j[i] = 1

        if np.sum(test_j) >= upper_ratio*(n-1):
            aggre_level.append('dept')
            n_cols_alg += 1

        elif np.sum(test_j) <= lower_ratio*(n-1):
            aggre_level.append('sku')
            n_cols_alg += n

        else:
            aggre_level.append('clus')
            clus_columns.append(j)
            n_cols_alg += z

    
    kemans_list = []
    if len(clus_columns) > 0:
        for i in clus_columns:            
            #X_clus = all_coeff[:, clus_columns]
            X_clus = all_coeff[selected_list,i].reshape(-1,1)
            kmeans = KMeans(n_clusters = z, random_state = 0, n_init='auto').fit(X_clus)
            kemans_list.append(kmeans)
    
    

    sample_dict = []
    a = 0
    for i in selected_list:
        sample_dict.append(X_hats[i].shape[0])
        a += X_hats[i].shape[0]

    data =  np.zeros((a,d))
    y = np.zeros(a)
    initial = 0
    for idx,i in enumerate(selected_list):
        data[initial:initial+sample_dict[idx]] = X_hats[i]
        y[initial:initial+sample_dict[idx]] = y_hats[i]
        initial = initial + sample_dict[idx]


    X_alg = np.zeros((a, n_cols_alg))

    count = 0
    
    count_dict = {}
    for i in range(d):
        if aggre_level[i] == 'dept':
            X_alg[:,count] = data[:,i]
            count_dict[count] = [selected_list,i]
            count += 1

        elif aggre_level[i] == 'clus':
            idx = clus_columns.index(i)
            for j in range(z):
                clus_items = list(np.where(kemans_list[idx].labels_ == j)[0])
                for sku in clus_items:
                    index2 = sku
                    if index2 == 0 :
                        initial = 0
                    else:
                        initial = np.sum(sample_dict[0:index2])
                    X_alg[initial:initial+sample_dict[index2], count] = data[initial:initial+sample_dict[index2], i]
                count_dict[count] = [[selected_list[k] for k in clus_items],i]
                count += 1
        else:
            for index2,j in enumerate(selected_list):
                if index2 == 0:
                    initial = 0
                else:
                    initial = np.sum(sample_dict[0:index2])
                X_alg[initial:initial+sample_dict[index2], count] = data[initial:initial+sample_dict[index2], i]
                count_dict[count] = [[j],i]
                count += 1
    
    try:
        model_0 = LinearRegression(fit_intercept=False,n_jobs = -1).fit(X_alg,y)
        for i in range(n_cols_alg):
            a = count_dict[i][0]
            j = count_dict[i][1]
            para = model_0.coef_[i]
            for k in a:
                DAC[k,j] = para
    except scipy.linalg.LinAlgError as e:
        print('error')
        DAC  = all_coeff


    # sgd_regressor = SGDRegressor(max_iter=100, tol=1e-3,fit_intercept=False,penalty = None) 
    # sgd_regressor.fit(X_alg, y)
    # if sgd_regressor.n_iter_ == sgd_regressor.max_iter:
    #     DAC  = all_coeff
    #     print("Reached the maximum number of iterations: {}".format(len(selected_list)))
    # else:
    # for i in range(n_cols_alg):
    #     a = count_dict[i][0]
    #     j = count_dict[i][1]
    #     para = sgd_regressor.coef_[i]
    #     for k in a:
    #         DAC[k,j] = para
       
        
    return DAC

#FDA tree
def shrunken_non_linear(X_hats,y_hats,saa_decision,max_depth,cv = 5,alphas = [0.05*i for i in range(20)]):
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

            bst = sm.OLS(label,data).fit()
            xgboost_dict[k] = bst
        total_data = np.concatenate(total_data,axis=0)
        total_label = np.concatenate(total_label,axis=0)
        rf = RandomForestRegressor(n_estimators= 20, max_depth = max_depth,max_features = 'sqrt').fit(total_data,total_label)
        pred_ols = []
        pred_rf = []
        true = []
        for k in selected_list:
            data = test_data_list[k][0]
            label = test_data_list[k][1]
            pred_ols.append(max(xgboost_dict[k].predict(data.reshape(-1,f))[0],0))
            pred_rf.append(max(rf.predict(data.reshape(-1,f))[0],0))
            true.append(label)
        models[i] = [pred_ols,pred_rf,true]
    alphas_dict = {}
    for alpha in alphas:
        #test the alpha performance
        test_perfomance = 0
        for i in range(cv):
            y_pred = alpha*np.array(models[i][0]) + (1-alpha)*np.array(models[i][1])
            y_true = np.array(models[i][2])
            test_perfomance += np.mean((y_pred - y_true)**2)
        alphas_dict[alpha] = test_perfomance/cv
    a = sorted(alphas_dict.items(), key=lambda x: x[1])
    #print(a)
    alpha_best = a[0][0]

    # # xgoost_dict = decentralised_Xgboost(X_hats,y_hats,saa_decision)
    # # rf = random_forest(X_hats,y_hats,saa_decision)
    return alpha_best
    # #splict the all data
    # X_train = {}
    # y_train = {}
    # for k in selected_list:
    #     X_train[k] = [X_hats[k][0:-1,:],X_hats[k][-1:,:]]
    #     y_train[k] = [y_hats[k][0:-1],y_hats[k][-1:]]
    
    # xgboost_dict = {}
    # total_data = []
    # total_label = []
    # for k in selected_list:
    #     total_data.append(X_train[k][0])
    #     total_label.append(y_train[k][0])
    #     train_data = X_train[k][0]
    #     train_label = y_train[k][0]

    #     #train xgboost
    #     #bst = xgb.XGBRegressor(tree_method="hist")
    #     #bst.fit(train_data, train_label)
    #     bst = sm.OLS(train_label,train_data).fit()
    #     xgboost_dict[k] = bst
    
    # #train the random forest
    # total_data = np.concatenate(total_data,axis=0)
    # total_label = np.concatenate(total_label,axis=0)
    # rf = RandomForestRegressor(n_estimators= 20, max_depth = 2).fit(total_data,total_label)
    
    # feature1 = []
    # feature2 = []
    # label = []
    # for k in selected_list:
    #      feature1 += list(xgboost_dict[k].predict(X_train[k][1].reshape(-1,f)))
    #      feature2 += list(rf.predict(X_train[k][1].reshape(-1,f)))
    #      label += list(list(y_train[k][1]))
    # feature1 = np.maximum(np.array(feature1),0)
    # feature2 = np.maximum(np.array(feature2),0)
    # # data = np.ones((feature1.shape[0],2))
    # # data[:,0] = feature1
    # # data[:,1] = feature2
    # label = np.array(label)
    # alpha_best = np.sum((label - feature2)*(feature1 - feature2))/np.sum((feature1 - feature2)**2)
    # residual = np.mean((label - alpha_best*feature1 - (1-alpha_best)*feature2))
    # feature = np.array([[feature1[i],feature2[i]] for i in range(len(feature1))])
    # label = np.array(label)
    
    #model = RidgeCV(alphas = [0.001,0.01,0.1,1,10],cv = 3).fit(data,label)
    #model = sm.OLS(label, data).fit()

    #return min(max(alpha_best,0),1)

                         

    # alphas_dict = {}
    # for alpha in alphas:
    #     #test the alpha performance
    #     y_true = []
    #     y_pred = []
    #     for k in selected_list:
    #         pred = alpha*xgboost_dict[k].predict(X_train[k][1].reshape(-1,f)) + (1-alpha)*rf.predict(X_train[k][1].reshape(-1,f))
    #         y_true += list(y_train[k][1])
    #         y_pred += list(pred)
    #     y_true = np.array(y_true)
    #     y_pred = np.array(y_pred)
    #     # print(test_performance)
    #     alphas_dict[alpha] = np.mean((y_pred - y_true)**2)
    # # tasks = []
    # # for alpha in alphas:
    # #     tasks.append([X_train,y_train,selected_list,alpha,f])
    # # pool = multiprocessing.Pool(processes = 8)
    # # results = pool.starmap(kernel,tasks)

    # # for j in results:
    # #     alphas_dict[j[0]] = j[1]

    #a = sorted(alphas_dict.items(), key=lambda x: x[1])
    #print(a)
    #alpha_best = a[0][0]

    # # xgoost_dict = decentralised_Xgboost(X_hats,y_hats,saa_decision)
    # # rf = random_forest(X_hats,y_hats,saa_decision)
    #return alpha_best

#create the decision and FDA Linear
def main(X_hats,y_hats, Xs,ys,X_PAB,y_PAB):
    K = len(X_hats)
    beta_0_hat = 0
    ys = np.array(ys)
    
    s1 = datetime.datetime.now()

    #Decouple OLS
    ols_dict, saa_decision = decentralized_ols(X_hats, y_hats)


    s2 = datetime.datetime.now()

    
    #Shrunken SAA
    Gupta_decision = Gupta(saa_decision, y_hats)
    
    s3 = datetime.datetime.now()

    #Shared Random Forest
    rf,max_depth = random_forest(X_hats,y_hats,saa_decision)

    s4 = datetime.datetime.now()
    
    #FDA tree
    alpha_non_linear = shrunken_non_linear(X_hats,y_hats,saa_decision,max_depth)

    s5 = datetime.datetime.now()
    
    #DAC
    DAC_para = DAC(ols_dict,saa_decision,0.9,0.6,X_hats, y_hats)

    s6 = datetime.datetime.now()

    #PAB linear
    PAB_para = PAB_linear(X_PAB,y_PAB,saa_decision)

    s7 = datetime.datetime.now()

    #PAB tree
    PAB_para_tree = PAB_tree(X_PAB,y_PAB,saa_decision)

    s8 = datetime.datetime.now()
    
    #Shared OLS
    beta_0_hat = centalised_ols(X_hats,y_hats,saa_decision)

    s9 = datetime.datetime.now()

    # Calculate the numerator and denominator for \hat{\alpha} of linear FDA
    numerator = 0
    denominator = 0
    
    for k in range(K):
        if ys[k] != 0 and saa_decision[k] != -1:
            beta_diff = (ols_dict[k].params - beta_0_hat.params).reshape(-1,1)
            x_k = Xs[k].reshape(-1,1)
            
            numerator += (x_k.T @ (beta_diff @ beta_diff.T) @ x_k)[0,0]
            numerator -= ( x_k.T @ ols_dict[k].cov_params() @ x_k)[0,0]

            denominator += (x_k.T @ (beta_diff @ beta_diff.T) @ x_k)[0,0]
    alpha_hat = numerator / denominator
    
    s10 = datetime.datetime.now()

    #Calculate the decision for all methods
    ols_decision = np.zeros(K)
    shrunken_decision = np.zeros(K)
    prior_decision = np.zeros(K)
    DAC_deicision = np.zeros(K)
    rf_decision = np.zeros(K)
    PAB_linear_decision = np.zeros(K)
    shrunken_non_linear_decision = np.zeros(K)
    PAB_tree_decision = np.zeros(K)

    
    for i in range(K):
        if ys[i] != 0 and saa_decision[i] != -1:
            ols_decision[i] = max(ols_dict[i].predict(Xs[i])[0],0)
            prior_decision[i] = max(beta_0_hat.predict(Xs[i])[0],0)
            shrunken_decision[i] = max(alpha_hat*ols_decision[i] + (1-alpha_hat)*prior_decision[i],0)
            DAC_deicision[i] = max(DAC_para[i,:] @ Xs[i],0)
            Gupta_decision[i] = max(Gupta_decision[i],0)
            rf_decision[i] = max(rf.predict(Xs[i].reshape(1,-1)),0)
            PAB_linear_decision[i] = max(PAB_para @ Xs[i],0)
            shrunken_non_linear_decision[i] = max(alpha_non_linear*ols_decision[i]+(1-alpha_non_linear)*rf_decision[i],0)
            PAB_tree_decision[i] = max(PAB_para_tree.predict(xgb.DMatrix(Xs[i].reshape(1,-1)))[0],0)
        else:
            ols_decision[i] = ys[i]
            shrunken_decision[i] = ys[i]
            prior_decision[i] = ys[i]
            Gupta_decision[i] = ys[i]
            DAC_deicision[i] = ys[i]
            rf_decision[i] = ys[i]
            PAB_linear_decision[i] = ys[i]
            shrunken_non_linear_decision[i] = ys[i]
            PAB_tree_decision[i] = ys[i]
        
    
    #Calculate the MSE out-of-sample cost
    ols_cost = np.mean((ols_decision - ys)**2)
    gupta_cost = np.mean((Gupta_decision-ys)**2)
    rf_cost = np.mean((rf_decision - ys)**2)
    shrunken_non_linear_cost = np.mean((shrunken_non_linear_decision - ys)**2)
    DAC_cost = np.mean((DAC_deicision - ys)**2)
    PAB_linear_cost = np.mean((PAB_linear_decision - ys)**2)
    PAB_tree_cost = np.mean((PAB_tree_decision - ys)**2)
    prior_cost = np.mean((prior_decision - ys)**2)
    shrunken_cost = np.mean((shrunken_decision-ys)**2)
    

    cost_list = [ols_cost,gupta_cost,rf_cost,
                 shrunken_non_linear_cost,DAC_cost,PAB_linear_cost,
                 PAB_tree_cost,prior_cost,shrunken_cost]
    decision_list = [list(ols_decision),list(Gupta_decision),list(rf_decision),
                     list(shrunken_non_linear_decision),list(DAC_deicision),list(PAB_linear_decision),
                     list(PAB_tree_decision),list(prior_decision),list(shrunken_decision)]
    time_list = [(s2-s1).seconds,(s3-s2).seconds,(s4-s3).seconds,
                 (s2-s1).seconds + (s4-s3).seconds + (s5-s4).seconds,
                 (s6-s5).seconds,(s7-s6).seconds,(s8-s7).seconds,
                 (s2-s1).seconds + (s9-s8).seconds,
                 (s2-s1).seconds + (s10-s9).seconds + (s9-s8).seconds]
    return cost_list,decision_list,time_list,[alpha_hat,0]

