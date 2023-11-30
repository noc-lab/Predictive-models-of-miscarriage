import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb   
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report,f1_score, roc_curve, auc, accuracy_score 
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# a function to drop variables we want to drop 
def df_drop(df, drop_cols):
    '''
    input:
    df: 2-d pandas dataframe of the dataset
    drop_cols: a list of features we want to remove 
    
    return:
    dataframe with drop_cols removed
    '''
    return df.drop(df.columns[df.columns.isin(drop_cols)], axis=1)


# Compute p-value using the chi-squared test for binary predictors
def chi2_cols(y,x):
    '''
    input:
    y: 1-d binary label array
    x: 1-d binary feature array
    
    return:
    chi2 p-value
    '''
    y_list=y.astype(int).tolist()
    x_list=x.astype(int).tolist()
    freq=np.zeros([2,2])

    for i in range(len(y_list)):
        if y_list[i]==0 and x_list[i]==0:
            freq[0,0]+=1
        if y_list[i]==1 and x_list[i]==0:
            freq[1,0]+=1
        if y_list[i]==0 and x_list[i]==1:
            freq[0,1]+=1
        if y_list[i]==1 and x_list[i]==1:
            freq[1,1]+=1
    y_0_sum=np.sum(freq[0,:])
    y_1_sum=np.sum(freq[1,:])
    x_0_sum=np.sum(freq[:,0])
    x_1_sum=np.sum(freq[:,1])
    total=y_0_sum+y_1_sum
    y_0_ratio=y_0_sum/total
    freq_=np.zeros([2,2])    
    freq_[0,0]=x_0_sum*y_0_ratio
    freq_[0,1]=x_1_sum*y_0_ratio
    freq_[1,0]=x_0_sum-freq_[0,0]
    freq_[1,1]=x_1_sum-freq_[0,1]

    stat,p_value=stats.chisquare(freq,freq_,axis=None)    
    return p_value#stat,


# Compute the variables statistics such as mean, std, p-value, and correlation
def stat_test(df, y):
    '''
    input:
    df: 2-d pandas dataframe of the dataset
    y: 1-d binary array of the label
    
    return:
    a dataframe of the variables statistics such as mean, std, p-value, and correlation-with-y
    '''
    name = pd.DataFrame(df.columns,columns=['Variable'])
    df0=df[y==0]
    df1=df[y==1]
    pvalue=[]
    y_corr=[]
    for col in df.columns:
        if df[col].nunique()==2:
            pvalue.append(chi2_cols( y,df[col]))
        else:
            pvalue.append(stats.ks_2samp(df0[col], df1[col]).pvalue)
        y_corr.append(df[col].corr(y))
    name['All_mean']=df.mean().values
    name['y1_mean']=df1.mean().values
    name['y0_mean']=df0.mean().values
    name['All_std']=df.std().values
    name['y1_std']=df1.std().values
    name['y0_std']=df0.std().values
    name['p-value']=pvalue
    name['y_corr']=y_corr
    return name.sort_values(by=['p-value'])


# Compute pairwise correlation of variables with each other
# and if the correlation is high (>0.9), we keep one variable of the highly-correlated variables
def high_corr(df, thres=0.9):
    '''
    input:
    df: 2-d dataframe of the dataset
    thres: Threshold we consider to determine highly correlated variables 
    
    return:
    a list of pairs of two highly correlated variables
    '''
    corr_matrix_raw = df.corr()
    corr_matrix = corr_matrix_raw.abs()
    high_corr_var_=np.where(corr_matrix>thres)
    high_corr_var=[(corr_matrix.index[x],corr_matrix.columns[y], corr_matrix_raw.iloc[x,y]) for x,y in zip(*high_corr_var_) if x!=y and x<y]
    return high_corr_var


# Compare correlation of outcome with highly-correlated pairs
# to decide on which var among the highly-correlated pairs we want to remove 
def highcorr_stats(df, col_y, thres=0.9):
    '''
    input:
    df: 2-d pandas dataframe of the dataset
    col_y: outcome name
    thres: Threshold we consider to determine highly correlated variables 
    
    return:
    a pandas dataframe displays highly-correlated pairs, their correlation with outcome, their p-value 
    '''
    selected_col_to_remove=[]
    data = []
    y = df[col_y].astype(int)
    df0=df[y==0]
    df1=df[y==1]
    
    select_rm_high_core_list = high_corr(df, thres)
    for ele in select_rm_high_core_list:
        col1 = df[ele[0]]
        col2 = df[ele[1]]
        col_y_ = df[col_y]
        cor_val1 = abs(col_y_.corr(col1))
        cor_val2 = abs(col_y_.corr(col2))
        if df[ele[0]].nunique()==2:
            p_val1 = chi2_cols(y, df[ele[0]])
        else:
            p_val1 = stats.ks_2samp(df0[ele[0]], df1[ele[0]]).pvalue
        if df_new[ele[1]].nunique()==2:
            p_val2 = chi2_cols(y, df[ele[1]])
        else:
            p_val2 = stats.ks_2samp(df0[ele[1]], df1[ele[1]]).pvalue
        if cor_val1 < cor_val2:
            selected_col_to_remove.append(ele[0])
        else:
            selected_col_to_remove.append(ele[1])
        data.append([abs(ele[2]), ele[0], cor_val1, p_val1, ele[1], cor_val2, p_val2])

    df_highcorr = pd.DataFrame(data, columns=['highcorr_abs','var1','y_corr1','p_val1','var2','y_corr2','p_val2'])
    return selected_col_to_remove, df_highcorr.sort_values(by=['highcorr_abs'],ascending=False)


# train a model only using the training set:
# tune the model hyperparameters via cross-validation and returns the model with the best cross-validation score fitted on the whole training set
def my_train(X_train, y_train, model='LR', penalty='l1', cv=5, scoring='f1', class_weight= 'balanced',seed=2020):    
    '''
    input:
    X_train: 2-d array of the training set except the label
    y_train: 1-d array training set label
    model: Type of algorithm we want to develop:  'LR', 'SVM', 'MLP', 'LR', or 'LGB'
    penalty: Regularization norm for linear models LR and SVM:  'l1' or 'l2'
    cv: Number of folds in cross-validation
    scoring: Strategy to evaluate the performance of the cross-validated model on the validation set: 'roc_auc', 'f1', etc
    class_weight: Weights associated with classes
    seed: random_state used to shuffle the data
    
    return:
    the model with the best cross-validation score fitted on the whole training dataset
    '''
    # use the training dataset to tune the model hyperparameters via cross-validation 
    # Support Vector Machine algorithm
    if model=='SVM':
        svc=LinearSVC(penalty=penalty, class_weight= class_weight, dual=False, max_iter=5000)#, tol=0.0001
        param_grid = {'C':[0.01,0.1,1,10,100]} #'kernel':('linear', 'rbf'), 
        gsearch = GridSearchCV(svc, param_grid, cv=cv, scoring=scoring) 
    
    # Boosted Trees algorithm    
    elif model=='LGB':        
        param_grid = {
            'num_leaves': range(2,6,2),
            'n_estimators': range(50,200,50)
            }
        lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', learning_rate=0.1, class_weight= class_weight, random_state=seed,force_row_wise=True)# eval_metric='auc' num_boost_round=2000,
        gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=cv,n_jobs=-1, scoring=scoring)
    
    # Random Forest algorithm     
    elif model=='RF': 
        rfc=RandomForestClassifier(n_estimators=100, random_state=seed, class_weight= class_weight, n_jobs=-1)
        param_grid = { 
            'max_features':[0.4,0.6],
            'max_depth' : [4,6],
            'min_samples_split': [30,40],
        }
        gsearch = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=cv, scoring=scoring)
    
    # Logistic Regression algorithm
    else:
        LR = LogisticRegression(penalty=penalty, class_weight= class_weight,solver='liblinear', random_state=seed)
        parameters = {'C':[0.1,1,10] } #,1,10
        gsearch = GridSearchCV(LR, parameters, cv=cv, scoring=scoring) 
    
    # fit the model with the best cross-validation score on the whole training dataset    
    gsearch.fit(X_train, y_train)
    clf=gsearch.best_estimator_
    print('Best parameters found by grid search are:', gsearch.best_params_)
    
    # returns the model with the best cross-validation score fitted on the whole training dataset
    return clf


# find optimal threshold that leads to the highest 'weighted_F1_score' among thresholds on the decision function used to compute fpr and tpr of the training set
def cal_f1_scores(y, y_pred_score):
    '''
    input: 
    y: Ground truth target values of the training set
    y_pred_score: Target scores of the training set that can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions
    
    return:
    Optimal threshold that leads to the highest 'weighted_F1_score' among thresholds on the decision function used to compute fpr and tpr of the training set
    '''
    # compute Receiver operating characteristic (ROC)
    fpr, tpr, thresholds = roc_curve(y, y_pred_score)
    thresholds = sorted(set(thresholds))
    metrics_all = []
    for thresh in thresholds:
        y_pred = np.array((y_pred_score > thresh))
        metrics_all.append(( thresh,auc(fpr, tpr), f1_score(y, y_pred, average='micro'), f1_score(y, y_pred, average='macro'),f1_score(y, y_pred, average='weighted')))
    metrics_df = pd.DataFrame(metrics_all, columns=['thresh','tr AUC',  'tr Accuracy', 'tr macro F1-score','tr weighted F1-score'])
    
    # returns the optimal threshold that leads to the highest 'weighted_F1_score' among thresholds on the decision function used to compute fpr and tpr of the training set
    return metrics_df.sort_values(by = 'tr weighted F1-score', ascending = False).head(1)#['thresh'].values[0]


# compute performance metrics evaluated on the test set
def cal_f1_scores_te(y, y_pred_score,thresh):
    '''
    input:
    y: Ground truth target values of the test set
    y_pred_score: Target scores of the test set that can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions
    
    return:
    a dataframe of performance metrics evaluated on the test set
    '''
    # compute Receiver operating characteristic (ROC)
    fpr, tpr, thresholds = roc_curve(y, y_pred_score)
    
    # compute the estimated targets 
    # the estimated target is 1 if the target score is greater than the optimal threshold on the decision function found using the training set
    y_pred = np.array((y_pred_score > thresh))
    metrics_all = [ (thresh,auc(fpr, tpr), f1_score(y, y_pred, average='micro'), f1_score(y, y_pred, average='macro'),f1_score(y, y_pred, average='weighted'),average_precision_score(y, y_pred_score),precision_score(y, y_pred, average='micro'), precision_score(y, y_pred, average='macro'), precision_score(y, y_pred, average='weighted'),recall_score(y, y_pred, average='weighted'))]
    metrics_df = pd.DataFrame(metrics_all, columns=['thresh','AUC', 'Accuracy', 'macro F1-score','weighted F1-score','AUPRC','micro_precision_score', 'macro_precision_score','weighted_precision_score','weighted_recall_score'])
    
    # returns a dataframe of performance metrics evaluated on the test set
    return metrics_df


# test the obtained model on the test set
def my_test(X_train, xtest, y_train, ytest, clf, target_names, report=False, model='LR'): 
    '''
    input:
    X_train: 2-d array of the training set except the label
    xtest: 2-d array of the test set except the label
    y_train: 1-d array training set label
    ytest: 1-d array test set label
    clf: the model with the best cross-validation score fitted on the whole training dataset
    target_names: 0 and 1 as the label is binary: ['0', '1'] 
    model: Type of algorithm we want to develop:  'LR', 'SVM', 'MLP', 'LR', or 'LGB'
    
    return:
    a dataframe of performance metrics evaluated on the test set
    '''
    # compute target scores of the training set that can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions
    if model=='SVM': 
        ytrain_pred_score=clf.decision_function(X_train)
    else:
        ytrain_pred_score=clf.predict_proba(X_train)[:,1]
    
    # find the optimal threshold on the decision function used to compute fpr and tpr
    metrics_tr =cal_f1_scores( y_train, ytrain_pred_score)
    thres_opt=metrics_tr['thresh'].values[0]   
    
    # compute target scores of the test set that can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions
    if model=='SVM': 
        ytest_pred_score=clf.decision_function(xtest)
    else:
        ytest_pred_score=clf.predict_proba(xtest)[:,1]
    metrics_te = cal_f1_scores_te(ytest, ytest_pred_score,thres_opt)    
    
    # returns a dataframe of performance metrics evaluated on the test set
    return metrics_te.merge(metrics_tr,on='thresh'),thres_opt,ytest_pred_score
 
    
# develop different ML algorithms 'LR', 'SVM', 'MLP', 'LR', or 'LGB'
def tr_predict(df_new, col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring='f1', test_size=0.2,report=False, RFE=False,my_range=range(1,11,1),pred_score=False):
    '''
    input:
    df_new: 2-d dataframe of the dataset after preprocessing icluding one-hot encoding and statistical feature selection
    col_y: Labe name 
    target_names: 0 and 1 as the label is binary: ['0', '1'] 
    model: Type of algorithm we want to develop:  'LR', 'SVM', 'MLP', 'LR', or 'LGB'
    penalty: Regularization norm for linear models LR and SVM:  'l1' or 'l2' 
    cv_folds: Number of folds in cross-validation
    scoring: Strategy to evaluate the performance of the cross-validated model on the validation set: 'roc_auc', 'f1', etc
    test_size: Proportion of the dataset to include in the test split
    
    return:
    a dataframe including the predictors' coefficients and statistics based on the selected algorithm
    '''
    # Standardize features by removing the mean and scaling to unit variance
    scaler = preprocessing.StandardScaler()           
    y= df_new[col_y].values # 1-d binary label array 
    metrics_all=[]# a list to keep metrics calculated on the test set for each run
    
    # the random_state that controls the shuffling applied to the data before applying the split
    my_seeds=range(2020, 2025)
    for seed in my_seeds:# we repeat the model development 5 times and we use a different seed for each run
        my_drops = [col_y]
        X = df_new.drop(df_new.columns[df_new.columns.isin(my_drops)], axis=1).values# dataset excluding the label in the format of 2-d array
        name_cols=df_new.drop(df_new.columns[df_new.columns.isin(my_drops)], axis=1).columns.values # features names
        
        # Fits transformer to X and returns a transformed version of X
        X = scaler.fit_transform(X)
        
        # Split the dataset to five random parts, where four parts constituted the training dataset, and the fifth part constituted the testing dataset
        X_train, xtest, y_train, ytest = train_test_split(X, y, stratify=y, test_size=test_size,  random_state=seed)#
        
        # train a model only using the training set
        clf = my_train(X_train, y_train, model=model, penalty=penalty, cv=cv_folds, scoring=scoring, class_weight= 'balanced',seed=seed)    
        
        # test the obtained model on the test set
        metrics_te,thres_opt, ytest_pred_score=my_test(X_train, xtest, y_train, ytest, clf, target_names, report=report, model=model)
        metrics_all.append(metrics_te)
    
    # compute the mean and standard deviation of the model performance statistics across these five runs
    metrics_df=pd.concat(metrics_all)
    metrics_df = metrics_df[cols_rep].describe().T[['mean','std']].stack().to_frame().T
    
    # refit using all samples to get non-biased coef.
    clf.fit(X, y)
    
    #create the dataframe of the predictors' coefficients based on model type
    if model=='LGB' or model=='RF': 
        df_coef_=pd.DataFrame(list(zip(name_cols, np.round(clf.feature_importances_,2))),columns=['Variable','coef_'])
    else:      
        df_coef_=pd.DataFrame(list(zip(name_cols, np.round(clf.coef_[0],2))),columns=['Variable','coef_'])
        df_coef_= df_coef_.append({'Variable': 'intercept_','coef_': np.round(clf.intercept_,2)}, ignore_index=True)
    df_coef_['coef_abs']=df_coef_['coef_'].abs()
    
    # return two dataframes
    # one dataframe including the predictors' coefficients and statistics based on the selected algorithm
    # the other dataframe including metrics (mean and std) evaluated on the test set
    return df_coef_.sort_values('coef_abs', ascending=False)[['Variable','coef_']], metrics_df
