"""
Author: VICTOR DJIEMBOU
addedAt: 28/11/2023
changes:
- 28/11/2023:
	- add pipeline methods
    """
#################################################
##          Libraries importation
#################################################

###### Begin

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import make_scorer, mean_absolute_error
#from sklearn.metrics import mean_squared_error as MSE
#from hyperopt import hp, fmin, tpe
from sklearn.model_selection import GridSearchCV, StratifiedKFold
#from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score,accuracy_score,f1_score, recall_score
from modules.file import save_model, save_dataset
#from sklearn.linear_model import LinearRegression
#from sklearn.svm import SVC
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import BaggingClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from catboost import CatBoostClassifier
#import lightgbm as lgb
#from lightgbm import LGBMRegressor
#from bayes_opt import BayesianOptimization
from memory_profiler import profile

###### End


#################################################
##          Methods definition
#################################################
# @profile
def test_train(dataframe, target, test_size=0.2, random_state=12):
    """Split train test data labels
    Args:
    dataframe: dataframe
    target: hue variables
    test_size: test data size,
    random_state: way to randomize example choose


    Returns:
    x_train, x_test, y_train, y_test
    """

    X = dataframe.drop([target], axis=1)
    Y = dataframe[target]

    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=test_size,random_state=random_state)
    #save_dataset(x_train, 'x_train')
    x_train.reset_index(inplace = True)
    x_test.reset_index(inplace = True)

    x_train = x_train.drop(['index'], axis=1)
    x_train.reset_index(inplace = True)
    x_train = x_train.drop(['index'], axis=1)

    x_test = x_test.drop(['index'], axis=1)
    x_test.reset_index(inplace = True)
    x_test = x_test.drop(['index'], axis=1)

    return x_train, x_test, y_train, y_test

# @profile
def init_models():
    """Load the dict of model to use to perform analyse
    Args:
    None


    Returns:
    A dict of init model
    """

    knc = KNeighborsClassifier() 
    #algorithm='ball_tree', leaf_size=10, n_neighbors=18, p=1, weights='distance'
    dtc = DecisionTreeClassifier()
    lrc = LogisticRegression()
    rfc = RandomForestClassifier()
    xgb = XGBClassifier(booster = 'gbtree', use_label_encoder=False)

    # svc = SVC(
    #     kernel = 'linear'
    #     #,cache_size = 7100
    #     #, verbose = True
    # )
    #mnb = MultinomialNB()
    #abc = AdaBoostClassifier()
    #bc = BaggingClassifier()
    #etc = ExtraTreesClassifier()
    #gbdt = GradientBoostingClassifier()	#cat = CatBoostClassifier(depth=7, iterations=300, l2_leaf_reg= 1, learning_rate= 0.1,verbose=0) #
    #lgb = lgb.LGBMClassifier(colsample_bytree= 0.7378703019867917,learning_rate= 0.007929963347654646,max_depth=5,min_child_weight= 0.05345076003503776,num_leaves= 20,subsample= 0.892939141154265)

    clfs = {
    'xgb':xgb,
    'dtc':dtc,
    'lrc':lrc,
    'rfc':rfc,
    #'sv' :svc,
    #'knn':knc
    }

    return clfs

# @profile
def init_training_store(dataframe):
    """Initialize training information storage dataframe
    Args:
    dataframe: a dataframe 


    Returns:
    New dataframe with just columns
    """

    cols = dataframe.columns.to_list()
    #print(cols)
    cols.extend([
    'precision',
    'accuracy',
    'recall',
    'f1-score'
    ])
    return pd.DataFrame(columns=cols)

# @profile
def get_xgb_imp(xgb):
    """Get XGBOOST feature score in prediction before training
    Args:
    xgb: a xgboost classifier


    Returns:
    The dict of features with their importance
    """

    imp_vals = xgb.get_booster().get_fscore()
    total = sum(imp_vals.values())
    return {k:v/total for k,v in imp_vals.items()}

# @profile
def save_shap(clf, name, x_test):
    explainer = shap.Explainer(clf.predict, x_test)
    shap_values = explainer(x_test)
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename = './plots/shap_summary_'+name+'_'+timestr+'.png'
    shap.summary_plot(shap_values,show=False)
    plt.savefig(filename,dpi=700) #.png,.pdf will also support here
    filename1 = './plots/shap_bar_'+name+'_'+timestr+'.png'
    shap.plots.bar(shap_values,show=False)
    plt.savefig(filename1,dpi=700) #.png,.pdf will also support here

# @profile
def train_classifier(name, clf,X_train,y_train,X_test,y_test, store, domain, prefix, cwd):
    """Train a classifier on a dataframe
    Args:
    name: name of classifier
    clf: classifier instance
    X_train: training data
    y_train: training class
    X_test: test data
    y_test: test class
    store: storage dataframe
    domain: domain action training


    Returns:
    the training storage
    """

    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred, average='macro')
    f1_score_r = f1_score(y_test,y_pred, average='macro')

    #save_model(clf, name+'_'+domain, prefix)
    link_to_model = save_model(
                    cwd= cwd, 
                    clf= clf, 
                    clf_name= f'{name}_{domain}', 
                    prefix= prefix
                    )
    if 'lr' in name or 'sv' in name:
        #print(f"support_vectors_:{clf.support_vectors_} dual_coef_{clf.dual_coef_}")
        vals = list(clf.coef_[0])
        #print(f"len {len(vals)}")
        #print(vals)
        vals.extend([precision,accuracy,recall,f1_score_r])
        keys = X_train.columns.to_list()
        keys.extend([
            'precision',
            'accuracy',
            'recall',
            'f1-score'
            ])
        #print(f"{name} keys:{len(keys)} vals:{len(vals)}")
        store.loc[name] = pd.Series(
                vals, 
                index=keys
                )
        store.fillna(0, inplace=True)
        #print(f"{store.isna().sum()} ---")
        #save_dataset(store, name+'_'+domain)
    elif 'rf' in name or 'dt' in name or 'kn' in name:
        vals = list(clf.feature_importances_)
        #print(f"len {len(vals)}")
        vals.extend([precision,accuracy,recall,f1_score_r])
        keys = X_train.columns.to_list()
        keys.extend([
            'precision',
            'accuracy',
            'recall',
            'f1-score'
            ])
        #print(f"{name} keys:{len(keys)} vals:{len(vals)}")
        store.loc[name] = pd.Series(
                vals, 
                index=keys
                )
        store.fillna(0, inplace=True)
        #print(f"{store.isna().sum()} +++")
        #save_dataset(store, name+'_'+domain)
    elif 'xg' in name:
        vals = get_xgb_imp(clf)
        keys = list(vals.keys())
        vals = list(vals.values())
        keys.extend([
            'precision',
            'accuracy',
            'recall',
            'f1-score'
            ])
        vals.extend([precision,accuracy,recall,f1_score_r])
        #print(f"{name} keys:{len(keys)} vals:{len(vals)}")
        store.loc[name] = pd.Series(
                vals, 
                index=keys
                )
        store.fillna(0, inplace=True)
        #print(f"{store.isna().sum()} ***")
        #save_dataset(store, name+'_'+domain)

    return store

# @profile
def train(clfs,x_train,y_train,x_test,y_test, store, domain, prefix, cwd):
    """Train our baseline classifiers
    Args:
    clfs: dict of classifiers instance
    X_train: training data
    y_train: training class
    X_test: test data
    y_test: test class
    store: storage dataframe
    domain: domain action training


    Returns:
    The training storage
    """

    for name,clf in clfs.items():
        print(name)
        store = train_classifier(name, clf, x_train,y_train,x_test,y_test, store, domain, prefix, cwd)

    # save of model training logs    
    link_to_evaluations_data = save_dataset(
                    cwd= cwd, 
                    dataframe= store, 
                    name= domain, 
                    prefix= prefix, 
                    sep= '\t'
                    )

    return store



