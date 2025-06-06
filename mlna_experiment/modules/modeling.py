"""
Author: VICTOR DJIEMBOU
addedAt: 28/11/2023
changes:
- 28/11/2023:
	- add pipeline methods
- 30/09/2024
    - replace coefficient importance by SHAPLey values in train classifier method
    """
#################################################
##          Libraries importation
#################################################

###### Begin

# 1. Imports des bibliothèques standard
import sys

# 2. Imports des bibliothèques externes
import numpy as np
import pandas as pd
import shap
import random

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score


# 3. Imports locaux ou spécifiques au projet
from .file import save_model, save_dataset


# Commentaires pour des imports non utilisés (à retirer ou décommenter si nécessaire)
# from sklearn.metrics import make_scorer, mean_absolute_error
# from sklearn.metrics import mean_squared_error as MSE
# from hyperopt import hp, fmin, tpe
# from lightgbm import LGBMClassifier
# from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
# from sklearn.linear_model import LinearRegression
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import BaggingClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from catboost import CatBoostClassifier
#import lightgbm as lgb
#from lightgbm import LGBMRegressor
#from bayes_opt import BayesianOptimization
# from memory_profiler import profile

###### End


#################################################
##          Methods definition
#################################################

# @profile
def init_models():
    """Load the dict of model to use to perform analyse
    Args:
    None


    Returns:
    A dict of init model
    """

    # knc = KNeighborsClassifier()
    #algorithm='ball_tree', leaf_size=10, n_neighbors=18, p=1, weights='distance'
    dtc = DecisionTreeClassifier()
    lrc = LogisticRegression()
    rfc = RandomForestClassifier()
    xgb = XGBClassifier(booster='gbtree', use_label_encoder=False)
    lda = LinearDiscriminantAnalysis()
    svc = SVC(
        kernel='linear'
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),  # 1 couche cachée de 100 neurones
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    perceptron = Perceptron()
    #mnb = MultinomialNB()
    #abc = AdaBoostClassifier()
    #bc = BaggingClassifier()
    #etc = ExtraTreesClassifier()
    #gbdt = GradientBoostingClassifier()	#cat = CatBoostClassifier(depth=7, iterations=300, l2_leaf_reg= 1, learning_rate= 0.1,verbose=0) #
    #lgb = lgb.LGBMClassifier(colsample_bytree= 0.7378703019867917,learning_rate= 0.007929963347654646,max_depth=5,min_child_weight= 0.05345076003503776,num_leaves= 20,subsample= 0.892939141154265)

    clfs = {
        'LDA': lda,
        'LR': lrc,
        'SVM': svc,
        'DT': dtc,
        'RF': rfc,
        'XGB': xgb,
        'MLP': mlp,
        'PER': perceptron,
    }

    return clfs

# @profile
def init_training_store(dataframe, withCost=True):
    """Initialize training information storage dataframe
    Args:
    dataframe: a dataframe 


    Returns:
    New dataframe with just columns
    """

    cols = dataframe.columns.to_list()
    cols.extend([
        'precision',
        'accuracy',
        'recall',
        'f1-score',
        'financial-cost'
    ]) if withCost else cols.extend([
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
    return {k: v / total for k, v in imp_vals.items()}

# @profile
def train_classifier(
        name,
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        store,
        domain,
        prefix,
        cwd,
        financialOption,
        duration_divider,
        rate_divider,
        original,
        withCost=True
):
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

    # print(original)
    if isinstance(clf, LinearDiscriminantAnalysis):
        try:
            clf.fit(X_train, y_train)
        except np.linalg.LinAlgError:
            print("SVD did not converge for LDA. Trying with 'lsqr' solver.")
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf.fit(X_train, y_train)
    elif isinstance(clf, LogisticRegression):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with Logistic Regression: {str(e)}")
            # Retry with different solver if needed
            clf = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
            clf.fit(X_train, y_train)
    elif isinstance(clf, SVC):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with Support Vector Classifier: {str(e)}")
            # Retry with lower max_iter if it doesn't converge
            clf = SVC(kernel='linear', random_state=42, max_iter=500)
            clf.fit(X_train, y_train)
    elif isinstance(clf, DecisionTreeClassifier):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with Decision Tree Classifier: {str(e)}")
            # Try with different parameters
            clf = DecisionTreeClassifier(random_state=42, max_depth=10)
            clf.fit(X_train, y_train)
    elif isinstance(clf, RandomForestClassifier):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with Random Forest Classifier: {str(e)}")
            # Retry with fewer estimators
            clf = RandomForestClassifier(random_state=42, n_estimators=50)
            clf.fit(X_train, y_train)
    elif isinstance(clf, XGBClassifier):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with XGBoost Classifier: {str(e)}")
            # Retry with fewer trees or lower learning rate
            clf = XGBClassifier(
                booster='gbtree',
                use_label_encoder=False,
                random_state=42,
                n_estimators=50,
                learning_rate=0.05
            )
            clf.fit(X_train, y_train)
    elif isinstance(clf, Perceptron):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with Perceptron: {str(e)}")
            clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
            clf.fit(X_train, y_train)
    elif isinstance(clf, MLPClassifier):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with MLPClassifier: {str(e)}")
            clf = MLPClassifier(
                hidden_layer_sizes=(100,),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
            clf.fit(X_train, y_train)
    else:
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with classifier: {str(e)}")

    y_pred = clf.predict(X_test)
    df_predictions = pd.DataFrame({'Prediction': y_pred})
    save_dataset(
        cwd=cwd,
        dataframe=df_predictions,
        name=f'{domain}_{name}_y_pred',
        prefix=prefix,
        sep=',',
        sub='/predictions'
    )
    cfm = compute_confusion_matrix(y_test, y_pred, list(np.unique(y_test)))

    accuracy_r = accuracy_macro(cfm)
    precision_r = precision_macro(cfm)
    f1_score_r = f1_macro(cfm)
    recall_r = recall_macro(cfm)
    if withCost:
        cost = compute_classification_financial_cost(
            list(y_pred),
            list(y_test),
            financialOption,
            original[1],
            duration_divider,
            rate_divider
        )

    # SHAP calculation and storage
    explainer = None
    shap_values = None

    if isinstance(clf, (LogisticRegression, Perceptron, SVC, LinearDiscriminantAnalysis)):
        explainer = shap.LinearExplainer(clf, X_train)
        shap_values = explainer.shap_values(X_test)
    elif isinstance(clf, (RandomForestClassifier, DecisionTreeClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        # Si classification binaire, shap_values sera une liste de deux éléments
        # print(shap_values.ndim)
        # # Vérification pour numpy.ndarray
        # if isinstance(shap_values, np.ndarray):
        #     if shap_values.ndim > 2:
        #         # Si c'est un tableau 3D, prenons la dernière dimension (généralement la classe positive)
        #         shap_values = shap_values[:, :, 1]  # [:, :, -1] pour la classe 0 et [:, :, 1] pour la classe 1
        # elif isinstance(shap_values, list) and len(shap_values) == 2:
        #     # Garde l'ancien comportement pour la compatibilité
        #     shap_values = shap_values[1]
    else:
        explainer = shap.KernelExplainer(clf.predict_proba, X_train)
        shap_values = explainer.shap_values(X_test)

    # Store SHAP values along with metrics
    shap_vals_mean = np.mean(np.abs(shap_values), axis=0)  # Mean absolute SHAP values

    vals = list(shap_vals_mean)
    vals.extend([precision_r, accuracy_r, recall_r, f1_score_r, cost] if withCost else [precision_r, accuracy_r, recall_r, f1_score_r] )

    keys = X_train.columns.to_list()
    keys.extend([
        'precision',
        'accuracy',
        'recall',
        'f1-score',
        'financial-cost'
    ]) if withCost else keys.extend([
        'precision',
        'accuracy',
        'recall',
        'f1-score'
    ])

    store.loc[name] = pd.Series(vals, index=keys)
    store.fillna(0, inplace=True)
    # store.fillna(0, inplace=True)
    #print(f"{store.isna().sum()} ***")
    #save_dataset(store, name+'_'+domain)

    return store


# @profile
def train(
        clfs,
        x_train,
        y_train,
        x_test,
        y_test,
        store,
        domain,
        prefix,
        cwd,
        duration_divider,
        rate_divider,
        financialOption,
        original,
        withCost=True
):
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

    for name, clf in clfs.items():
        # try:
        print(name, domain)
        store = train_classifier(
            name=name,
            clf=clf,
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            store=store,
            domain=domain,
            prefix=prefix,
            cwd=cwd,
            financialOption=financialOption,
            withCost=withCost,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            original=original
        )
    # except e:
    #     print(f"An exception occurred during training: {name} {e}")

    # save of model training logs    
    # link_to_evaluations_data = save_dataset(
    #     cwd=cwd,
    #     dataframe=store,
    #     name=domain,
    #     prefix=prefix,
    #     sep=',',
    #     sub="/evaluation"
    # )

    return store

def compute_classification_financial_cost(ypred, yreal, financialOption, test_dataset, duration_divider, rate_divider):
    """
    Compute the financial cost of an investment
    Parameters
    ----------
    ypred : predicted values
    yreal : real values
    financialOption : financial option
    test_dataset : test dataset
    duration_divider : duration divider
    rate_divider : rate divider

    Returns
    -------
    cost : financial cost
    """
    cost = 0
    # print(len(test_dataset),len(yreal))
    for i,example in enumerate(list(test_dataset.index)):
        rate = test_dataset[financialOption['rate']][example] / rate_divider
        amount = test_dataset[financialOption['amount']][example]
        duration = test_dataset[financialOption['duration']][example] / duration_divider
        # print(rate, amount, duration)
        true_label = yreal[i]
        predicted_label = ypred[i]

        if predicted_label == 0 and true_label == 1:  # Bad debtor announced as good
            cost += amount * 1 # exxpected to really be cost += amount * loss_given_default but like it's hard to fine or compute this information for any dataset and the impact is not really observe, we decide to replace by 1

        elif predicted_label == 1 and true_label == 0:  # Good customer announced as bad
            deficit = amount * rate * duration
            cost += deficit

    return cost
def compute_confusion_matrix(true_labels, predicted_labels, labels):
    """
    Computes the confusion matrix.
    :param true_labels: true labels
    :param predicted_labels: predicted labels
    :param labels:
    :return:
    """
    num_classes = len(labels)
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    for true, predicted in zip(true_labels, predicted_labels):
        true_index = labels.index(true)
        predicted_index = labels.index(predicted)
        confusion_matrix[true_index][predicted_index] += 1

    return confusion_matrix

def precision_macro(confusion_matrix):
    num_classes = len(confusion_matrix)
    precisions = []
    for i in range(num_classes):
        tp = confusion_matrix[i][i]
        fp = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)
        denom = tp + fp
        precisions.append(tp / denom if denom else 0.0)
    return sum(precisions) / num_classes

def recall_macro(confusion_matrix):
    num_classes = len(confusion_matrix)
    recalls = []
    for i in range(num_classes):
        tp = confusion_matrix[i][i]
        fn = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)
        denom = tp + fn
        recalls.append(tp / denom if denom else 0.0)
    return sum(recalls) / num_classes

def f1_macro(confusion_matrix):
    p = precision_macro(confusion_matrix)
    r = recall_macro(confusion_matrix)
    denom = p + r
    return (2 * p * r) / denom if denom else 0.0

def accuracy_macro(confusion_matrix):
    correct = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    total = sum(sum(row) for row in confusion_matrix)
    return correct / total if total else 0.0
