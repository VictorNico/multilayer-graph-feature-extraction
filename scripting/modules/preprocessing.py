"""
    Author: VICTOR DJIEMBOU
    addedAt: 15/11/2023
    changes:
        - 15/11/2023:
            - add pipeline methods
"""

#################################################
##          Libraries importation
#################################################

###### Begin

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from itertools import combinations
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import collections

###### End


#################################################
##          Methods definition
#################################################

def nominal_factor_encoding(data, variables_list):
    """Apply One Hot Encoding (OHE) on ordinal factor dimension
    Args:
      data: A dataframe containing the dimension to standardize
      variables_list: List of dimension on which apply the OHE

    Returns:
      The new dataframe with all dimension standardized.
    """
    dataframe = data.copy(deep=True)
    ohe = OneHotEncoder()
    for col in variables_list:
        print(f"{col} ---<>----- {dataframe[col].unique().tolist()}")
        dataframe[col] = dataframe[col].apply(lambda x: x +'_'+ col.replace(' ', '_'))
    ohe.fit(dataframe[variables_list])
    merge_ohe_col = np.concatenate((ohe.categories_)) # list of all new dimension names
    ohe_data = pd.DataFrame(ohe.transform(dataframe[variables_list]).toarray(), columns=merge_ohe_col) # make the one hot encoding and save the result inside a temp source
    dataframe = pd.concat([ohe_data, dataframe], axis=1) #  concat existing and news columns dimensions
    dataframe = dataframe.drop(variables_list, axis=1) # remove all nominal unencoded dimensions
    return (dataframe, ohe.categories_)

def ordinal_factor_encoding(data, variables_list):
    """Apply LabelEncoding on ordinal factor dimension
    Args:
      data: A dataframe containing the dimension to standardize
      variables_list: List of dimension on which apply the encode

    Returns:
      The new dataframe with all dimension encoded.
    """
    
    dataframe = data.copy(deep=True)
    # 1) for each variable
    for var in variables_list:
        label_encoder = LabelEncoder()
        dataframe[var] = label_encoder.fit_transform(dataframe[var])
    return dataframe

def numeric_uniform_standardization(data, variables_list):
    """Use max division standardize dimension
    Args:
      data: A dataframe containing the dimension to standardize
      variables_list: List of dimension on which apply the standardization

    Returns:
      The new dataframe with all dimension standardized.
    """
    
    dataframe = data.copy(deep=True)
    # 1) for each variable
    for var in variables_list:
        # get maximum value
        maxi = dataframe[var].max()
        dataframe[var] = dataframe[var]/maxi
    return dataframe

def numeric_standardization_with_outliers(data, variables_list):
    """Use IQR to standardize dimension with extrem values
    Args:
      data: A dataframe containing the dimension to standardize
      variables_list: List of dimension on which apply the standardization aware outliers

    Returns:
      The new dataframe with all dimension standardized.
    """
    
    dataframe = data.copy(deep=True)
    # 1) for each variable
    for var in variables_list:
        # a) compute Q1 and Q3
        Q1 = dataframe[var].quantile(0.25)
        Q3 = dataframe[var].quantile(0.75)
        # b) compute IQR
        IQR = Q3 - Q1
        # c) compute sup and inf
        sup = Q3 + (1.5 * IQR)
        inf = Q1 - (1.5 * IQR)
        for line in dataframe.index.values.tolist():
            # if less than inf
            if dataframe.loc[line, var] < inf:
                dataframe.loc[line, var] = inf/sup
            # else greater than sup
            elif dataframe.loc[line, var] > sup:
                dataframe.loc[line, var] = 1
            # else
            else:
                dataframe.loc[line, var] = dataframe.loc[line, var]/sup
    return dataframe

def numeric_vector_is_nominal(series):
    """Use Value count to know if a column identify as numeric one has a nominal logic
    Args:
        Series: is the Series vector of column

    Returns
        The Flag boolean value of decusion
    """
    data = None
    if isinstance(series, pd.Series):
        data = series
    elif isinstance(series, list) or isinstance(series, np.ndarray):
        data = pd.Series(series)
    else:
        return None
    # Compter le nombre d'occurrences de chaque valeur
    occurrences = data.value_counts()

    # Vérifier si toutes les valeurs sont uniques
    suit_serie_nominale = (occurrences == 1).all()

    return suit_serie_nominale

def is_numeric_vector_with_outlier(column):
    """Use IDR to detect outliers
    Args:
        column: is the Series vector of column

    Returns
        The Flag boolean value of decusion
    """
    data = None
    if isinstance(column, pd.Series):
        data = column
    else:
        data = pd.Series(column)
    # Calculer le 1er et le 3e quartile
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    # Calculer l'intervalle interquartile (IQR)
    IQR = Q3 - Q1

    # Calculer les limites supérieure et inférieure pour détecter les valeurs aberrantes
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR

    # Identifier les valeurs aberrantes
    valeurs_aberrantes = sum(data < borne_inf) + sum(data > borne_sup)
    return valeurs_aberrantes > 0

def discretise_numeric_dimension(columns, dataframe, inplace=False, verbose=False):
    """Discretise numerical dimension with quantile boundary
    Args:
        column: list of numerical column
        dataframe: the source dataframe of treatments

    Returns
        The Flag boolean value of decusion
    """

    if inplace:
        data = dataframe
    else:
        data = dataframe.copy(deep=True)


    if isinstance(columns, list) and isinstance(dataframe, pd.DataFrame):
        for col in columns:
            # define boundaries
            bins = sorted(list(set([
                0, 
                data[col].quantile(0.05), 
                data[col].quantile(0.15), 
                data[col].quantile(0.25), 
                data[col].quantile(0.5), 
                data[col].quantile(0.75),
                data[col].quantile(0.85),
                np.float('inf')
                ])))

            # discretize the column
            discretizer = KBinsDiscretizer(n_bins=len(bins)+1, encode='ordinal', strategy='quantile')
            discretized_data = discretizer.fit_transform(data[[col]])


            # define discret labels
            labels = [f'{col}_{int(i[0])}' for i in discretized_data]

            print(f"""
                labels: {set(labels)}
                cols: {col}
                bins: {bins}
                types: {data[col].dtype}
                """) if verbose else None

            # apply
            data[col] = labels

    if not inplace:
        return data

def get_numeric_vector_with_outlier(dataframe):
    """Get numerical columns with outliers
    Args:
        column: is the Series vector of column

    Returns
        The Flag boolean value of decusion
    """
    outliers_columns = []
    for column in get_numerical_columns(dataframe):
        if is_numeric_vector_with_outlier(dataframe[column]):
            outliers_columns.append(column)
    return outliers_columns

def get_categorial_columns(dataframe):
    """Use select_dtypes to detect categorial columns
    Args:
        data: is the data matrix

    Returns
        The list of nominal columns
    """
    nominal_columns = None
    if isinstance(dataframe, pd.DataFrame):
        nominal_columns = dataframe.select_dtypes(include=['object'])
    return nominal_columns.columns.tolist()

def get_numerical_columns(dataframe):
    """Use select_dtypes to detect numeric columns
    Args:
        data: is the data matrix

    Returns
        The list of nominal columns
    """
    numerical_columns = None
    if isinstance(dataframe, pd.DataFrame):
        numerical_columns = dataframe.select_dtypes(include=['int64', 'float64'])
    return numerical_columns.columns.tolist()

def get_combinations(data, k):
    """Use combinations to denerate the list of layer to be build later
    Args:
        data: is the list of column
        k: the number of layer

    Returns
        The list of combination
    """
    
    # Générer toutes les combinaisons de taille k
    comb = combinations(data, k)
    
    # Convertir l'objet combinaisons en une liste de tuples
    combinations_list = list(comb)
    uniques = []
    
    # Ne recupérer que les ensembles unique
    for elt in combinations_list:
        uniques.append(set(elt))

    combinations_list = []
    uniques = np.unique(uniques)
    for elt in uniques:
        combinations_list.append(list(elt)) if len(elt) == k else None
    
    return combinations_list

def is_ordinal(column):
    """Detect if a column has ordinal object values
    Args:
        column: a dimension fo dataset 
    Returns
        True if the case
    """
    unique_values = column.unique()
    sorted_values = sorted(unique_values)
    return list(unique_values) == sorted_values or list(unique_values) == sorted_values[::-1]

def get_ordinal_columns(dataframe):
    """Get the list of Ordinal dimension in dataframe
    Args:
        dataframe: a pandas dataframe
    Returns
        The list of Ordinal column
    """
    ordinal_columns = []
    for column in dataframe.columns:
        if is_ordinal(dataframe[column]):
            ordinal_columns.append(column)
    return ordinal_columns

def get_SMOTHE_dataset(X, y, random_state=42, sampling_strategy='auto'):
    """Compute a SMOTHED dataset 
    Args:
        X: borrowers info
        y: type debt
        random_state: factor that allow us to get always the same sample on each call
        sampling_strategy: sampling strategy, can take value such as 'minority', 'not majority' or a dict of class to sample

    Returns
        A balanced borrowers info and classes
    """

    # print(f"""
    #     columns: {dataframe.columns.tolist()}
    #     x: {X.columns.tolist()}
    #     y: {y.columns.tolist()}
    #     target: {target_variable}
    #     dataframe: {isinstance(dataframe, pd.DataFrame)}
    #     """)
    # X = dataframe.drop([target_variable], axis=1)
    # y = dataframe.loc[:,[target_variable]]    # class boosting model
    oversampler = SMOTE(random_state= random_state, sampling_strategy= sampling_strategy)
    # Data for smoting
    X_r, y_r = oversampler.fit_resample(X, y)
    # DATA_OVER = X_r
    # DATA_OVER[target_variable] = y_r

    return X_r, y_r