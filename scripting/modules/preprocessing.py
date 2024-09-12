"""
    Author: VICTOR DJIEMBOU
    addedAt: 15/11/2023
    changes:
        - 15/11/2023:
            - add pipeline methods
        - 07/01/2024:
            - ajout d'une personnalisation des modalités de variables catégorielles lig47-49
            - transformer les colonnes en int64 en float64 lig108, lig89
            - optimisation de la borne maximal lors d'une normalisation avec outliers pour eviter les quantile 75 à valeur 0. lig113
            - 
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
from scipy.spatial.distance import euclidean
import math
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
# from imblearn.over_sampling import RandomOverSampler
# import collections
# from memory_profiler import profile

###### End


#################################################
##          Methods definition
#################################################
# @profile
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
        dataframe[col] = dataframe[col].apply(lambda x: x +'__'+ col.replace(' ', '_'))
    ohe.fit(dataframe[variables_list])
    merge_ohe_col = np.concatenate((ohe.categories_)) # list of all new dimension names
    ohe_data = pd.DataFrame(ohe.transform(dataframe[variables_list]).toarray(), columns=merge_ohe_col) # make the one hot encoding and save the result inside a temp source
    dataframe = pd.concat([ohe_data, dataframe], axis=1) #  concat existing and news columns dimensions
    dataframe = dataframe.drop(variables_list, axis=1) # remove all nominal unencoded dimensions
    return (dataframe, ohe.categories_)

# @profile
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

# @profile
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
        dataframe[var] =  dataframe[var].astype('float64')
        maxi = dataframe[var].max()
        dataframe[var] = dataframe[var]/maxi
    return dataframe

# @profile
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
        dataframe[var] =  dataframe[var].astype('float64')
        # dataframe[var].astype('float64', copy=False)
        Q1 = dataframe[var].quantile(0.25)
        Q3 = dataframe[var].quantile(0.75)
        Q3 = dataframe[var].quantile(1) if Q3 == 0 else Q3
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

# @profile
def numeric_vector_is_nominal(series):
    """Use Value count to know if a column identify as numeric one has a nominal logic
    Args:
        series: is the Series vector of column

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

# @profile
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

# @profile
def discretise_numeric_dimension(columns, dataframe, inplace=False, verbose=False):
    """Discretise numerical dimension with quantile boundary
    Args:
        columns: list of numerical column
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
            bins = sorted(list({0, data[col].quantile(0.05), data[col].quantile(0.15), data[col].quantile(0.25),
                                data[col].quantile(0.5), data[col].quantile(0.75), data[col].quantile(0.85),
                                np.float64('inf')}))

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

# @profile
def get_numeric_vector_with_outlier(dataframe):
    """Get numerical columns with outliers
    Args:
        dataframe: is the Series vector of column

    Returns
        The Flag boolean value of decusion
    """
    outliers_columns = []
    for column in get_numerical_columns(dataframe):
        if is_numeric_vector_with_outlier(dataframe[column]):
            outliers_columns.append(column)
    return outliers_columns

# @profile
def get_categorial_columns(dataframe):
    """Use select_dtypes to detect categorial columns
    Args:
        dataframe: is the data matrix

    Returns
        The list of nominal columns
    """
    nominal_columns = None
    if isinstance(dataframe, pd.DataFrame):
        nominal_columns = dataframe.select_dtypes(include=['object'])
    return nominal_columns.columns.tolist()

# @profile
def get_numerical_columns(dataframe):
    """Use select_dtypes to detect numeric columns
    Args:
        dataframe: is the data matrix

    Returns
        The list of nominal columns
    """
    numerical_columns = None
    if isinstance(dataframe, pd.DataFrame):
        numerical_columns = dataframe.select_dtypes(include=['int64', 'float64'])
    return numerical_columns.columns.tolist()

# @profile
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

# @profile
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

# @profile
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

# @profile
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

def random_sample_merge_v2(df, target_column, percentage):
    # Séparation du dataframe en fonction des valeurs de la colonne cible
    # lenDF = df.shape[0]
    groups = df.groupby(target_column)

    # Trouver la classe minoritaire et sa taille
    # minority_class = groups.min()[target_column]
    # minority_size = groups.size().min()
    # print(f"{minority_size} -- {lenDF - minority_size} {percentage}")
    # minority_size_percentage = minority_size * percentage
    
    # Liste pour stocker les échantillons de chaque groupe
    samples = []
    
    # Pour chaque groupe, effectuer un échantillonnage aléatoire du pourcentage spécifié
    for group_name, group_df in groups:
        sample_size = int(group_df.shape[0] * percentage)
        sample = group_df.sample(n=sample_size)
        samples.append(sample)
    
    # Fusionner les échantillons en un seul dataframe
    merged_df = pd.concat(samples)
    
    # Réorganiser l'index du dataframe fusionné
    merged_df.reset_index(drop=True, inplace=True)
    
    return merged_df

def random_sample_merge(df, target_column, percentage):
    # Séparation du dataframe en fonction des valeurs de la colonne cible
    lenDF = df.shape[0]
    groups = df.groupby(target_column)

    # Trouver la classe minoritaire et sa taille
    print(groups.min())
    # minority_class = groups.min()[target_column]
    minority_size = groups.size().min()
    print(f"{minority_size} -- {lenDF - minority_size} {percentage}")
    minority_size_percentage = minority_size * percentage

    # Liste pour stocker les échantillons de chaque groupe
    samples = []

    # Pour chaque groupe, effectuer un échantillonnage aléatoire du pourcentage spécifié
    for group_name, group_df in groups:
        sample_size = int(minority_size_percentage)
        sample = group_df.sample(n=sample_size)
        samples.append(sample)

    # Fusionner les échantillons en un seul dataframe
    merged_df = pd.concat(samples)

    # Réorganiser l'index du dataframe fusionné
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df

def entropy(data):
    """
    Calculates the entropy of a given dataset.

    Args:
        data: The array of class labels.

    Returns:
        The entropy value.
    """
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def information_gain(data, split_point, class_labels):
    """
    Calculates the information gain for a given split point.

    Args:
        data: The array of continuous values.
        split_point: The split point to evaluate.
        class_labels: The array of class labels.

    Returns:
        The information gain value.
    """
    left_data = data[data <= split_point]
    right_data = data[data > split_point]
    total_entropy = entropy(class_labels)
    left_entropy = entropy(class_labels[data <= split_point])
    right_entropy = entropy(class_labels[data > split_point])
    left_weight = len(left_data) / len(data)
    right_weight = len(right_data) / len(data)
    information_gain = total_entropy - (left_weight * left_entropy + right_weight * right_entropy)
    return information_gain

def split_data(data, class_labels, num_bins, threshold):
    """
    Recursively splits the data based on the best split point until termination condition is met.

    Args:
        data: The array of continuous values.
        class_labels: The array of class labels.
        num_bins: The maximum number of bins.
        threshold: The information gain threshold to terminate the splitting.

    Returns:
        A list of split points.
    """
    if len(data) <= num_bins or information_gain(data, data[0], class_labels) < threshold:
        #print('in')
        return []
    else:
        sorted_data = np.sort(data)
        split_points = []
        for i in range(1, len(sorted_data)):
            if sorted_data[i] != sorted_data[i-1]:
                split_point = (sorted_data[i] + sorted_data[i-1]) / 2
                split_points.append(split_point)
        best_split_point = max(split_points, key=lambda x: information_gain(data, x, class_labels))
        left_data = data[data <= best_split_point]
        right_data = data[data > best_split_point]
        return (
            [best_split_point] + 
            split_data(left_data, class_labels[data <= best_split_point], num_bins, threshold) + 
            split_data(right_data, class_labels[data > best_split_point], num_bins, threshold))

def discretize_data(data, class_labels, num_bins, threshold):
    """
    Discretizes the continuous data using entropy-based discretization.

    Args:
        data: The array of continuous values.
        class_labels: The array of class labels.
        num_bins: The maximum number of bins.
        threshold: The information gain threshold to terminate the splitting.

    Returns:
        The discretized data.
    """
    split_points = np.sort(split_data(data, class_labels, num_bins, threshold))
    print(split_points)
    discrete_data = np.digitize(data, split_points)
    return discrete_data

def discretise_numeric_dimension_by_entropy(columns, dataframe, target, inplace=False, verbose=False, bins=3, threshold=.00000001):
    """Discretise numerical dimension with quantile boundary
    Args:
        columns: list of numerical column
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
            dat = discretize_data(data[col].values, data[target], bins, threshold)

            # define discret labels
            labels = [f'{col}___{int(i)}' for i in dat]

            print(f"""
                labels: {set(labels)}
                cols: {col}
                bins: {3}
                types: {data[col].dtype}
                """) if verbose else None

            # apply
            data[col] = labels

    if not inplace:
        return data
