"""
=============================================================================
Module de Prétraitement de Données pour Machine Learning
=============================================================================

Auteur: VICTOR DJIEMBOU
Date de création: 15/11/2023
Dernière modification: 07/01/2024

Description:
    Ce module fournit un ensemble complet de fonctions pour le prétraitement
    de données dans le cadre d'expérimentations de Machine Learning.
    Il gère l'encodage de variables, la normalisation, la discrétisation,
    et l'équilibrage de classes.

    Fonctionnalités principales:
    - Encodage de variables catégorielles (One-Hot, Label Encoding)
    - Normalisation de variables numériques (uniforme, avec outliers)
    - Détection automatique de types de variables
    - Discrétisation supervisée et non-supervisée
    - Équilibrage de classes (SMOTE, sampling)
    - Détection de valeurs aberrantes (outliers)
    - Génération de combinaisons de features

Dépendances:
    - pandas: Manipulation de DataFrames
    - numpy: Calculs numériques
    - sklearn: Encoders et discrétiseurs
    - imblearn: Techniques de rééchantillonnage (SMOTE)
    - scipy: Calculs de distances

Historique des modifications:
    - 15/11/2023: Ajout des méthodes de pipeline
    - 07/01/2024:
        * Personnalisation des modalités de variables catégorielles (lig 47-49)
        * Transformation int64 → float64 (lig 108, 89)
        * Optimisation borne max avec outliers (quantile 75 à 0) (lig 113)

=============================================================================
"""

#################################################
##          Importation des bibliothèques
#################################################

###### Début

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from itertools import combinations
from imblearn.over_sampling import SMOTE
from scipy.spatial.distance import euclidean
import math

# Imports commentés (non utilisés actuellement)
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
# from imblearn.over_sampling import RandomOverSampler
# import collections
# from memory_profiler import profile

###### Fin


#################################################
##          Définition des méthodes
#################################################

# @profile  # Décorateur pour profilage mémoire (commenté)
def nominal_factor_encoding(data, variables_list):
    """
    Applique l'encodage One-Hot (OHE) sur les variables catégorielles nominales.

    L'encodage One-Hot transforme chaque catégorie d'une variable en une colonne
    binaire (0 ou 1). Cette méthode est appropriée pour les variables nominales
    sans ordre naturel (ex: couleurs, pays, types).

    Args:
        data (pandas.DataFrame): DataFrame contenant les variables à encoder
        variables_list (list): Liste des noms de colonnes à encoder

    Returns:
        tuple: (DataFrame encodé, dictionnaire des catégories par variable)
            - DataFrame: Nouvelles colonnes binaires + colonnes originales non encodées
            - dict: Catégories créées pour chaque variable (pour traçabilité)

    Exemple:
        >>> df = pd.DataFrame({'couleur': ['rouge', 'bleu', 'rouge']})
        >>> encoded_df, categories = nominal_factor_encoding(df, ['couleur'])
        >>> print(encoded_df.columns)
        ['rouge__couleur', 'bleu__couleur', ...]

    Traitement spécial:
        - Variables avec une seule valeur unique sont ignorées (pas d'encodage)
        - Préfixage des valeurs avec le nom de colonne pour éviter les collisions
          (ex: 'rouge' devient 'rouge__couleur')

    Modification 07/01/2024:
        Ajout du préfixage pour rendre les modalités uniques entre colonnes
    """
    # Copie profonde pour ne pas modifier le DataFrame original
    dataframe = data.copy(deep=True)

    # Initialisation de l'encodeur One-Hot
    ohe = OneHotEncoder()

    # ============================================================
    # PRÉFIXAGE DES VALEURS CATÉGORIELLES
    # ============================================================
    # Pour éviter les collisions entre colonnes ayant des modalités identiques
    # Ex: 'Oui' dans colonne 'Fumeur' vs 'Oui' dans colonne 'Diabétique'
    for col in variables_list:
        # Vérification: colonne avec une seule valeur unique
        if dataframe[col].nunique() <= 1:
            print(f"Skipping column '{col}' (only one unique value: {dataframe[col].unique()})")
            # Retrait de la colonne de la liste d'encodage
            variables_list = [v for v in variables_list if v != col]
        else:
            # Affichage des valeurs uniques avant encodage
            print(f"{col} ---<>----- {dataframe[col].unique().tolist()}")

            # Préfixage: 'rouge' → 'rouge__couleur'
            # replace(' ', '_') pour éviter les espaces dans les noms de colonnes
            dataframe[col] = dataframe[col].apply(lambda x: f"{x}__{col.replace(' ', '_')}")

    # Vérification: si aucune colonne valide à encoder
    if not variables_list:
        print("No valid column to encode. Returning original dataframe.")
        return dataframe, {}

    # ============================================================
    # ENCODAGE ONE-HOT
    # ============================================================
    # Apprentissage des catégories à partir des données
    ohe.fit(dataframe[variables_list])

    # Extraction de toutes les nouvelles dimensions (noms de colonnes binaires)
    merge_ohe_col = np.concatenate((ohe.categories_))

    # Transformation des données en matrice binaire
    ohe_data = pd.DataFrame(
        ohe.transform(dataframe[variables_list]).toarray(),
        columns=merge_ohe_col
    )

    # Concaténation: colonnes encodées + colonnes non encodées
    dataframe = pd.concat([ohe_data, dataframe], axis=1)

    # Suppression des colonnes originales (maintenant encodées)
    dataframe = dataframe.drop(variables_list, axis=1)

    return (dataframe, ohe.categories_)


# @profile
def ordinal_factor_encoding(data, variables_list):
    """
    Applique l'encodage Label sur les variables catégorielles ordinales.

    Le Label Encoding transforme chaque catégorie en un entier (0, 1, 2, ...).
    Cette méthode préserve l'ordre naturel des catégories (ex: 'faible' < 'moyen' < 'élevé').

    Args:
        data (pandas.DataFrame): DataFrame contenant les variables à encoder
        variables_list (list): Liste des noms de colonnes ordinales

    Returns:
        list: [DataFrame encodé, dictionnaire de mapping]
            - DataFrame: Colonnes transformées en entiers
            - dict: Mapping {variable: {catégorie: code}} pour chaque variable

    Exemple:
        >>> df = pd.DataFrame({'niveau': ['faible', 'moyen', 'élevé', 'moyen']})
        >>> encoded_df, mapping = ordinal_factor_encoding(df, ['niveau'])
        >>> print(mapping['niveau'])
        {'faible': 0, 'moyen': 1, 'élevé': 2}

    Avantage vs One-Hot:
        - Moins de colonnes créées (1 colonne encodée vs n colonnes binaires)
        - Préserve l'ordre naturel des catégories
        - Mieux adapté aux algorithmes sensibles à l'ordre (arbres de décision)

    Note:
        L'ordre d'encodage est déterminé par l'ordre alphabétique des catégories.
        Pour un ordre personnalisé, utiliser OrdinalEncoder avec categories explicites.
    """
    # Copie profonde du DataFrame
    dataframe = data.copy(deep=True)

    # Dictionnaire pour stocker les mappings de chaque variable
    label_enc = {}

    # Traitement de chaque variable ordinale
    for var in variables_list:
        # Initialisation d'un nouveau LabelEncoder pour cette variable
        label_encoder = LabelEncoder()

        # Fit + Transform: apprentissage et transformation en une seule étape
        dataframe[var] = label_encoder.fit_transform(dataframe[var])

        # Sauvegarde du mapping catégorie → code
        # Ex: {'faible': 0, 'moyen': 1, 'élevé': 2}
        label_enc[var] = dict(
            zip(label_encoder.classes_,
                label_encoder.transform(label_encoder.classes_))
        )

    return [dataframe, label_enc]


# @profile
def numeric_uniform_standardization(data, variables_list):
    """
    Normalise les variables numériques par division par le maximum.

    Cette méthode de normalisation simple ramène toutes les valeurs dans [0, 1]
    en divisant par la valeur maximale de chaque variable.

    Args:
        data (pandas.DataFrame): DataFrame contenant les variables à normaliser
        variables_list (list): Liste des noms de colonnes numériques

    Returns:
        pandas.DataFrame: DataFrame avec variables normalisées dans [0, 1]

    Exemple:
        >>> df = pd.DataFrame({'age': [20, 40, 60, 80]})
        >>> normalized = numeric_uniform_standardization(df, ['age'])
        >>> print(normalized['age'].tolist())
        [0.25, 0.5, 0.75, 1.0]  # Divisé par max=80

    Formule:
        x_norm = x / max(x)

    Avantages:
        - Simple et rapide
        - Préserve les ratios entre valeurs
        - Valeur max devient toujours 1.0

    Inconvénients:
        - Sensible aux outliers (valeurs extrêmes)
        - Ne centre pas les données autour de 0
        - Pas adapté si max est une valeur aberrante

    Quand l'utiliser:
        - Variables sans outliers significatifs
        - Quand la valeur max est représentative
        - Pour des features déjà dans une plage raisonnable

    Modification 07/01/2024:
        Conversion explicite en float64 pour éviter les problèmes de types
    """
    # Copie profonde du DataFrame
    dataframe = data.copy(deep=True)

    # Normalisation de chaque variable
    for var in variables_list:
        # Conversion en float64 pour précision des calculs
        # Évite les problèmes avec int64 lors de la division
        dataframe[var] = dataframe[var].astype('float64')

        # Calcul de la valeur maximale
        maxi = dataframe[var].max()

        # Normalisation: division par le maximum
        # Résultat: toutes valeurs ∈ [0, 1], avec max=1
        dataframe[var] = dataframe[var] / maxi

    return dataframe


# @profile
def numeric_standardization_with_outliers(data, variables_list):
    """
    Normalise les variables numériques en gérant les valeurs aberrantes (outliers).

    Cette méthode utilise l'Intervalle Inter-Quartile (IQR) pour détecter et
    borner les outliers, puis normalise les valeurs dans [0, 1].

    Args:
        data (pandas.DataFrame): DataFrame contenant les variables à normaliser
        variables_list (list): Liste des colonnes numériques avec outliers

    Returns:
        pandas.DataFrame: DataFrame avec variables normalisées et outliers bornés

    Algorithme:
        1. Calculer Q1 (quartile 25%) et Q3 (quartile 75%)
        2. Calculer IQR = Q3 - Q1
        3. Définir bornes: inf = Q1 - 1.5×IQR, sup = Q3 + 1.5×IQR
        4. Pour chaque valeur:
           - Si < inf: normaliser à inf/sup
           - Si > sup: normaliser à 1.0
           - Sinon: normaliser à valeur/sup

    Exemple:
        >>> df = pd.DataFrame({'salaire': [30000, 35000, 40000, 1000000]})
        >>> # 1000000 est un outlier, sera normalisé à 1.0
        >>> normalized = numeric_standardization_with_outliers(df, ['salaire'])

    Avantages:
        - Robuste aux outliers extrêmes
        - Préserve l'information de distribution
        - Outliers sont bornés, pas supprimés

    Modification 07/01/2024:
        Optimisation pour Q3=0: utilise le maximum (quantile 100%) au lieu de 0
        pour éviter les divisions par zéro
    """
    # Copie profonde du DataFrame
    dataframe = data.copy(deep=True)

    # Traitement de chaque variable
    for var in variables_list:
        # Conversion en float64 pour précision
        dataframe[var] = dataframe[var].astype('float64')

        # ============================================================
        # CALCUL DES QUARTILES ET IQR
        # ============================================================
        # Q1: 25% des valeurs sont en dessous
        Q1 = dataframe[var].quantile(0.25)

        # Q3: 75% des valeurs sont en dessous
        Q3 = dataframe[var].quantile(0.75)

        # MODIFICATION 07/01/2024: Gestion du cas Q3=0
        # Si Q3=0, utiliser le maximum pour éviter division par zéro
        Q3 = dataframe[var].quantile(1) if Q3 == 0 else Q3

        # IQR: Intervalle Inter-Quartile (mesure de dispersion)
        IQR = Q3 - Q1

        # ============================================================
        # DÉFINITION DES BORNES POUR OUTLIERS
        # ============================================================
        # Bornes selon la règle de Tukey (1.5 × IQR)
        sup = Q3 + (1.5 * IQR)  # Borne supérieure
        inf = Q1 - (1.5 * IQR)  # Borne inférieure

        # ============================================================
        # NORMALISATION AVEC GESTION DES OUTLIERS
        # ============================================================
        # Traitement ligne par ligne (pourrait être vectorisé pour performance)
        for line in dataframe.index.values.tolist():
            # Cas 1: Outlier inférieur (valeur < inf)
            if dataframe.loc[line, var] < inf:
                dataframe.loc[line, var] = inf / sup

            # Cas 2: Outlier supérieur (valeur > sup)
            elif dataframe.loc[line, var] > sup:
                dataframe.loc[line, var] = 1  # Plafonné à 1.0

            # Cas 3: Valeur normale (inf ≤ valeur ≤ sup)
            else:
                dataframe.loc[line, var] = dataframe.loc[line, var] / sup

    return dataframe


# @profile
def numeric_vector_is_nominal(series):
    """
    Détermine si une colonne numérique a une logique nominale (identifiants).

    Une colonne numérique est nominale si toutes ses valeurs sont uniques,
    ce qui suggère qu'elle représente des IDs plutôt que des mesures.

    Args:
        series (pandas.Series, list, ou numpy.ndarray): Série à analyser

    Returns:
        bool: True si toutes valeurs uniques (logique nominale)
        None: Si type non supporté

    Exemple:
        >>> ids = pd.Series([1001, 1002, 1003])  # IDs uniques
        >>> numeric_vector_is_nominal(ids)
        True

        >>> ages = pd.Series([25, 30, 25, 35])  # Valeurs répétées
        >>> numeric_vector_is_nominal(ages)
        False

    Cas d'usage:
        Identifier les colonnes à ne PAS normaliser (IDs, matricules, etc.)
    """
    data = None

    # Conversion selon le type d'entrée
    if isinstance(series, pd.Series):
        data = series
    elif isinstance(series, list) or isinstance(series, np.ndarray):
        data = pd.Series(series)
    else:
        return None

    # Comptage des occurrences de chaque valeur
    occurrences = data.value_counts()

    # Vérification: toutes les valeurs apparaissent exactement 1 fois?
    suit_serie_nominale = (occurrences == 1).all()

    return suit_serie_nominale


# @profile
def is_numeric_vector_with_outlier(column):
    """
    Détecte la présence d'outliers dans une colonne numérique via l'IQR.

    Utilise la méthode de l'Intervalle Inter-Quartile (IQR) pour identifier
    les valeurs aberrantes selon la règle de Tukey.

    Args:
        column (pandas.Series ou list): Colonne à analyser

    Returns:
        bool: True si au moins un outlier est détecté, False sinon

    Algorithme:
        1. Calculer Q1 et Q3
        2. Calculer IQR = Q3 - Q1
        3. Définir bornes: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
        4. Compter les valeurs hors bornes

    Exemple:
        >>> data = pd.Series([10, 12, 11, 13, 100])  # 100 est outlier
        >>> is_numeric_vector_with_outlier(data)
        True

    Règle de Tukey:
        Valeurs > Q3 + 1.5×IQR ou < Q1 - 1.5×IQR sont des outliers
    """
    data = None

    # Conversion en Series si nécessaire
    if isinstance(column, pd.Series):
        data = column
    else:
        data = pd.Series(column)

    # Calcul des quartiles
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    # Calcul de l'IQR
    IQR = Q3 - Q1

    # Définition des bornes de détection
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR

    # Comptage des valeurs aberrantes
    valeurs_aberrantes = sum(data < borne_inf) + sum(data > borne_sup)

    return valeurs_aberrantes > 0


# @profile
def discretise_numeric_dimension(columns, dataframe, inplace=False, verbose=False):
    """
    Discrétise des variables numériques continues en catégories ordinales.

    Transforme des variables numériques en variables catégorielles en utilisant
    des frontières de quantiles prédéfinis.

    Args:
        columns (list): Liste des colonnes numériques à discrétiser
        dataframe (pandas.DataFrame): DataFrame source
        inplace (bool): Si True, modifie le DataFrame original
        verbose (bool): Si True, affiche des informations de débogage

    Returns:
        pandas.DataFrame ou None: DataFrame discrétisé (si inplace=False)

    Quantiles utilisés:
        0, 5%, 15%, 25%, 50%, 75%, 85%, inf

    Exemple:
        >>> df = pd.DataFrame({'age': [20, 30, 40, 50, 60, 70, 80]})
        >>> discretized = discretise_numeric_dimension(['age'], df)
        >>> # 'age' devient 'age_0', 'age_1', 'age_2', etc.

    Labels générés:
        Format: 'nom_colonne_code_bin'
        Ex: 'age_0', 'age_1', 'age_2'
    """
    # Gestion du mode inplace
    if inplace:
        data = dataframe
    else:
        data = dataframe.copy(deep=True)

    # Vérification des types d'entrée
    if isinstance(columns, list) and isinstance(dataframe, pd.DataFrame):
        for col in columns:
            # ============================================================
            # DÉFINITION DES FRONTIÈRES (BINS)
            # ============================================================
            # Utilisation de quantiles pour créer des bins équilibrés
            # sorted() et set() pour éliminer les doublons et trier
            bins = sorted(list({
                0,
                data[col].quantile(0.05),   # 5e percentile
                data[col].quantile(0.15),   # 15e percentile
                data[col].quantile(0.25),   # Q1
                data[col].quantile(0.5),    # Médiane
                data[col].quantile(0.75),   # Q3
                data[col].quantile(0.85),   # 85e percentile
                np.float64('inf')            # Infini pour borne supérieure
            }))

            # ============================================================
            # DISCRÉTISATION
            # ============================================================
            # KBinsDiscretizer avec stratégie 'quantile'
            discretizer = KBinsDiscretizer(
                n_bins=len(bins) + 1,
                encode='ordinal',
                strategy='quantile'
            )

            # Transformation de la colonne
            discretized_data = discretizer.fit_transform(data[[col]])

            # ============================================================
            # GÉNÉRATION DES LABELS
            # ============================================================
            # Création de labels catégoriels pour chaque bin
            # Format: 'nom_colonne_code_entier'
            labels = [f'{col}_{int(i[0])}' for i in discretized_data]

            # Affichage en mode verbeux
            print(f"""
                labels: {set(labels)}
                cols: {col}
                bins: {bins}
                types: {data[col].dtype}
                """) if verbose else None

            # Remplacement des valeurs numériques par les labels
            data[col] = labels

    # Retour selon le mode inplace
    if not inplace:
        return data


# @profile
def get_numeric_vector_with_outlier(dataframe):
    """
    Identifie toutes les colonnes numériques contenant des outliers.

    Args:
        dataframe (pandas.DataFrame): DataFrame à analyser

    Returns:
        list: Noms des colonnes numériques avec outliers

    Exemple:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 1000]})
        >>> get_numeric_vector_with_outlier(df)
        ['b']  # Colonne 'b' contient 1000 comme outlier
    """
    outliers_columns = []

    # Test de chaque colonne numérique
    for column in get_numerical_columns(dataframe):
        if is_numeric_vector_with_outlier(dataframe[column]):
            outliers_columns.append(column)

    return outliers_columns


# @profile
def get_categorial_columns(dataframe):
    """
    Identifie les colonnes catégorielles (type 'object').

    Args:
        dataframe (pandas.DataFrame): DataFrame à analyser

    Returns:
        list: Noms des colonnes de type 'object'

    Note:
        Pandas utilise le dtype 'object' pour les strings et catégories
    """
    nominal_columns = None

    if isinstance(dataframe, pd.DataFrame):
        # Sélection des colonnes de type 'object'
        nominal_columns = dataframe.select_dtypes(include=['object'])

    return nominal_columns.columns.tolist()


# @profile
def get_numerical_columns(dataframe):
    """
    Identifie les colonnes numériques (int64, float64).

    Args:
        dataframe (pandas.DataFrame): DataFrame à analyser

    Returns:
        list: Noms des colonnes numériques

    Note:
        Sélectionne uniquement int64 et float64, pas int32 ou autres variants
    """
    numerical_columns = None

    if isinstance(dataframe, pd.DataFrame):
        # Sélection des colonnes numériques
        numerical_columns = dataframe.select_dtypes(include=['int64', 'float64'])

    return numerical_columns.columns.tolist()


# @profile
def get_combinations(data, k):
    """
    Génère toutes les combinaisons uniques de k éléments depuis une liste.

    Utilisé pour créer des couches (layers) de features dans les réseaux multiniveaux.

    Args:
        data (list): Liste d'éléments (colonnes, features, etc.)
        k (int): Taille des combinaisons

    Returns:
        list: Liste de combinaisons (chaque combinaison est une liste)

    Exemple:
        >>> columns = ['A', 'B', 'C']
        >>> get_combinations(columns, 2)
        [['A', 'B'], ['A', 'C'], ['B', 'C']]

    Note:
        Les doublons sont éliminés via np.unique() sur les sets
    """
    # Génération de toutes les combinaisons de taille k
    comb = combinations(data, k)

    # Conversion en liste de tuples
    combinations_list = list(comb)

    # Conversion en sets pour éliminer les doublons
    uniques = []
    for elt in combinations_list:
        uniques.append(set(elt))

    # Déduplication via np.unique
    combinations_list = []
    uniques = np.unique(uniques)

    # Filtrage: ne garder que les ensembles de taille k
    for elt in uniques:
        combinations_list.append(list(elt)) if len(elt) == k else None

    return combinations_list


# @profile
def is_ordinal(column):
    """
    Détecte si une colonne a des valeurs ordinales (ordre naturel).

    Args:
        column (pandas.Series): Colonne à tester

    Returns:
        bool: True si les valeurs sont déjà triées (croissant ou décroissant)

    Exemple:
        >>> data = pd.Series(['faible', 'moyen', 'élevé'])
        >>> is_ordinal(data)
        True  # Si ordre alphabétique correspond à ordre logique
    """
    unique_values = column.unique()
    sorted_values = sorted(unique_values)

    # Test: ordre croissant OU décroissant
    return (list(unique_values) == sorted_values or
            list(unique_values) == sorted_values[::-1])


# @profile
def get_ordinal_columns(dataframe):
    """
    Identifie toutes les colonnes ordinales dans un DataFrame.

    Args:
        dataframe (pandas.DataFrame): DataFrame à analyser

    Returns:
        list: Noms des colonnes ordinales

    Note:
        Utilise is_ordinal() pour chaque colonne
    """
    ordinal_columns = []

    for column in dataframe.columns:
        if is_ordinal(dataframe[column]):
            ordinal_columns.append(column)

    return ordinal_columns


# @profile
def get_SMOTHE_dataset(X, y, random_state=42, sampling_strategy='auto'):
    """
    Équilibre un dataset déséquilibré avec la technique SMOTE.

    SMOTE (Synthetic Minority Over-sampling Technique) génère des exemples
    synthétiques pour les classes minoritaires en interpolant entre exemples existants.

    Args:
        X (pandas.DataFrame ou numpy.ndarray): Features
        y (pandas.Series ou numpy.ndarray): Labels de classe
        random_state (int): Seed pour reproductibilité
        sampling_strategy (str ou dict): Stratégie de rééchantillonnage
            - 'auto' ou 'minority': rééchantillonner la classe minoritaire
            - 'not majority': toutes sauf la majoritaire
            - dict: spécifier le nombre d'exemples par classe

    Returns:
        tuple: (X_resampled, y_resampled)
            Features et labels équilibrés

    Exemple:
        >>> X = pd.DataFrame({'f1': [1, 2, 3, 4], 'f2': [5, 6, 7, 8]})
        >>> y = pd.Series([0, 0, 0, 1])  # Déséquilibré: 3 vs 1
        >>> X_bal, y_bal = get_SMOTHE_dataset(X, y)
        >>> print(y_bal.value_counts())
        0    3
        1    3  # Maintenant équilibré

    Avantages SMOTE:
        - Génère des exemples synthétiques (pas de duplication)
        - Améliore les performances sur classes minoritaires
        - Réduit le biais vers les classes majoritaires

    Note:
        SMOTE peut générer du bruit si les classes se chevauchent
    """
    # Initialisation de l'oversampler SMOTE
    oversampler = SMOTE(
        random_state=random_state,
        sampling_strategy=sampling_strategy
    )

    # Rééchantillonnage des données
    X_r, y_r = oversampler.fit_resample(X, y)

    return X_r, y_r


def random_sample_merge_v2(df, target_column, percentage, min_per_class=2):
    """
    Échantillonne aléatoirement un pourcentage de chaque classe.

    Version améliorée garantissant un minimum d'exemples par classe.

    Args:
        df (pandas.DataFrame): DataFrame à échantillonner
        target_column (str): Nom de la colonne de classe
        percentage (float): Proportion à échantillonner (0.0 à 1.0)
        min_per_class (int): Nombre minimum d'exemples par classe

    Returns:
        pandas.DataFrame: DataFrame échantillonné avec index réinitialisé

    Exemple:
        >>> df = pd.DataFrame({'X': range(100), 'y': [0]*50 + [1]*50})
        >>> sample = random_sample_merge_v2(df, 'y', 0.2, min_per_class=5)
        >>> # Au moins 5 exemples de chaque classe, même si 20% < 5

    Logique:
        Pour chaque classe:
        sample_size = max(len(classe) × percentage, min_per_class)
    """
    # Groupement par classe
    groups = df.groupby(target_column)

    # Liste pour stocker les échantillons
    samples = []

    # Échantillonnage de chaque groupe
    for group_name, group_df in groups:
        # Calcul de la taille de l'échantillon
        sample_size = max(int(len(group_df) * percentage), min_per_class)

        # Sécurité: ne pas dépasser la taille du groupe
        sample_size = min(sample_size, len(group_df))

        # Échantillonnage aléatoire avec seed fixe
        sample = group_df.sample(n=sample_size, random_state=42)
        samples.append(sample)

    # Fusion des échantillons
    merged_df = pd.concat(samples)

    # Réinitialisation de l'index
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df


def random_sample_merge(df, target_column, percentage):
    """
    Échantillonne proportionnellement à la classe minoritaire.

    Version antérieure qui base la taille d'échantillon sur la classe minoritaire.

    Args:
        df (pandas.DataFrame): DataFrame à échantillonner
        target_column (str): Colonne de classe
        percentage (float): Proportion de la classe minoritaire

    Returns:
        pandas.DataFrame: DataFrame échantillonné

    Logique:
        1. Trouver la taille de la classe minoritaire
        2. Calculer: sample_size = minority_size × percentage
        3. Échantillonner sample_size exemples de CHAQUE classe

    Note:
        Toutes les classes ont la même taille d'échantillon,
        basée sur la classe minoritaire
    """
    lenDF = df.shape[0]
    groups = df.groupby(target_column)

    # Taille de la classe minoritaire
    print(groups.min())
    minority_size = groups.size().min()
    print(f"{minority_size} -- {lenDF - minority_size} {percentage}")

    # Taille d'échantillon basée sur la minoritaire
    minority_size_percentage = minority_size * percentage

    samples = []

    # Échantillonnage uniforme pour toutes les classes
    for group_name, group_df in groups:
        sample_size = int(minority_size_percentage)
        sample = group_df.sample(n=sample_size)
        samples.append(sample)

    # Fusion et réinitialisation
    merged_df = pd.concat(samples)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df


# ============================================================
# FONCTIONS POUR DISCRÉTISATION SUPERVISÉE (ENTROPIE)
# ============================================================

def entropy(data):
    """
    Calcule l'entropie d'un ensemble de labels de classe.

    L'entropie mesure l'impureté ou le désordre dans un ensemble de données.
    Une entropie élevée indique une distribution uniforme des classes.

    Args:
        data (numpy.ndarray): Array de labels de classe

    Returns:
        float: Valeur d'entropie (bits d'information)

    Formule:
        H(S) = -Σ p(i) × log2(p(i))
        où p(i) est la probabilité de la classe i

    Exemple:
        >>> labels = np.array([0, 0, 1, 1])  # 50% classe 0, 50% classe 1
        >>> entropy(labels)
        1.0  # Entropie maximale pour 2 classes

        >>> labels = np.array([0, 0, 0, 1])  # 75% classe 0, 25% classe 1
        >>> entropy(labels)
        0.811  # Moins d'entropie (plus d'ordre)
    """
    # Comptage des occurrences de chaque classe
    _, counts = np.unique(data, return_counts=True)

    # Calcul des probabilités
    probabilities = counts / len(data)

    # Formule de l'entropie de Shannon
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def information_gain(data, split_point, class_labels):
    """
    Calcule le gain d'information pour un point de découpe donné.

    Le gain d'information mesure la réduction d'entropie après un split.
    Un gain élevé indique que le split sépare bien les classes.

    Args:
        data (numpy.ndarray): Array de valeurs continues
        split_point (float): Point de découpe à évaluer
        class_labels (numpy.ndarray): Array de labels de classe correspondants

    Returns:
        float: Valeur du gain d'information

    Formule:
        IG = H(parent) - [pL × H(left) + pR × H(right)]
        où:
        - H(parent): entropie avant le split
        - H(left), H(right): entropies après le split
        - pL, pR: proportions des données à gauche/droite

    Exemple:
        >>> data = np.array([1, 2, 3, 4])
        >>> labels = np.array([0, 0, 1, 1])
        >>> information_gain(data, 2.5, labels)
        1.0  # Split parfait: 0,0 | 1,1
    """
    # Séparation des données selon le point de découpe
    left_data = data[data <= split_point]
    right_data = data[data > split_point]

    # Entropie totale avant le split
    total_entropy = entropy(class_labels)

    # Entropies des sous-ensembles gauche et droit
    left_entropy = entropy(class_labels[data <= split_point])
    right_entropy = entropy(class_labels[data > split_point])

    # Poids des sous-ensembles (proportions)
    left_weight = len(left_data) / len(data)
    right_weight = len(right_data) / len(data)

    # Calcul du gain d'information
    information_gain = total_entropy - (left_weight * left_entropy +
                                       right_weight * right_entropy)

    return information_gain


def split_data(data, class_labels, num_bins, threshold):
    """
    Divise récursivement les données basé sur le gain d'information.

    Algorithme de discrétisation supervisée qui trouve les meilleurs points
    de découpe pour séparer les classes.

    Args:
        data (numpy.ndarray): Valeurs continues à discrétiser
        class_labels (numpy.ndarray): Labels de classe correspondants
        num_bins (int): Nombre maximum de bins
        threshold (float): Seuil de gain d'information pour arrêter

    Returns:
        list: Liste de points de découpe

    Algorithme récursif:
        1. Si conditions d'arrêt: retourner []
        2. Sinon:
           a. Trouver tous les points de découpe potentiels
           b. Sélectionner celui avec le plus grand gain d'information
           c. Récursivement diviser les sous-ensembles gauche et droit

    Conditions d'arrêt:
        - Moins de num_bins exemples
        - Gain d'information < threshold

    Exemple:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        >>> split_data(data, labels, 10, 0.01)
        [4.5]  # Split entre 4 et 5
    """
    # Condition d'arrêt: trop peu de données ou gain insuffisant
    if (len(data) <= num_bins or
        information_gain(data, data[0], class_labels) < threshold):
        return []
    else:
        # Tri des données
        sorted_data = np.sort(data)

        # Génération des points de découpe potentiels
        # (milieu entre chaque paire de valeurs adjacentes différentes)
        split_points = []
        for i in range(1, len(sorted_data)):
            if sorted_data[i] != sorted_data[i-1]:
                split_point = (sorted_data[i] + sorted_data[i-1]) / 2
                split_points.append(split_point)

        # Sélection du meilleur point de découpe
        best_split_point = max(
            split_points,
            key=lambda x: information_gain(data, x, class_labels)
        )

        # Séparation des données
        left_data = data[data <= best_split_point]
        right_data = data[data > best_split_point]

        # Récursion sur les sous-ensembles
        return (
            [best_split_point] +
            split_data(left_data, class_labels[data <= best_split_point],
                      num_bins, threshold) +
            split_data(right_data, class_labels[data > best_split_point],
                      num_bins, threshold)
        )


def discretize_data(data, class_labels, num_bins, threshold):
    """
    Discrétise des données continues en utilisant l'entropie.

    Wrapper pour split_data() qui retourne les données discrétisées.

    Args:
        data (numpy.ndarray): Valeurs continues
        class_labels (numpy.ndarray): Labels de classe
        num_bins (int): Nombre max de bins
        threshold (float): Seuil de gain d'information

    Returns:
        numpy.ndarray: Données discrétisées (codes entiers)

    Exemple:
        >>> data = np.array([1, 2, 3, 10, 11, 12])
        >>> labels = np.array([0, 0, 0, 1, 1, 1])
        >>> discretize_data(data, labels, 5, 0.01)
        array([0, 0, 0, 1, 1, 1])  # 2 bins séparant les classes
    """
    # Obtention des points de découpe
    split_points = np.sort(split_data(data, class_labels, num_bins, threshold))
    print(split_points)

    # Discrétisation: assignation de chaque valeur à un bin
    discrete_data = np.digitize(data, split_points)

    return discrete_data


def discretise_numeric_dimension_by_entropy(columns, dataframe, target,
                                           inplace=False, verbose=False,
                                           bins=3, threshold=.00000001):
    """
    Discrétise des variables numériques en utilisant l'entropie (supervisé).

    Version supervisée de discrétisation qui utilise les labels de classe
    pour trouver les meilleurs points de découpe.

    Args:
        columns (list): Colonnes à discrétiser
        dataframe (pandas.DataFrame): DataFrame source
        target (str): Nom de la colonne cible (labels)
        inplace (bool): Modifier le DataFrame original?
        verbose (bool): Afficher des infos de débogage?
        bins (int): Nombre max de bins
        threshold (float): Seuil de gain d'information

    Returns:
        pandas.DataFrame ou None: DataFrame discrétisé (si inplace=False)

    Exemple:
        >>> df = pd.DataFrame({
        ...     'age': [20, 25, 30, 60, 65, 70],
        ...     'classe': [0, 0, 0, 1, 1, 1]
        ... })
        >>> discretized = discretise_numeric_dimension_by_entropy(
        ...     ['age'], df, 'classe', bins=3
        ... )
        >>> # 'age' discrétisé selon la séparation des classes

    Avantage vs non-supervisé:
        Les frontières de bins sont optimisées pour séparer les classes,
        plutôt que basées sur des quantiles arbitraires.
    """
    # Gestion du mode inplace
    if inplace:
        data = dataframe
    else:
        data = dataframe.copy(deep=True)

    # Vérification des types
    if isinstance(columns, list) and isinstance(dataframe, pd.DataFrame):
        for col in columns:
            # Discrétisation supervisée par entropie
            dat = discretize_data(
                data[col].values,
                data[target],
                bins,
                threshold
            )

            # Génération des labels
            # Format: 'nom_colonne___code_bin'
            labels = [f'{col}___{int(i)}' for i in dat]

            # Affichage en mode verbeux
            print(f"""
                labels: {set(labels)}
                cols: {col}
                bins: {3}
                types: {data[col].dtype}
                """) if verbose else None

            # Remplacement des valeurs
            data[col] = labels

    # Retour selon le mode inplace
    if not inplace:
        return data
