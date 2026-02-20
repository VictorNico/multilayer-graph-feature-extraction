"""
Module d'Analyse Exploratoire des Données (EDA)

Auteur: VICTOR DJIEMBOU
Date de création: 02/12/2023
Dernière modification: 02/12/2023

Description:
    Ce module fournit des fonctions pour l'analyse exploratoire des données,
    notamment la détection et le traitement des valeurs manquantes (NaN).

Fonctionnalités principales:
    - Détection des colonnes avec valeurs manquantes
    - Imputation intelligente des valeurs manquantes (médiane/mode)
    - Suppression des colonnes avec trop de valeurs manquantes

Changements:
    - 02/12/2023:
      - Ajout de la méthode get_na_columns
      - Ajout de l'imputation intelligente par médiane/mode
"""

#################################################
##          Importation des bibliothèques
#################################################

import pandas as pd  # Manipulation de DataFrames
import numpy as np   # Calculs numériques
# from sklearn.impute import KNNImputer  # Imputation par KNN (désactivé)


#################################################
##          Définition des méthodes
#################################################


def get_na_columns(dataframe):
    """
    Détecte les colonnes contenant des valeurs manquantes (NaN) et calcule leur proportion

    Cette fonction analyse un DataFrame pour identifier toutes les colonnes qui contiennent
    au moins une valeur manquante, et calcule la proportion de valeurs manquantes par rapport
    au nombre total de lignes.

    Paramètres:
        dataframe (DataFrame): Le dataset à analyser

    Retourne:
        list: Liste de tuples (nom_colonne, proportion_na)
              où proportion_na est un float entre 0 et 1
              Retourne une liste vide si l'entrée n'est pas un DataFrame

    Exemple:
        >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [1, 2, 3]})
        >>> get_na_columns(df)
        [('A', 0.333)]
    """
    # Vérification que l'entrée est bien un DataFrame pandas
    if isinstance(dataframe, pd.DataFrame):
        # Calcul du nombre de NaN par colonne via isna().sum()
        # Calcul de la proportion en divisant par le nombre total de lignes
        # Création d'une liste de tuples (nom_colonne, proportion) seulement si val > 0
        NAs = [
            (dataframe.columns.tolist()[i], val/dataframe.shape[0])
            for i, val in enumerate(dataframe.isna().sum().values.tolist())
            if val > 0  # Filtre pour ne garder que les colonnes avec au moins 1 NaN
        ]
    else:
        # Si ce n'est pas un DataFrame, retourne une liste vide
        NAs = []

    return NAs


def impute_nan_values(dataframe, variables):
    """
    Remplace ou supprime les colonnes avec valeurs manquantes selon un seuil

    Cette fonction applique une stratégie d'imputation intelligente:
    - Si proportion NaN < 50%: Imputation par médiane (numériques) ou mode (catégorielles)
    - Si proportion NaN >= 50%: Suppression de la colonne (trop de données manquantes)

    Paramètres:
        dataframe (DataFrame): Dataset avec valeurs manquantes
        variables (list): Liste de tuples (nom_colonne, proportion_na)
                         obtenue via get_na_columns()

    Retourne:
        DataFrame: Nouveau DataFrame avec valeurs manquantes traitées
                   (copie profonde, l'original n'est pas modifié)

    Stratégies d'imputation:
        - Colonnes numériques: Remplacement par la médiane
        - Colonnes catégorielles: Remplacement par le mode (valeur la plus fréquente)
        - Colonnes avec >50% de NaN: Suppression complète

    Exemple:
        >>> df = pd.DataFrame({'A': [1, 2, None, 4], 'B': [None, None, None, 1]})
        >>> na_cols = get_na_columns(df)
        >>> df_clean = impute_nan_values(df, na_cols)
        # 'A' sera imputée par la médiane (2.5)
        # 'B' sera supprimée (75% de NaN)
    """
    # Vérification des types d'entrée
    if isinstance(dataframe, pd.DataFrame) and isinstance(variables, list):
        # Création d'une copie profonde pour éviter de modifier l'original
        data = dataframe.copy(deep=True)

        # Traitement de chaque colonne avec valeurs manquantes
        for (col, tho) in variables:  # tho = threshold/proportion de NaN
            if tho < 0.5:
                # Cas 1: Moins de 50% de NaN -> Imputation
                # Note: Alternative commentée avec quantile(0.5) équivaut à median()

                # Vérification du type de données de la colonne
                if pd.api.types.is_numeric_dtype(data[col]):
                    # Colonnes numériques: Imputation par la médiane
                    # La médiane est robuste aux valeurs extrêmes
                    data[col] = data[col].fillna(data[col].median())
                else:
                    # Colonnes non-numériques (catégorielles): Imputation par le mode
                    # Le mode est la valeur la plus fréquente
                    # mode()[0] car mode() retourne une Series, on prend le premier élément
                    data[col] = data[col].fillna(data[col].mode()[0])
            else:
                # Cas 2: 50% ou plus de NaN -> Suppression de la colonne
                # Trop de données manquantes pour une imputation fiable
                data.drop([col], axis=1, inplace=True)

    return data
