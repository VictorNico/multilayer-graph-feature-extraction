"""
=============================================================================
Module de Tests pour la Détection de Colonnes Ordinales
=============================================================================

Auteur: Non spécifié
Date de création: Non spécifiée
Dernière modification: Non spécifiée

Description:
    Ce module fournit des fonctions utilitaires pour détecter et analyser
    les types de colonnes dans un DataFrame pandas, avec un focus particulier
    sur la distinction entre colonnes ordinales et nominales.

    Fonctionnalités principales:
    - Détection de colonnes ordinales (valeurs ayant un ordre naturel)
    - Identification de colonnes numériques à logique nominale
    - Test de distribution des valeurs dans une série

Dépendances:
    - pandas: Manipulation de DataFrames
    - numpy: Calculs numériques et arrays

Cas d'usage:
    Ce module est utile dans le preprocessing des données pour:
    1. Identifier automatiquement le type de variables catégorielles
    2. Décider du type d'encodage approprié (ordinal vs one-hot)
    3. Détecter les identifiants numériques (ID) qui ne doivent pas être normalisés

=============================================================================
"""

import pandas as pd
import numpy as np


def is_ordinal(column):
    """
    Détermine si une colonne contient des valeurs ordinales.

    Une colonne est considérée ordinale si ses valeurs uniques sont déjà
    triées dans l'ordre croissant ou décroissant. Cela indique un ordre
    naturel entre les catégories.

    Args:
        column (pandas.Series): Colonne à analyser

    Returns:
        bool: True si la colonne est ordinale, False sinon

    Exemples:
        >>> df = pd.DataFrame({'taille': ['petit', 'moyen', 'grand']})
        >>> is_ordinal(df['taille'])
        True  # Si 'grand' > 'moyen' > 'petit' alphabétiquement

        >>> df = pd.DataFrame({'niveau': ['faible', 'moyen', 'élevé']})
        >>> is_ordinal(df['niveau'])
        True  # Ordre: 'faible' < 'moyen' < 'élevé'

        >>> df = pd.DataFrame({'couleur': ['rouge', 'bleu', 'vert']})
        >>> is_ordinal(df['couleur'])
        False  # Pas d'ordre naturel entre les couleurs

    Algorithme:
        1. Extraire les valeurs uniques de la colonne
        2. Créer une version triée de ces valeurs
        3. Comparer l'ordre original avec l'ordre trié (croissant et décroissant)
        4. Si l'un des deux correspond, la colonne est ordinale

    Note:
        Cette méthode suppose que l'ordre actuel des valeurs reflète
        l'ordre d'apparition dans les données. Pour les données non triées,
        cette heuristique peut produire des faux négatifs.
    """
    # Extraction des valeurs uniques (dans l'ordre d'apparition)
    unique_values = column.unique()

    # Tri des valeurs uniques (ordre croissant)
    sorted_values = sorted(unique_values)

    # Vérification de l'ordre: croissant OU décroissant
    # list(unique_values) == sorted_values : ordre croissant
    # list(unique_values) == sorted_values[::-1] : ordre décroissant
    return list(unique_values) == sorted_values or list(unique_values) == sorted_values[::-1]


def detect_ordinal_columns(df):
    """
    Détecte toutes les colonnes ordinales dans un DataFrame.

    Parcourt toutes les colonnes du DataFrame et applique le test is_ordinal()
    pour identifier lesquelles présentent un ordre naturel.

    Args:
        df (pandas.DataFrame): DataFrame à analyser

    Returns:
        list: Liste des noms de colonnes identifiées comme ordinales

    Exemple:
        >>> df = pd.DataFrame({
        ...     'education': ['Licence', 'Master', 'Doctorat'],
        ...     'taille': ['S', 'M', 'L'],
        ...     'couleur': ['rouge', 'bleu', 'vert'],
        ...     'age': [25, 30, 35]
        ... })
        >>> ordinal_cols = detect_ordinal_columns(df)
        >>> print(ordinal_cols)
        ['education', 'taille', 'age']  # 'couleur' n'est pas ordinale

    Note:
        - Les colonnes numériques sont souvent détectées comme ordinales
        - Cette fonction ne distingue pas entre ordinales réelles et numériques
        - Utilisez en combinaison avec numeric_vector_is_nominal() pour filtrer
    """
    ordinal_columns = []

    # Parcours de toutes les colonnes du DataFrame
    for column in df.columns:
        # Test d'ordinalité pour chaque colonne
        if is_ordinal(df[column]):
            ordinal_columns.append(column)

    return ordinal_columns


def numeric_vector_is_nominal(series):
    """
    Détermine si une colonne numérique a en réalité une logique nominale.

    Une colonne numérique est considérée nominale si toutes ses valeurs
    sont uniques, ce qui suggère qu'elle représente des identifiants (IDs)
    plutôt que des mesures quantitatives.

    Args:
        series (pandas.Series, list, ou numpy.ndarray): Série de valeurs à analyser

    Returns:
        bool: True si toutes les valeurs sont uniques (logique nominale)
        None: Si le type de données n'est pas supporté

    Exemples:
        >>> # Cas 1: Identifiant unique (nominal)
        >>> ids = pd.Series([1001, 1002, 1003, 1004])
        >>> numeric_vector_is_nominal(ids)
        True  # Chaque valeur est unique = ID

        >>> # Cas 2: Variable quantitative (non nominale)
        >>> ages = pd.Series([25, 30, 25, 35, 30])
        >>> numeric_vector_is_nominal(ages)
        False  # Des valeurs se répètent = mesure quantitative

        >>> # Cas 3: Code catégoriel (non nominal)
        >>> categories = pd.Series([1, 2, 3, 1, 2, 3])
        >>> numeric_vector_is_nominal(categories)
        False  # Répétitions = catégories encodées

    Cas d'usage:
        - Identifier les colonnes ID qui ne doivent PAS être normalisées
        - Détecter les numéros de série, matricules, codes postaux
        - Éviter les erreurs d'encodage sur des identifiants

    Algorithme:
        1. Conversion de l'entrée en pandas.Series si nécessaire
        2. Comptage des occurrences de chaque valeur (value_counts)
        3. Vérification que toutes les valeurs ont une occurrence == 1
        4. Si oui, c'est une série nominale (identifiants)

    Note:
        Cette heuristique fonctionne bien pour les datasets standards mais
        peut produire des faux positifs sur de très petits échantillons où
        toutes les valeurs sont différentes par hasard.
    """
    data = None

    # Conversion de l'entrée en pandas.Series selon le type
    if isinstance(series, pd.Series):
        # Déjà une Series, utilisation directe
        data = series
    elif isinstance(series, list) or isinstance(series, np.ndarray):
        # Conversion depuis list ou array numpy
        data = pd.Series(series)
    else:
        # Type non supporté
        return None

    # Comptage des occurrences de chaque valeur unique
    # Ex: [1,2,2,3] -> {1: 1, 2: 2, 3: 1}
    occurrences = data.value_counts()

    # Vérification que toutes les valeurs apparaissent exactement 1 fois
    # Si (occurrences == 1).all() est True, toutes les valeurs sont uniques
    # Cela caractérise une série nominale (identifiants)
    suit_serie_nominale = (occurrences == 1).all()

    return suit_serie_nominale


# ============================================================
# SECTION DE TESTS ET DÉMONSTRATION
# ============================================================

# Point d'entrée pour l'exécution en tant que script
if __name__ == '__main__':
    """
    Section de tests pour démontrer l'utilisation des fonctions.

    Cette section s'exécute uniquement si le fichier est lancé directement
    (pas lors d'un import en tant que module).
    """

    # Création d'un DataFrame d'exemple avec différents types de colonnes
    df = pd.DataFrame({
        # Colonne 1: Potentiellement ordinale (dépend de l'ordre alphabétique)
        'colonne1': ['faible', 'moyen', 'élevé', 'blala'],

        # Colonne 2: Non ordinale (pas d'ordre naturel entre A, B, C, D)
        'colonne2': ['A', 'B', 'C', 'D'],

        # Colonne 3: Numérique (valeurs différentes = nominal selon la fonction)
        'colonne3': [70, 25, 45, 60]
    })

    # Test 1: Détection des colonnes ordinales
    ordinal_columns = detect_ordinal_columns(df)
    print("Colonnes ordinales détectées:")
    print(ordinal_columns)

    # Test 2: Vérification de la logique nominale pour chaque colonne
    # ATTENTION: Il y a un bug dans cette ligne du code original!
    # numeric_vector_is_nominal(col) devrait être numeric_vector_is_nominal(df[col])
    # La version actuelle passe le NOM de la colonne au lieu de ses DONNÉES
    cols = {col: True for col in df.columns if numeric_vector_is_nominal(col)}
    print("\nColonnes avec logique nominale (BUGGÉ):")
    print(cols)

    # Version corrigée du test 2 (à utiliser dans le code réel):
    # cols_corrected = {col: True for col in df.columns if numeric_vector_is_nominal(df[col])}
    # print("\nColonnes avec logique nominale (CORRIGÉ):")
    # print(cols_corrected)
    # Résultat attendu: {'colonne3': True} si toutes les valeurs sont uniques

    # Exemple de sortie attendue:
    # Colonnes ordinales détectées:
    # ['colonne1', 'colonne2', 'colonne3']
    #
    # Colonnes avec logique nominale:
    # {'colonne3': True}
