"""
=============================================================================
Module de Gestion des Fichiers et Données pour MLNA Experiment
=============================================================================

Auteur: VICTOR DJIEMBOU
Date de création: 15/11/2023
Dernière modification: 15/11/2023

Description:
    Ce module fournit un ensemble de fonctions utilitaires pour la gestion des
    fichiers et données dans le cadre d'expérimentations de Machine Learning
    avec des réseaux multiniveaux (MLNA - MultiLayer Network Analysis).

    Fonctionnalités principales:
    - Chargement de datasets depuis différents formats (CSV, Excel, JSON)
    - Sauvegarde de modèles ML, graphes, et datasets
    - Lecture de modèles et graphes sauvegardés
    - Gestion de configurations
    - Création de répertoires de sortie

Dépendances:
    - pandas: Manipulation de données tabulaires
    - numpy: Calculs numériques
    - networkx: Gestion de graphes
    - joblib: Sérialisation de modèles
    - configparser: Gestion de fichiers de configuration

Historique des modifications:
    - 15/11/2023: Ajout des méthodes de pipeline de traitement

=============================================================================
"""

#################################################
##          Importation des bibliothèques
#################################################

###### Début

import sys
import pandas as pd
import time
import os
import joblib
from networkx import write_gml, write_graphml_lxml, read_gml  #, read_graphml_lxml
import configparser
import copy
import shutil


###### Fin


#################################################
##          Définition des méthodes
#################################################

def load_data_set_from_url(path, na_values, sep='\t', encoding='utf-8', index_col=None, verbose=False):
    """
    Charge un dataset depuis différents formats de fichiers.

    Cette fonction détecte automatiquement le format du fichier (CSV, Excel, JSON)
    à partir de son extension et utilise le lecteur pandas approprié.

    Args:
        path (str): Chemin vers le fichier dataset à charger
        na_values (list): Liste des valeurs à considérer comme manquantes (NaN)
        sep (str, optional): Délimiteur utilisé dans le fichier (défaut: '\t' pour CSV)
        encoding (str, optional): Encodage du fichier (défaut: 'utf-8')
        index_col (int, optional): Colonne à utiliser comme index (défaut: None)
        verbose (bool, optional): Si True, affiche des informations de débogage (défaut: False)

    Returns:
        pandas.DataFrame: Le dataset chargé en mémoire
        ou str: Message d'erreur si l'extension n'est pas supportée

    Exemple:
        >>> df = load_data_set_from_url('data/train.csv', na_values=['NA', '?'], sep=',')
        >>> print(df.shape)
        (1000, 15)

    Note:
        Les formats supportés sont: .csv, .xlsx, .json
    """

    # Extraction de l'extension du fichier pour déterminer le format
    extension = os.path.splitext(path)[1] if isinstance(path, str) else None

    # Si aucune extension n'est trouvée, on assume que c'est un CSV
    if extension == None:
        extension = '.csv'

    # Affichage de l'extension détectée (mode verbeux)
    print(f"file ext know as {extension}") if verbose else None

    # Dictionnaire de mapping: extension -> fonction de lecture pandas
    readers = {
        ".csv": pd.read_csv,      # Fichiers CSV (Comma/Tab Separated Values)
        ".xlsx": pd.read_excel,   # Fichiers Excel
        ".json": pd.read_json,    # Fichiers JSON
    }

    # Chargement du dataset avec le lecteur approprié
    # Pour CSV: on utilise les paramètres sep et encoding
    # Pour autres formats: on n'utilise que index_col et na_values
    dataset = (readers[extension](path, sep=sep, encoding='utf-8', index_col=index_col,
                                  na_values=na_values) if '.csv' in extension else readers[extension](path,
                                                                                                      index_col=index_col,
                                                                                                      na_values=na_values)) if \
    readers[extension] else f"no reader define for the extension {extension}"

    return dataset


def load_config(config_path):
    """
    Charge un fichier de configuration INI.

    Utilise le module configparser pour lire et parser un fichier de configuration
    au format INI avec des sections et des paires clé-valeur.

    Args:
        config_path (str): Chemin vers le fichier de configuration (.ini)

    Returns:
        configparser.ConfigParser: Objet contenant la configuration chargée

    Exemple:
        >>> config = load_config('config/experiment.ini')
        >>> database_url = config['database']['url']

    Format attendu du fichier INI:
        [section1]
        key1 = value1
        key2 = value2

        [section2]
        key3 = value3
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def load_data_set_from_url_and_convert_to_dict(path, na_values, sep='\t', encoding='utf-8', index_col=None,
                                               verbose=False):
    """
    Charge un dataset depuis un fichier et le convertit en dictionnaire.

    Cette fonction combine le chargement d'un dataset avec sa conversion en dictionnaire.
    Utile pour la sérialisation JSON ou le passage de données à des APIs.

    Args:
        path (str): Chemin vers le fichier dataset à charger
        na_values (list): Liste des valeurs à considérer comme manquantes (NaN)
        sep (str, optional): Délimiteur utilisé dans le fichier (défaut: '\t')
        encoding (str, optional): Encodage du fichier (défaut: 'utf-8')
        index_col (int, optional): Colonne à utiliser comme index (défaut: None)
        verbose (bool, optional): Si True, affiche des informations de débogage

    Returns:
        dict: Le dataset sous forme de dictionnaire avec l'orientation 'list'
              Format: {colonne1: [valeurs...], colonne2: [valeurs...], ...}

    Exemple:
        >>> data_dict = load_data_set_from_url_and_convert_to_dict('data/train.csv',
        ...                                                          na_values=['NA'])
        >>> print(data_dict.keys())
        dict_keys(['age', 'income', 'label'])

    Note:
        L'orientation 'list' signifie que chaque colonne est une liste de valeurs
    """

    # Extraction de l'extension du fichier
    extension = os.path.splitext(path)[1] if isinstance(path, str) else None

    # Valeur par défaut si aucune extension
    if extension == None:
        extension = '.csv'

    # Mode verbeux: affichage de l'extension
    print(f"file ext know as {extension}") if verbose else None

    # Dictionnaire des lecteurs de fichiers
    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".json": pd.read_json,
    }

    # Chargement du dataset selon le format détecté
    dataset = (readers[extension](path, sep=sep, encoding='utf-8', index_col=index_col,
                                  na_values=na_values) if '.csv' in extension else readers[extension](path,
                                                                                                      index_col=index_col,
                                                                                                      na_values=na_values)) if \
        readers[extension] else f"no reader define for the extension {extension}"

    # Conversion en dictionnaire avec orientation 'list'
    # Chaque clé est un nom de colonne, chaque valeur est une liste
    return dataset.to_dict(orient='list')


def save_model(cwd, clf, prefix, clf_name, ext=".sav", sub="/model_storage", times=True):
    """
    Sauvegarde un modèle de Machine Learning sur le disque.

    Utilise joblib pour sérialiser un modèle entraîné. Le fichier peut inclure
    un timestamp pour tracer les différentes versions.

    Args:
        cwd (str): Répertoire de travail courant (Current Working Directory)
        clf: Instance du modèle à sauvegarder (scikit-learn, XGBoost, etc.)
        prefix (str): Préfixe pour le nom du fichier (peut être utilisé pour versioning)
        clf_name (str): Nom descriptif du modèle (ex: 'RandomForest', 'XGBoost')
        ext (str, optional): Extension du fichier (défaut: '.sav')
        sub (str, optional): Sous-répertoire de stockage (défaut: '/model_storage')
        times (bool, optional): Si True, ajoute un timestamp au nom (défaut: True)

    Returns:
        str: Chemin complet vers le fichier sauvegardé

    Exemple:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> # ... entraînement du modèle ...
        >>> path = save_model('/home/user/project', model, 'v1', 'RF_classifier')
        >>> print(path)
        /home/user/project/model_storage/RF_classifier_2023_11_15_14_30_45.sav

    Note:
        Le format joblib est plus efficace que pickle pour les objets numpy volumineux
    """
    # Création du répertoire de stockage s'il n'existe pas
    create_domain(cwd + sub)

    # Génération du timestamp au format YYYY_MM_DD_HH_MM_SS
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")

    # Construction du chemin complet: répertoire + nom + timestamp + extension
    filename = cwd + sub + '/' + clf_name + '_' + (timestr if times is True else '') + ext

    # Sérialisation du modèle avec joblib (compression implicite)
    joblib.dump(clf, filename)

    return filename


def save_graph(cwd, graph, name, rows_len, prefix, cols_len):
    """
    Sauvegarde un graphe NetworkX au format GML compressé.

    Le format GML (Graph Modeling Language) est un format standard pour la
    représentation de graphes. La compression .gz réduit la taille du fichier.

    Args:
        cwd (str): Répertoire de travail courant
        graph (networkx.Graph): Instance du graphe à sauvegarder
        name (str): Nom descriptif du graphe
        rows_len (int): Nombre d'exemples/nœuds utilisés pour construire le graphe
        prefix (str): Préfixe pour organisation des fichiers
        cols_len (int): Nombre de dimensions/features utilisées

    Returns:
        str: Chemin complet vers le fichier GML sauvegardé

    Exemple:
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_edges_from([(1,2), (2,3), (3,1)])
        >>> path = save_graph('/project', G, 'triangle_graph', 3, 'exp1', 2)
        >>> print(path)
        /project/graph_storage/triangle_graph_3_2_2023_11_15_14_30_45.gml.gz

    Note:
        Le format GML préserve les attributs des nœuds et des arêtes
    """
    # Création du répertoire de stockage des graphes
    create_domain(cwd + '/graph_storage/')

    # Génération du timestamp
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")

    # Construction du nom de fichier avec métadonnées
    # Format: nom_nb_exemples_nb_dimensions_timestamp.gml.gz
    filename = cwd + '/graph_storage' + '/' + name + '_' + str(rows_len) + '_' + str(
        cols_len) + '_' + timestr + '.gml.gz'

    # Écriture du graphe au format GML avec compression
    write_gml(graph, filename)

    return filename


def save_digraph(cwd, graph, name, rows_len, cols_len, prefix=None):
    """
    Sauvegarde un graphe orienté (DiGraph) au format GraphML.

    GraphML est un format XML pour les graphes. Contrairement à GML, il supporte
    mieux les graphes orientés et les attributs typés.

    Args:
        cwd (str): Répertoire de travail courant
        graph (networkx.DiGraph): Instance du graphe orienté à sauvegarder
        name (str): Nom descriptif du graphe
        rows_len (int): Nombre d'exemples utilisés
        cols_len (int): Nombre de dimensions utilisées
        prefix (str, optional): Préfixe pour organisation (défaut: None)

    Returns:
        str: Chemin complet vers le fichier GraphML sauvegardé

    Exemple:
        >>> import networkx as nx
        >>> DG = nx.DiGraph()
        >>> DG.add_edges_from([(1,2), (2,3)])  # Arêtes orientées
        >>> path = save_digraph('/project', DG, 'directed_chain', 3, 2)

    Note:
        GraphML est plus verbeux que GML mais plus expressif pour les graphes complexes
    """
    # Création du répertoire de stockage
    create_domain(cwd + '/graph_storage/')

    # Génération du timestamp
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")

    # Construction du chemin du fichier
    filename = cwd + '/graph_storage/' + name + '_' + str(rows_len) + '_' + str(cols_len) + '_' + timestr + '.gml.gz'

    # Écriture au format GraphML avec support LXML pour les performances
    write_graphml_lxml(graph, filename)

    return filename


def save_dataset(cwd, dataframe, name, prefix=None, sep='\t', sub="/data_selection_storage", index=True, times=True):
    """
    Sauvegarde un DataFrame pandas au format CSV.

    Cette fonction permet de sauvegarder des datasets intermédiaires ou finaux
    lors du pipeline de traitement des données.

    Args:
        cwd (str): Répertoire de travail courant
        dataframe (pandas.DataFrame): DataFrame à sauvegarder
        name (str): Nom du fichier (sans extension)
        prefix (str, optional): Préfixe pour versioning (défaut: None)
        sep (str, optional): Séparateur de colonnes (défaut: '\t' pour TSV)
        sub (str, optional): Sous-répertoire de stockage (défaut: '/data_selection_storage')
        index (bool, optional): Si True, sauvegarde l'index (défaut: True)
        times (bool, optional): Si True, ajoute un timestamp (défaut: True)

    Returns:
        str: Chemin complet vers le fichier CSV sauvegardé

    Exemple:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
        >>> path = save_dataset('/project', df, 'processed_data', sep=',')
        >>> print(path)
        /project/data_selection_storage/processed_data_2023_11_15_14_30_45.csv

    Note:
        Le séparateur '\t' (tabulation) est préféré pour éviter les conflits
        avec les virgules dans les données textuelles
    """
    # Création du répertoire de stockage s'il n'existe pas
    create_domain(cwd + sub)

    # Génération du timestamp
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")

    # Construction du nom de fichier avec timestamp optionnel
    filename = cwd + sub + '/' + name + '_' + (timestr if times is True else '') + '.csv'

    # Sauvegarde du DataFrame en CSV
    # - sep: séparateur de colonnes
    # - encoding: UTF-8 pour support international
    # - index: inclusion ou non de la colonne d'index
    dataframe.to_csv(filename, sep=sep, encoding='utf-8', index=index)

    return filename


def read_model(path):
    """
    Charge un modèle de Machine Learning depuis le disque.

    Désérialise un modèle précédemment sauvegardé avec joblib.

    Args:
        path (str): Chemin complet vers le fichier du modèle (.sav, .pkl, etc.)

    Returns:
        object: Instance du modèle chargé, prêt pour prédictions

    Exemple:
        >>> model = read_model('/project/model_storage/RF_2023_11_15.sav')
        >>> predictions = model.predict(X_test)

    Note:
        Le modèle doit avoir été sauvegardé avec la même version de scikit-learn
        (ou bibliothèque correspondante) pour éviter les problèmes de compatibilité
    """
    return joblib.load(path)


def read_graph(path):
    """
    Charge un graphe depuis un fichier GML.

    Restaure un graphe NetworkX précédemment sauvegardé au format GML.

    Args:
        path (str): Chemin vers le fichier GML (.gml ou .gml.gz)

    Returns:
        networkx.Graph: Instance du graphe chargé avec tous ses attributs

    Exemple:
        >>> G = read_graph('/project/graph_storage/my_graph.gml.gz')
        >>> print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    Note:
        NetworkX détecte automatiquement si le fichier est compressé (.gz)
    """
    return read_gml(path)


# def read_digraph(path):
#     """
#     Charge un graphe orienté depuis un fichier GraphML.
#
#     [FONCTION COMMENTÉE - Non utilisée actuellement]
#
#     Args:
#         path (str): Chemin vers le fichier GraphML
#
#     Returns:
#         networkx.DiGraph: Instance du graphe orienté chargé
#     """
#     return read_graphml_lxml(path)

def read_dataset(path, sep='\t'):
    """
    Charge un dataset depuis un fichier CSV.

    Lit un fichier CSV avec les paramètres standards du projet et reconstruit
    le DataFrame avec son index.

    Args:
        path (str): Chemin vers le fichier CSV
        sep (str, optional): Séparateur de colonnes (défaut: '\t')

    Returns:
        pandas.DataFrame: Le dataset chargé en mémoire

    Exemple:
        >>> df = read_dataset('/project/data/processed_data.csv', sep=',')
        >>> print(df.head())

    Note:
        index_col=0 signifie que la première colonne sera utilisée comme index
    """
    return pd.read_csv(path, sep=sep, encoding='utf-8', index_col=0)


def create_domain(directory, verbose=True):
    """
    Crée un répertoire d'analyse s'il n'existe pas déjà.

    Cette fonction gère de manière robuste la création de répertoires,
    incluant la gestion des erreurs courantes (permissions, existence).

    Args:
        directory (str): Chemin du répertoire à créer (peut inclure sous-répertoires)
        verbose (bool, optional): Si True, affiche des messages de statut (défaut: True)

    Returns:
        bool: True si le répertoire a été créé, False s'il existait déjà ou en cas d'erreur

    Exemple:
        >>> create_domain('/project/output/experiments/run_1')
        Directory '/project/output/experiments/run_1' created successfully.
        True

        >>> create_domain('/project/output/experiments/run_1')
        Directory '/project/output/experiments/run_1' already exists.
        False

    Note:
        La fonction crée tous les répertoires parents nécessaires (mkdir -p)
    """
    state = False
    try:
        # Création du répertoire et de tous les parents nécessaires
        os.makedirs(directory)
        state = True

        # Message de succès en mode verbeux
        print(f"Directory '{directory}' created successfully.") if verbose else None

    except FileExistsError:
        # Le répertoire existe déjà - ce n'est pas une erreur critique
        print(f"Directory '{directory}' already exists.") if verbose else None

    except OSError as e:
        # Erreur système (permissions, disque plein, etc.)
        print(f"An error occurred while creating directory '{directory}': {e}") if verbose else None

    return state
