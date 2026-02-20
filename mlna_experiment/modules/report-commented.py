"""
=============================================================================
Module de Génération de Rapports et Visualisations pour MLNA
=============================================================================

Auteur: VICTOR DJIEMBOU
Date de création: 15/11/2023
Dernière modification: 15/11/2023

Description:
    Ce module fournit des fonctionnalités pour créer des rapports visuels
    et HTML à partir des résultats d'expérimentations de Machine Learning.
    Il gère la création de graphiques, de tableaux comparatifs, et l'export
    en différents formats.

    Fonctionnalités principales:
    - Visualisation d'importances de features (barres horizontales)
    - Génération de rapports HTML comparatifs
    - Affichage de graphes NetworkX
    - Export de tableaux et graphiques
    - Système de coloration personnalisé pour visualisations

Dépendances:
    - matplotlib: Création de graphiques
    - seaborn: Stylisation des graphiques
    - networkx: Visualisation de graphes
    - pandas: Manipulation de données
    - IPython: Affichage HTML dans notebooks

Cas d'usage:
    - Génération automatique de rapports d'expériences ML
    - Comparaison visuelle de performances de modèles
    - Documentation des résultats d'expérimentation
    - Création de dashboards de résultats

Historique des modifications:
    - 15/11/2023: Ajout des méthodes de pipeline de visualisation

=============================================================================
"""

#################################################
##          Importation des bibliothèques
#################################################

###### Début

# Bibliothèques de visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# import missingno as msno  # Pour visualiser les données manquantes (commenté)
# import plotly.express as px  # Visualisations interactives (commenté)
# import plotly.figure_factory as ff  # Graphiques complexes (commenté)
# import plotly.graph_objects as go  # Objets graphiques Plotly (commenté)
# import shap  # Explainabilité des modèles (commenté)

# Bibliothèques utilitaires
# import sys
import numpy as np
import time
# import os
# import joblib
import networkx as nx
from IPython.core.display import HTML
# import imgkit  # Conversion HTML vers images (commenté)

# Import relatif depuis le module file
from .file import create_domain

import pandas as pd
# from memory_profiler import profile  # Profilage mémoire (commenté)
# from markdown2pdf import convert  # Conversion Markdown vers PDF (commenté)

# ============================================================
# CONFIGURATION DU STYLE DE VISUALISATION
# ============================================================
# %matplotlib inline  # Pour notebooks Jupyter (commenté)
sns.set_style('darkgrid')  # Style de grille sombre pour seaborn

# Configuration globale de matplotlib
mpl.rcParams['font.size'] = 14              # Taille de police par défaut
mpl.rcParams['figure.facecolor'] = '#00000000'  # Fond transparent (RGBA)

# Duplication de la configuration (possiblement pour assurer la prise en compte)
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.facecolor'] = '#00000000'

###### Fin


#################################################
##          Définition des méthodes
#################################################

# Accepte une liste d'objets IpyTable et retourne un tableau HTML contenant chaque table
# @profile  # Décorateur de profilage mémoire (commenté)
def multi_table(table_list):
    """
    Crée un tableau HTML contenant plusieurs tableaux IPython côte à côte.

    Cette fonction est utile pour afficher plusieurs résultats en parallèle
    dans un notebook Jupyter, permettant une comparaison visuelle directe.

    Args:
        table_list (list): Liste d'objets de type table IPython ayant une méthode _repr_html_()

    Returns:
        IPython.core.display.HTML: Objet HTML affichable dans un notebook

    Exemple d'utilisation:
        >>> table1 = pd.DataFrame({'A': [1,2,3]}).style
        >>> table2 = pd.DataFrame({'B': [4,5,6]}).style
        >>> multi_table([table1, table2])
        # Affiche les deux tableaux côte à côte

    Structure HTML générée:
        <table>
          <tr style="background-color:#2020d1; color: #FFFFFF;">
            <td>[table1]</td>
            <td>[table2]</td>
            ...
          </tr>
        </table>

    Note:
        Le style de l'en-tête utilise un fond bleu (#2020d1) et texte blanc
    """
    # Construction de la ligne HTML avec chaque table dans une cellule
    return HTML(
        '<table><tr style="background-color:#2020d1; color: #FFFFFF;">' +
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>'
    )


# @profile
def plot_graph(CRP_G_1):
    """
    Visualise et sauvegarde un graphe NetworkX avec couleurs personnalisées.

    Cette fonction dessine un graphe en utilisant les attributs de couleur
    des nœuds et des arêtes, puis sauvegarde le résultat en haute résolution.

    Args:
        CRP_G_1 (networkx.Graph): Graphe NetworkX avec attributs 'color' sur
                                   les nœuds et les arêtes

    Returns:
        None: La fonction affiche et sauvegarde le graphe

    Processus:
        1. Extraction des couleurs des arêtes et nœuds
        2. Dessin du graphe avec NetworkX
        3. Sauvegarde en haute résolution (700 DPI)
        4. Affichage du graphe

    Exemple:
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_node(1, color='red')
        >>> G.add_node(2, color='blue')
        >>> G.add_edge(1, 2, color='green')
        >>> plot_graph(G)

    Note:
        - Le fichier est sauvegardé dans './plots/' avec timestamp
        - Format de sortie: PNG à 700 DPI (haute qualité)
        - Les labels des nœuds ne sont pas affichés (with_labels=False par défaut)
    """
    # Extraction des couleurs des arêtes depuis les attributs du graphe
    colors = nx.get_edge_attributes(CRP_G_1, 'color').values()

    # Extraction des couleurs des nœuds
    colorsN = nx.get_node_attributes(CRP_G_1, 'color').values()

    # Dessin du graphe avec les couleurs extraites
    nx.draw(
        CRP_G_1,
        edge_color=colors,    # Couleurs des arêtes
        node_color=colorsN    # Couleurs des nœuds
        # with_labels=True    # Commenté: ne pas afficher les labels
    )

    # Génération du timestamp pour nom de fichier unique
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")

    # Construction du chemin du fichier de sortie
    # Format: graph_mln_1_person_home_ownership_loan_intent_YYYY_MM_DD_HH_MM_SS.png
    filename1 = './plots/graph_mln_1_person_home_ownership_loan_intent' + '_' + timestr + '.png'

    # Sauvegarde en haute résolution (700 DPI)
    plt.savefig(filename1, dpi=700)  # Formats supportés: .png, .pdf

    # Affichage du graphe
    plt.show()


# @profile
def custom_color(dataframe, graph_a=[]):
    """
    Génère un schéma de couleurs personnalisé basé sur les noms de colonnes.

    Cette fonction attribue des couleurs spécifiques aux colonnes selon
    des suffixes prédéfinis, permettant de distinguer visuellement différents
    types de features dans les graphiques.

    Args:
        dataframe (pandas.DataFrame ou pandas.Index): DataFrame ou liste de colonnes
        graph_a (list, optional): Paramètre non utilisé (possiblement pour extensions futures)

    Returns:
        list: [liste_couleurs, liste_noms_colonnes]
            - liste_couleurs: Couleur attribuée à chaque colonne
            - liste_noms_colonnes: Noms des colonnes

    Schéma de couleurs:
        - '_PER' dans le nom → 'green' (features personnelles?)
        - '_GLO' dans le nom → 'yellow' (features globales?)
        - Autres → 'dodgerblue' (couleur par défaut)

    Exemple:
        >>> df = pd.DataFrame({'age_PER': [1], 'income_GLO': [2], 'score': [3]})
        >>> colors, cols = custom_color(df)
        >>> print(colors)
        ['green', 'yellow', 'dodgerblue']

    Cas d'usage:
        - Coloration de barres dans les graphiques d'importance de features
        - Distinction visuelle de types de variables
        - Légendes automatiques basées sur les noms
    """
    # Conversion en liste de noms de colonnes
    cols = dataframe.tolist()

    colors = []

    # Attribution de couleurs selon le suffixe
    for col in cols:
        if '_PER' in col:
            # Features personnelles (Personal) → vert
            colors.append('green')
        elif '_GLO' in col:
            # Features globales (Global) → jaune
            colors.append('yellow')
        else:
            # Features par défaut → bleu ciel
            colors.append('dodgerblue')

    return [colors, cols]


class color:
    """
    Classe utilitaire pour les codes de couleur ANSI dans le terminal.

    Fournit des constantes pour styliser du texte dans la console avec
    des couleurs et des styles (gras, souligné).

    Attributs de classe (codes ANSI):
        PURPLE: Violet
        CYAN: Cyan clair
        DARKCYAN: Cyan foncé
        BLUE: Bleu
        GREEN: Vert
        YELLOW: Jaune
        RED: Rouge
        BOLD: Texte en gras
        UNDERLINE: Texte souligné
        END: Réinitialisation du style

    Exemple d'utilisation:
        >>> print(f"{color.RED}Erreur critique!{color.END}")
        Erreur critique!  # En rouge dans le terminal

        >>> print(f"{color.BOLD}{color.GREEN}Succès!{color.END}")
        Succès!  # En vert gras

    Note:
        Ces codes fonctionnent dans la plupart des terminaux Unix/Linux/Mac.
        Support limité sur Windows (utiliser colorama pour compatibilité).
    """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'  # Réinitialisation (important!)


def get_color():
    """
    Retourne la classe color pour accès aux codes ANSI.

    Returns:
        class: La classe color avec tous les codes de couleur

    Exemple:
        >>> colors = get_color()
        >>> print(f"{colors.BLUE}Texte bleu{colors.END}")
    """
    return color


def model_desc():
    """
    Retourne un dictionnaire de correspondance pour les noms de modèles.

    Cette fonction fournit un mapping entre codes courts et noms complets
    de modèles de Machine Learning.

    Returns:
        dict: Mapping {code_court: nom_complet}

    Modèles supportés:
        - LDA: Linear Discriminant Analysis
        - LR: Logistic Regression
        - SVM: Support Vector Machine
        - DT: Decision Tree
        - RF: Random Forest
        - XGB: XGBoost
        - MLP: Multi-Layer Perceptron
        - PER: Perceptron

    Exemple:
        >>> models = model_desc()
        >>> print(models['RF'])
        'RF'  # Dans cette version, les noms sont identiques

    Note:
        Dans la version actuelle, les clés et valeurs sont identiques.
        Cela pourrait être étendu pour des noms plus descriptifs:
        {'LDA': 'Linear Discriminant Analysis', ...}
    """
    modelD = {
        'LDA': 'LDA',
        'LR': 'LR',
        'SVM': 'SVM',
        'DT': 'DT',
        'RF': 'RF',
        'XGB': 'XGB',
        'MLP': 'MLP',
        'PER': 'PER'
    }
    return modelD


# @profile
def plot_features_importance_as_barh(data, getColor, modelDictName, plotTitle,
                                    cwd, graph_a=[], save=True, prefix=None):
    """
    Génère des graphiques à barres horizontales montrant l'importance des features.

    Cette fonction crée un graphique pour chaque modèle dans les données,
    affichant les 20 features les plus importantes (en valeur absolue)
    avec leurs métriques de performance.

    Args:
        data (pandas.DataFrame): DataFrame contenant les importances et métriques
                                Colonnes attendues: features + 'f1-score', 'recall',
                                'precision', 'accuracy'
                                Index: noms des modèles
        getColor (function): Fonction pour générer les couleurs des barres
        modelDictName (dict): Mapping {code_modèle: nom_affichage}
        plotTitle (str): Titre du graphique
        cwd (str): Répertoire de travail courant (pour sauvegarde)
        graph_a (list, optional): Paramètre passé à getColor
        save (bool, optional): Si True, sauvegarde les graphiques (défaut: True)
        prefix (str, optional): Préfixe pour noms de fichiers (non utilisé actuellement)

    Returns:
        None: La fonction crée et sauvegarde les graphiques

    Processus pour chaque modèle:
        1. Extraction des importances de features (sans métriques)
        2. Tri par valeur absolue (importance décroissante)
        3. Sélection des 20 premières features
        4. Création du graphique à barres horizontales
        5. Ajout des métriques dans le label de l'axe X
        6. Sauvegarde en haute résolution (700 DPI)

    Exemple de structure de données:
        >>> data = pd.DataFrame({
        ...     'feature1': [0.5, 0.3],
        ...     'feature2': [0.3, 0.2],
        ...     'accuracy': [0.85, 0.80],
        ...     'precision': [0.83, 0.78],
        ...     'recall': [0.87, 0.82],
        ...     'f1-score': [0.85, 0.80]
        ... }, index=['RF', 'LR'])
        >>> plot_features_importance_as_barh(
        ...     data, custom_color, model_desc(),
        ...     'Feature Importance', '/project'
        ... )

    Format de sauvegarde:
        ./plots/Feature_Importance_RF_2023_11_15_14_30_45.png

    Note:
        - Hauteur du graphique adaptée au nombre de features
        - Ligne verticale à x=0 pour référence
        - Graphiques fermés après sauvegarde pour libérer la mémoire
    """
    # Traitement de chaque modèle (ligne du DataFrame)
    for index in data.index.values.tolist():
        # ============================================================
        # PRÉPARATION DES DONNÉES
        # ============================================================
        # Suppression des colonnes de métriques pour ne garder que les features
        ok = data.drop([
            'f1-score',
            'recall',
            'precision',
            'accuracy'
        ], axis=1)

        # Tri des features par importance absolue (décroissant)
        # key=lambda row: abs(row) pour trier par valeur absolue
        ok = ok.sort_values(
            by=index,
            axis=1,
            ascending=True,  # True car barh dessine de bas en haut
            key=lambda row: abs(row)
        ).head(20)  # Sélection des 20 premières

        # ============================================================
        # CONFIGURATION DE LA FIGURE
        # ============================================================
        width = 10  # Largeur fixe
        # Hauteur adaptée: nombre de features / 4
        height = int(len(np.unique(ok.columns.tolist())) / 4)

        # Extraction des données du modèle courant
        ok = ok.loc[[index], :]

        # Création d'un DataFrame pour le plotting
        df = pd.DataFrame({
            'Attr': ok.columns.tolist(),     # Noms des features
            'Val': ok.values[0]              # Valeurs d'importance
        })

        # ============================================================
        # CRÉATION DU GRAPHIQUE À BARRES HORIZONTALES
        # ============================================================
        df.plot.barh(
            x='Attr',                        # Axe Y: noms de features
            y='Val',                         # Axe X: valeurs d'importance
            figsize=(width, height),         # Dimensions du graphique
            color=getColor(ok, graph_a)[0]  # Couleurs personnalisées
        )

        # Titre du graphique
        plt.title(f"{plotTitle}")

        # Ligne verticale à x=0 pour référence (importances positives/négatives)
        plt.axvline(x=0, color=".5")

        # ============================================================
        # LABEL DE L'AXE X AVEC MÉTRIQUES
        # ============================================================
        # Construction d'un label contenant toutes les métriques de performance
        label = f"""
        {modelDictName[index]}
        ACCURACY: {data.loc[index, 'accuracy']}, PRECISION: {data.loc[index, 'precision']},
        RECALL: {data.loc[index, 'recall']}, F1-SCORE: {data.loc[index, 'f1-score']}
        """

        plt.xlabel(label)

        # Ajustement de la marge gauche pour afficher les longs noms de features
        plt.subplots_adjust(left=0.3)

        # ============================================================
        # SAUVEGARDE DU GRAPHIQUE
        # ============================================================
        if save == True:
            # Création du répertoire plots s'il n'existe pas
            create_domain(cwd + '/plots/')

            # Génération du timestamp
            timestr = time.strftime("%Y_%m_%d_%H_%M_%S")

            # Construction du nom de fichier
            # Format: titre_modèle_timestamp.png
            filename1 = (cwd + '/plots' + '/' +
                        ('_'.join(plotTitle.split(' '))) + '_' +
                        index + '_' + timestr + '.png')

            # Ajustement du layout pour éviter le clipping
            plt.tight_layout()

            # Sauvegarde en haute résolution
            plt.savefig(filename1, dpi=700)

            # Fermeture de la figure pour libérer la mémoire
            plt.close()

        # plt.show()  # Commenté: ne pas afficher dans un script automatisé


# @profile
def print_summary(table_list, modelDict):
    """
    Génère un tableau HTML comparatif des performances de modèles.

    Cette fonction crée un tableau HTML stylisé comparant les métriques de
    performance de plusieurs modèles à travers différentes étapes d'un pipeline.
    Les meilleures valeurs pour chaque métrique sont mises en évidence.

    Args:
        table_list (list): Liste de tuples (nom_étape, DataFrame_résultats)
                          Chaque DataFrame doit avoir:
                          - Index: noms des modèles
                          - Colonnes: 'accuracy', 'precision', 'recall', 'f1-score'
        modelDict (dict): Mapping {code_modèle: nom_affichage}

    Returns:
        tuple: (objet HTML, string HTML)
            - Objet HTML pour affichage dans notebook
            - String HTML pour sauvegarde

    Structure du tableau:
        - Une section par modèle (lignes groupées avec rowspan)
        - Une colonne par étape du pipeline
        - Une colonne par métrique (accuracy, precision, recall, f1-score)
        - Mise en évidence des meilleures valeurs en bleu gras grande taille

    Exemple d'utilisation:
        >>> baseline = pd.DataFrame({
        ...     'accuracy': [0.8, 0.7],
        ...     'precision': [0.78, 0.68],
        ...     'recall': [0.82, 0.72],
        ...     'f1-score': [0.80, 0.70]
        ... }, index=['RF', 'LR'])
        >>> optimized = pd.DataFrame({
        ...     'accuracy': [0.85, 0.75],
        ...     'precision': [0.83, 0.73],
        ...     'recall': [0.87, 0.77],
        ...     'f1-score': [0.85, 0.75]
        ... }, index=['RF', 'LR'])
        >>> html_obj, html_str = print_summary(
        ...     [('Baseline', baseline), ('Optimized', optimized)],
        ...     model_desc()
        ... )

    Style appliqué:
        - Meilleures valeurs: color:blue; font-size: 40px; font-weight: bold;
        - Bordures: 1px solid black
        - Fond: blanc (#FFFFFF)
        - Texte: noir (#000000)

    Algorithme de mise en évidence:
        Pour chaque métrique de chaque modèle:
        1. Comparer la valeur à travers toutes les étapes
        2. Si la valeur est maximale, appliquer le style de mise en évidence
        3. Utilisation de sum() et round() pour gérer les égalités

    Note:
        Le HTML généré est autonome (inclut les styles CSS)
    """
    # ============================================================
    # EXTRACTION DE LA LISTE DES MODÈLES
    # ============================================================
    # Utilisation du premier DataFrame pour obtenir la liste des modèles
    baseline = table_list[0][1].index.values.tolist()

    # ============================================================
    # CONSTRUCTION DE L'EN-TÊTE DU TABLEAU
    # ============================================================
    head = '<tr><td></td><td></td><td>Accuracy</td>' + \
           '<td>Precision</td><td>Recall</td><td>F1-score</td></tr>'

    # ============================================================
    # CONSTRUCTION DU CORPS DU TABLEAU
    # ============================================================
    body = ''

    # Traitement de chaque modèle
    for model in baseline:
        # Ligne de début pour ce modèle (avec rowspan pour fusionner les lignes)
        modelLines = f'<tr><td rowspan="{len(table_list)}" >{modelDict[model]}</td>'

        # Traitement de chaque étape du pipeline
        for i, (step, data) in enumerate(table_list):
            # ============================================================
            # CALCUL DE LA MISE EN ÉVIDENCE
            # ============================================================
            # Pour chaque métrique, vérifier si la valeur courante est maximale
            # Formule complexe pour déterminer si style='...' doit être appliqué

            # Mise en évidence de l'accuracy
            acc_style = "color:blue; font-size: 40px; font-weight: bold;" * int(
                sum([round(data.loc[model, "accuracy"], 4) ==
                     max(round(dat.loc[model, "accuracy"], 4),
                         round(data.loc[model, "accuracy"], 4))
                     for step, dat in table_list]) / len(table_list)
            )

            # Même logique pour precision, recall, f1-score
            prec_style = "color:blue; font-size: 40px; font-weight: bold;" * int(
                sum([round(data.loc[model, "precision"], 4) ==
                     max(round(dat.loc[model, "precision"], 4),
                         round(data.loc[model, "precision"], 4))
                     for step, dat in table_list]) / len(table_list)
            )

            rec_style = "color:blue; font-size: 40px; font-weight: bold;" * int(
                sum([round(data.loc[model, "recall"], 4) ==
                     max(round(dat.loc[model, "recall"], 4),
                         round(data.loc[model, "recall"], 4))
                     for step, dat in table_list]) / len(table_list)
            )

            f1_style = "color:blue; font-size: 40px; font-weight: bold;" * int(
                sum([round(data.loc[model, "f1-score"], 4) ==
                     max(round(dat.loc[model, "f1-score"], 4),
                         round(data.loc[model, "f1-score"], 4))
                     for step, dat in table_list]) / len(table_list)
            )

            # ============================================================
            # CONSTRUCTION DES CELLULES
            # ============================================================
            # Si pas la dernière étape: ajouter '</tr><tr>' pour nouvelle ligne
            if i < len(table_list) - 1:
                modelLines = (
                    modelLines +
                    f'<td>{step}</td>' +
                    f'<td style="{acc_style}">{round(data.loc[model, "accuracy"], 4)}</td>' +
                    f'<td style="{prec_style}">{round(data.loc[model, "precision"], 4)}</td>' +
                    f'<td style="{rec_style}">{round(data.loc[model, "recall"], 4)}</td>' +
                    f'<td style="{f1_style}">{round(data.loc[model, "f1-score"], 4)}</td>' +
                    '</tr> <tr >'
                )
            # Si dernière étape: fermer avec '</tr>'
            else:
                modelLines = (
                    modelLines +
                    f'<td>{step}</td>' +
                    f'<td style="{acc_style}">{round(data.loc[model, "accuracy"], 4)}</td>' +
                    f'<td style="{prec_style}">{round(data.loc[model, "precision"], 4)}</td>' +
                    f'<td style="{rec_style}">{round(data.loc[model, "recall"], 4)}</td>' +
                    f'<td style="{f1_style}">{round(data.loc[model, "f1-score"], 4)}</td>' +
                    '</tr>'
                )

        # Ajout des lignes du modèle au corps du tableau
        body = body + modelLines

    # ============================================================
    # CONSTRUCTION DU HTML COMPLET
    # ============================================================
    # Définition du style CSS
    style = '<style>table, th, td {border: 1px solid black;border-collapse: collapse;}</style>'

    # Construction du tableau HTML
    table_html = f'<table style="border: 2px solid black; width: 100% !important; ' \
                f'background-color: #FFFFFF; color:#000000;">{head}{body}</table>'

    # Construction du document HTML complet
    htm = f'<html><head>{style}<title> Summary </title></head>' \
          f'<body style="background-color: white;">{table_html}</body></html>'

    # Retour: objet HTML (pour notebook) + string HTML (pour sauvegarde)
    return (HTML(htm), htm)


# @profile
def create_file(content, cwd, filename, extension=".html", prefix=None):
    """
    Crée et sauvegarde un fichier avec le contenu spécifié.

    Cette fonction est utilisée pour sauvegarder des rapports HTML,
    des fichiers texte, ou tout autre contenu généré.

    Args:
        content (str): Contenu à écrire dans le fichier
        cwd (str): Répertoire de travail courant
        filename (str): Nom de base du fichier (sans extension)
        extension (str, optional): Extension du fichier (défaut: ".html")
        prefix (str, optional): Préfixe pour le nom de fichier (non utilisé)

    Returns:
        str: Chemin complet vers le fichier créé

    Processus:
        1. Création du répertoire 'reports' s'il n'existe pas
        2. Génération d'un nom de fichier avec timestamp
        3. Écriture du contenu dans le fichier
        4. Retour du chemin complet

    Exemple:
        >>> html_content = '<html><body>Rapport</body></html>'
        >>> path = create_file(html_content, '/project', 'summary')
        >>> print(path)
        /project/reports/summary2023_11_15_14_30_45.html

    Formats supportés:
        - .html (rapports HTML)
        - .txt (rapports texte)
        - .md (Markdown)
        - .csv (données exportées)
        - Tout autre format texte

    Note:
        Le fichier est toujours créé avec un timestamp pour éviter l'écrasement
    """
    # ============================================================
    # CRÉATION DU RÉPERTOIRE DE SORTIE
    # ============================================================
    create_domain(cwd + '/reports/')

    # ============================================================
    # GÉNÉRATION DU NOM DE FICHIER
    # ============================================================
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename1 = cwd + '/reports' + '/' + filename + timestr + extension

    # ============================================================
    # ÉCRITURE DU FICHIER
    # ============================================================
    # Ouverture en mode écriture ('w')
    _file = open(filename1, "w")

    # Écriture du contenu
    _file.write(content)

    # Fermeture du fichier (important!)
    _file.close()

    return filename1
