"""
===============================================================================
FICHIER: 03_graph_construction.py
===============================================================================
Auteur: DJIEMBOU TIENTCHEU Victor Nico
Date de création: 28/11/2023
Dernière modification: 11/06/2025

DESCRIPTION:
    Ce script gère la construction et l'analyse de graphes multicouches pour
    l'extraction de descripteurs dans le cadre d'un système d'apprentissage
    automatique. Il implémente trois modes principaux:
    - MLNA-1: Analyse monocouche (une variable à la fois)
    - MLNA-K: Analyse combinatoire (k=2 variables)
    - MLNA-TOP-K: Analyse des meilleures k variables sélectionnées

DÉPENDANCES:
    - networkx: Pour la manipulation de graphes
    - pandas: Pour la manipulation de données
    - numpy: Pour les calculs numériques
    - modules.preprocessing: Fonctions de prétraitement
    - modules.file: Fonctions de manipulation de fichiers
    - modules.graph: Fonctions de construction de graphes

STRUCTURE:
    1. Extraction de descripteurs depuis les graphes (PageRank)
    2. Génération de configurations de données
    3. Construction MLNA-1 (monocouche)
    4. Construction MLNA-K (multicouche combinatoire)
    5. Construction MLNA-TOP-K (variables sélectionnées)
    6. Fonction principale main()

UTILISATION:
    python 03_graph_construction.py --cwd <repertoire> --dataset_folder <nom>
           --alpha <valeur> --turn <numero> [--graph_with_class]
===============================================================================
"""

# ============================================================================
# IMPORTATIONS
# ============================================================================

from .cpu_limitation_usage import *
import sys

# Ajout du répertoire parent pour importer les modules personnalisés
sys.path.append('..')

# Modules de prétraitement et manipulation de données
from modules.preprocessing import get_combinations  # Génération de combinaisons
from modules.file import *   # Fonctions de gestion de fichiers
from modules.graph import *  # Fonctions de construction de graphes
import statistics # Calculs statistiques
import numpy as np
import scipy.sparse as sp



# ============================================================================
# FONCTION: extract_descriptors_from_graph_model
# ============================================================================


def extract_descriptors_from_graph_model(
    graph=None,           # Graphe multicouche sans classe
    graphWithClass=None,  # Flag indiquant si on utilise les classes
    alpha=.85,            # Facteur d'amortissement pour PageRank
    borrower=None,        # ID de l'emprunteur analysé
    layers=1              # Nombre de couches du graphe
):
    """
    Extrait les descripteurs basés sur le graphe pour un emprunteur donné.

    Cette fonction calcule deux types de descripteurs:
    1. Descripteurs GLOBAUX (GLO): Calculés sur l'ensemble du graphe
    2. Descripteurs PERSONNALISÉS (PER): Calculés avec personnalisation pour l'emprunteur

    Les descripteurs incluent:
    - DEGREE: Degré (nombre de voisins similaires)
    - INTRA: PageRank intra-couche (nœuds modalités)
    - INTER: PageRank inter-couche (nœuds emprunteurs)
    - COMBINE: PageRank combiné
    - M_*: Maximum des scores de modalités

    Paramètres:
    -----------
    graph : networkx.DiGraph
        Graphe multicouche représentant les relations emprunteur-modalité
    graphWithClass : bool
        Si True, inclut les descripteurs basés sur les classes
    alpha : float (défaut=0.85)
        Facteur d'amortissement PageRank (probabilité de continuer la marche)
        Valeur standard: 0.85 (Google)
    borrower : int
        Identifiant de l'emprunteur pour lequel extraire les descripteurs
    layers : int (défaut=1)
        Nombre de couches du graphe à considérer

    Retourne:
    ---------
    descriptors : dict
        Dictionnaire contenant tous les descripteurs calculés
        Format des clés: 'Att_<TYPE>_<CONTEXT>_<MODE>_'
        - TYPE: DEGREE, INTRA, INTER, COMBINE, M_INTRA, M_INTER, M_COMBINE
        - CONTEXT: GLO (global), PER (personnalisé)
        - MODE: MX (sans classe), CX (avec classe)

    Exemple de descripteurs retournés:
        {
            'Att_DEGREE_GLO': 10,                  # 10 emprunteurs similaires
            'Att_INTRA_GLO_MX_': 0.0234,          # Score PageRank intra
            'Att_M_COMBINE_PER_CX_': 0.0456,      # Score max personnalisé
            'YN_COMBINE_PER': 0.678,               # Prob. classe négative
            'YP_COMBINE_PER': 0.322                # Prob. classe positive
        }

    Notes:
    ------
    - Les descripteurs avec suffixe '_M_' représentent le maximum des modalités
    - Les descripteurs YN/YP ne sont calculés que si graphWithClass=True
    - La normalisation des scores est gérée par standard_extraction()
    """

    # Initialisation du dictionnaire de descripteurs
    descriptors = {}

    ################################
    ####### Descripteurs Globaux ###
    ################################
    nodes_combine, nodes_intra, nodes_inter = get_all_perso_nodes_labels(graph, borrower, layers)
    # DEGREE: Nombre d'emprunteurs ayant les mêmes valeurs de modalités
    # Pour une seule couche (layers=1), on compte les voisins directs
    # Pour plusieurs couches, on compte les emprunteurs partageant toutes les modalités
    descriptors['Att_DEGREE_GLO'] = (
        get_number_of_borrowers_with_same_n_layer_value(
                borrower=borrower,
                graph=graph,
                layer_nber=0  # Couche 0 pour analyse simple
           )[1]  # [1] donne le compte, [0] donne la liste
        if layers == 1 else
        get_number_of_borrowers_with_same_custom_layer_value(
            borrower=borrower,
            graph=graph,
            custom_layer=list(range(layers)) # Toutes les couches spécifiées
        )[1]
    )
    # -----------------------------------------------------------------------
    # PageRank batché : 1 conversion graph→scipy + 1 SpMM par itération
    # au lieu de 6 nx.pagerank() séquentiels (6 conversions + 6×50 SpMV)
    # -----------------------------------------------------------------------
    node_list = list(graph.nodes())
    node_idx  = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    # Matrice de transition row-normalisée (une seule fois)
    A = nx.to_scipy_sparse_array(graph, nodelist=node_list, format='csr', dtype=float)
    out_deg = np.array(A.sum(axis=1)).flatten()
    is_dangling = np.where(out_deg == 0)[0]
    inv_deg = np.where(out_deg > 0, 1.0 / out_deg, 0.0)
    A_norm = sp.diags(inv_deg) @ A   # row-normalisée
    AT     = A_norm.T.tocsr()        # pour x_new = AT @ x_old (right-multiply)

    # Vecteur de personnalisation pour chaque colonne : p(nodes) = 1/|nodes| si dans nodes, 0 sinon
    def _make_p(nodes_for_perso):
        if not nodes_for_perso:
            return np.full(N, 1.0 / N)
        p = np.zeros(N)
        w = 1.0 / len(nodes_for_perso)
        for n in nodes_for_perso:
            if n in node_idx:
                p[node_idx[n]] = w
        return p

    all_intra = [n for n in node_list if '-M-' in n]
    all_inter = [n for n in node_list if '-U-' in n]

    # Matrice de personnalisation P : (N, 6)
    # col 0 → combine_GLO (uniforme)   col 3 → combine_PER (voisins du borrower)
    # col 1 → intra_GLO  (tous -M-)    col 4 → inter_PER
    # col 2 → inter_GLO  (tous -U-)    col 5 → intra_PER
    P = np.column_stack([
        _make_p([]),           # col 0
        _make_p(all_intra),    # col 1
        _make_p(all_inter),    # col 2
        _make_p(nodes_combine),# col 3
        _make_p(nodes_inter),  # col 4
        _make_p(nodes_intra),  # col 5
    ])

    # Power iteration batché — même formule que NetworkX 3.x _pagerank_scipy
    # x_new = alpha * (AT @ x + dangling_sum * p) + (1-alpha) * p
    X = P.copy()
    for _ in range(500):
        D = X[is_dangling, :].sum(axis=0)                          # (6,)
        X_new = alpha * (AT.dot(X) + P * D[np.newaxis, :]) + (1.0 - alpha) * P
        if np.abs(X_new - X).sum(axis=0).max() < N * 1e-6:
            break
        X = X_new

    # Reconstruction des dicts pour compatibilité avec les fonctions aval
    bipart_combine        = dict(zip(node_list, X[:, 0]))
    bipart_intra_pagerank = dict(zip(node_list, X[:, 1]))
    bipart_inter_pagerank = dict(zip(node_list, X[:, 2]))
    _pr_perso_combine     = dict(zip(node_list, X[:, 3]))
    _pr_perso_inter       = dict(zip(node_list, X[:, 4]))
    _pr_perso_intra       = dict(zip(node_list, X[:, 5]))

    # ref = nx.pagerank(graph, alpha=alpha, max_iter=500)
    # custom = dict(zip(node_list, X[:, 0]))
    # max_diff = max(abs(ref[n] - custom[n]) for n in node_list)
    # status = "✅ ISO" if max_diff < 1e-5 else f"❌ Ecart : {max_diff:.2e}"
    # print(f"[PageRank batch] max_diff={max_diff:.2e} | {status}")
    # -----------------------------------------------------------------------

    suffix = "CX" if graphWithClass else "MX"

    descriptors[f'Att_INTRA_GLO_{suffix}_']   = get_max_modality_pagerank_score(borrower, graph, layers, bipart_intra_pagerank)
    descriptors[f'Att_INTER_GLO_{suffix}_']   = get_max_modality_pagerank_score(borrower, graph, layers, bipart_inter_pagerank)
    descriptors[f'Att_COMBINE_GLO_{suffix}_'] = get_max_modality_pagerank_score(borrower, graph, layers, bipart_combine)
    descriptors[f'Att_M_INTRA_GLO_{suffix}_'] = get_max_borrower_pr(bipart_intra_pagerank, target=borrower)
    descriptors[f'Att_M_INTER_GLO_{suffix}_'] = get_max_borrower_pr(bipart_inter_pagerank, target=borrower)
    descriptors[f'Att_M_COMBINE_GLO_{suffix}_'] = get_max_borrower_pr(bipart_combine, target=borrower)

    descriptors['Att_DEGREE_PER'] = descriptors['Att_DEGREE_GLO']

    descriptors[f'Att_COMBINE_PER_{suffix}_'] = get_max_borrower_pr(_pr_perso_combine, target=borrower)
    if graphWithClass is True:
        _class_scores = get_class_pr(_pr_perso_combine)
        descriptors['YN_COMBINE_PER'] = _class_scores[0]
        descriptors['YP_COMBINE_PER'] = _class_scores[1]

    descriptors[f'Att_INTER_PER_{suffix}_'] = get_max_borrower_pr(_pr_perso_inter, target=borrower)
    if graphWithClass is True:
        _class_inter = get_class_pr(_pr_perso_inter)
        descriptors['YN_INTER_PER'] = _class_inter[0]
        descriptors['YP_INTER_PER'] = _class_inter[1]

    descriptors[f'Att_INTRA_PER_{suffix}_'] = get_max_borrower_pr(_pr_perso_intra, target=borrower)
    if graphWithClass is True:
        _class_intra = get_class_pr(_pr_perso_intra)
        descriptors['YN_INTRA_PER'] = _class_intra[0]
        descriptors['YP_INTRA_PER'] = _class_intra[1]

    descriptors[f'Att_M_COMBINE_PER_{suffix}_'] = get_max_modality_pagerank_score(borrower, graph, 1, _pr_perso_combine)
    descriptors[f'Att_M_INTER_PER_{suffix}_']   = get_max_modality_pagerank_score(borrower, graph, 1, _pr_perso_inter)
    descriptors[f'Att_M_INTRA_PER_{suffix}_']   = get_max_modality_pagerank_score(borrower, graph, 1, _pr_perso_intra)

    return descriptors

# ============================================================================
# FONCTION: generate_config_df
# ============================================================================

def generate_config_df(
    graphWithClass=False,  # Utilisation de graphes avec classes
    mlnL='/mlna_1',        # Niveau MLNA (mlna_1, mlna_2, etc.)
    cwd=None,              # Répertoire de travail
    root=None,             # Répertoire racine
    domain=None,           # Nom du domaine/dataset
    extracts_g=None,       # Descripteurs globaux (train)
    extracts_p=None,       # Descripteurs personnalisés (train)
    extracts_g_t=None,     # Descripteurs globaux (test)
    extracts_p_t=None,     # Descripteurs personnalisés (test)
    name=None,             # Nom de la configuration
):
    """
    Génère et sauvegarde les configurations de descripteurs extraits.

    Cette fonction organise les descripteurs en DataFrames selon leur type
    (global/personnalisé, avec/sans classe) et les sauvegarde dans la structure
    de répertoires appropriée.

    Paramètres:
    -----------
    graphWithClass : bool
        Si True, génère des configurations incluant les descripteurs de classe (CY, CXY)
    mlnL : str
        Niveau du graphe multicouche ('/mlna_1', '/mlna_2', '/mlna_k_b')
    cwd : str
        Chemin du répertoire de travail courant
    root : str
        Chemin du répertoire racine du projet
    domain : str
        Nom du domaine/dataset (ex: 'german', 'australian')
    extracts_g : dict
        Descripteurs globaux pour l'ensemble d'entraînement
        Format: {'Att_DEGREE_GLO': [val1, val2, ...], ...}
    extracts_p : dict
        Descripteurs personnalisés pour l'ensemble d'entraînement
    extracts_g_t : dict
        Descripteurs globaux pour l'ensemble de test
    extracts_p_t : dict
        Descripteurs personnalisés pour l'ensemble de test
    name : str
        Nom de la variable/configuration (ex: nom de l'attribut analysé)

    Structure de sauvegarde:
    ------------------------
    cwd/mlna_X/variable_name/
        ├── global/
        │   ├── withClass/
        │   │   └── descriptors/  # GLO_CX
        │   └── withoutClass/
        │       └── descriptors/  # GLO_MX
        ├── personalized/
        │   ├── withClass/
        │   │   └── descriptors/  # PER_CX, PER_CY, PER_CXY
        │   └── withoutClass/
        │       └── descriptors/  # PER_MX
        └── config_df_for_variable_name.conf

    Retourne:
    ---------
    None (les fichiers sont sauvegardés sur disque)

    Fichiers générés:
    -----------------
    - DataFrames CSV: Descripteurs organisés par type
    - Fichier .conf: Configuration complète pour reconstruction ultérieure
    """

    # Initialisation du dictionnaire de configuration
    config_df = dict()

    """
    ====================
    Données d'entraînement
    ====================
    """

    # DataFrame des descripteurs GLOBAUX (train)
    # Préfixe: C=avec classe, M=sans classe
    config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df'] = pd.DataFrame(extracts_g)

    # DataFrame des descripteurs PERSONNALISÉS X (train)
    # Filtre: exclut les descripteurs de classe (YN_, YP_)
    config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df'] = pd.DataFrame({key: extracts_p[key] for key in extracts_p.keys() if not ("Y" in key)})

    # DataFrames supplémentaires si graphe avec classe
    if graphWithClass is True:
        # Descripteurs de CLASSE uniquement (YN_, YP_)
        config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df'] = pd.DataFrame({key: extracts_p[key] for key in extracts_p.keys() if ("Y" in key)})

        # Tous les descripteurs personnalisés (X + Y)
        config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df'] = pd.DataFrame(extracts_p)

    """
    ====================
    Données de test
    ====================
    """

    # Même structure que pour l'entraînement, mais avec suffixe '_t'
    config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df_t'] = pd.DataFrame(extracts_g_t)
    config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df_t'] = pd.DataFrame({key: extracts_p_t[key] for key in extracts_p_t.keys() if not ("Y" in key)})
    if graphWithClass is True:
        config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df_t'] = pd.DataFrame({key: extracts_p_t[key] for key in extracts_p_t.keys() if ("Y" in key)})
        config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df_t'] = pd.DataFrame(extracts_p_t)

    """
    ====================
    Sauvegarde des fichiers
    ====================
    """

    # Sauvegarde des descripteurs GLOBAUX (train)
    save_dataset(
        cwd=cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_GLO_df_for_{name}',
        prefix=domain,
        sep=',',
        sub='/descriptors'
    )

    # Sauvegarde des descripteurs PERSONNALISÉS X (train)
    save_dataset(
        cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_PER_df_for_{name}',
        prefix=domain,
        sep=',',
        sub='/descriptors'
    )

    # Sauvegarde des descripteurs GLOBAUX (test)
    save_dataset(
        cwd=cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df_t'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_GLO_df_t_for_{name}',
        prefix=domain,
        sep=',',
        sub='/descriptors'
    )

    # Sauvegarde des descripteurs PERSONNALISÉS X (test)
    save_dataset(
        cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df_t'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_PER_df_t_for_{name}',
        prefix=domain,
        sep=',',
        sub='/descriptors'
    )

    # Sauvegardes additionnelles si graphe avec classe
    if graphWithClass is True:
        # Descripteurs de classe Y (train)
        save_dataset(
            cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}Y_PER_df_for_{name}',
            prefix=domain,
            sep=',',
            sub='/descriptors'
        )

        # Tous descripteurs personnalisés XY (train)
        save_dataset(
            cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}XY_PER_df_for_{name}',
            prefix=domain,
            sep=',',
            sub='/descriptors'
        )

        # Descripteurs de classe Y (test)
        save_dataset(
            cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df_t'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}Y_PER_df_t_for_{name}',
            prefix=domain,
            sep=',',
            sub='/descriptors'
        )

        # Tous descripteurs personnalisés XY (test)
        save_dataset(
            cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df_t'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}XY_PER_df_t_for_{name}',
            prefix=domain,
            sep=',',
            sub='/descriptors'
        )

    # Sauvegarde de la configuration complète en fichier .conf
    # Permet de recharger tous les DataFrames ultérieurement
    save_model(
        cwd=cwd + f'{mlnL}/{name}',
        clf=config_df,
        prefix="",
        clf_name=f'config_df_for_{name}_{"withClass" if graphWithClass else "withoutClass"}',
        ext=".conf",
        sub=""
    )

# ============================================================================
# FONCTION: make_mlna_1_variable_v2
# ============================================================================

def make_mlna_1_variable_v2(
        x_traini,                    # Features d'entraînement
        x_testi,                     # Features de test
        y_traini,                    # Labels d'entraînement
        y_testi,                     # Labels de test
        OHE,                         # Liste des encodages one-hot par variable
        nominal_factor_colums,       # Noms des colonnes catégorielles
        cwd,                         # Répertoire de travail
        root,                        # Répertoire racine
        domain,                      # Nom du dataset
        target_variable,             # Nom de la variable cible
        alpha,                       # Facteur d'amortissement PageRank
        graphWithClass=False         # Inclure les classes dans le graphe
):
    """
    Construction MLNA-1: Analyse monocouche (une variable à la fois).

    Cette fonction construit un graphe multicouche pour CHAQUE variable catégorielle
    individuellement et extrait les descripteurs correspondants. C'est la première
    étape du protocole MLNA qui permet d'évaluer l'importance de chaque variable.

    Algorithme:
    -----------
    1. Pour chaque variable i dans nominal_factor_colums:
        a. Construire le graphe multicouche MLN avec une seule couche
        b. Sauvegarder le graphe
        c. Pour chaque emprunteur du train:
            - Extraire descripteurs globaux et personnalisés
            - Retirer l'arête vers sa classe (leave-one-out)
        d. Pour chaque emprunteur du test:
            - Ajouter l'emprunteur au graphe
            - Extraire descripteurs globaux et personnalisés
        e. Normaliser tous les descripteurs
        f. Générer et sauvegarder la configuration

    Paramètres:
    -----------
    x_traini, x_testi : pd.DataFrame
        Ensembles d'entraînement et de test (features uniquement)
    y_traini, y_testi : pd.Series
        Labels correspondants
    OHE : list of np.ndarray
        Liste des colonnes encodées one-hot pour chaque variable
        Exemple: [array(['cat1_A', 'cat1_B']), array(['cat2_X', 'cat2_Y', 'cat2_Z'])]
    nominal_factor_colums : list of str
        Noms originaux des variables catégorielles
        Exemple: ['categorical_var1', 'categorical_var2']
    cwd : str
        Chemin du répertoire de résultats (ex: 'results/german/0.85/cat')
    root : str
        Racine du projet
    domain : str
        Nom du dataset
    target_variable : str
        Nom de la variable cible (ex: 'loan_status')
    alpha : float
        Facteur d'amortissement pour PageRank (généralement 0.85)
    graphWithClass : bool
        Si True, construit des graphes incluant les nœuds de classe

    Structure des descripteurs extraits:
    -------------------------------------
    extracts_g (globaux):
        - Att_DEGREE_GLO: Degré des nœuds
        - Att_INTRA_GLO_XX_: PageRank intra-couche
        - Att_INTER_GLO_XX_: PageRank inter-couche
        - Att_COMBINE_GLO_XX_: PageRank combiné
        - Att_M_*_GLO_XX_: Maximum des modalités
        (XX = CX avec classe, MX sans classe)

    extracts_p (personnalisés):
        - Att_DEGREE_PER: Degré personnalisé
        - Att_*_PER_XX_: PageRank personnalisés
        - YN_*_PER, YP_*_PER: Probabilités de classe (si graphWithClass)
        - Att_M_*_PER_XX_: Maximum des modalités personnalisé

    Retourne:
    ---------
    None (les résultats sont sauvegardés sur disque)

    Fichiers générés pour chaque variable:
    ---------------------------------------
    cwd/mlna_1/variable_name/
        ├── variable_name_mln.graphml          # Graphe NetworkX
        ├── global/withoutClass/descriptors/   # Descripteurs globaux
        ├── personalized/withoutClass/descriptors/  # Descripteurs personnalisés
        └── config_df_for_variable_name.conf   # Configuration

    Notes:
    ------
    - Utilise le leave-one-out pour l'entraînement (retire l'arête vers la classe)
    - Normalise les descripteurs avec standard_extraction()
    - Crée un fichier flag à la fin pour éviter les recalculs
    """

    ## Copie des données pour éviter les modifications
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi

    # Extraction des identifiants d'emprunteurs (indices des DataFrames)
    PERSONS = get_persons(x_train)
    PERSONS_T = get_persons(x_test)

    # Préparation des données: fusion X_train + y_train
    copT = x_train.copy(deep=True)
    copT[target_variable] = y_train.copy(deep=True)
    # Boucle sur chaque variable catégorielle
    for i in range(len(OHE)):
        # Vérification si cette variable a déjà été traitée
        # Recherche du fichier flag de complétion
        if sum([f'config_df_for_{nominal_factor_colums[i]}_{"withClass" if graphWithClass else "withoutClass"}' in file for _, _, files in
             os.walk(cwd + f'/mlna_1/{nominal_factor_colums[i]}/') for
             file in files]) > 0 :
            continue # Variable déjà traitée, passer à la suivante



        # Construction du graphe multicouche
        # build_mlg_with_class: inclut les nœuds de classe
        # build_mlg: graphe simple sans classe
        MLN = build_mlg_with_class(
            copT,
            [OHE[i]], # Une seule couche (variable i)
            target_variable
        ) if (graphWithClass is True) else build_mlg(
            copT,
            [OHE[i]]
        )

        # Sauvegarde du graphe en format GraphML
        save_graph(
            cwd=cwd + f'/mlna_1/{nominal_factor_colums[i]}',
            graph=MLN,
            name=f'{nominal_factor_colums[i]}_mln',
            rows_len=pd.concat([x_train, y_train], axis=1).shape[0],
            prefix=domain,
            cols_len=len(OHE[i])
        )

        # Initialisation des dictionnaires de descripteurs
        # Structure: {nom_descripteur: [valeur_emp1, valeur_emp2, ...]}
        extracts_g = {
            "Att_DEGREE_GLO" : [],
            "Att_INTRA_GLO_CX_" : [],
            "Att_INTRA_GLO_MX_" : [],
            "Att_INTER_GLO_CX_" : [],
            "Att_INTER_GLO_MX_" : [],
            "Att_COMBINE_GLO_CX_" : [],
            "Att_COMBINE_GLO_MX_" : [],
            "Att_M_INTRA_GLO_MX_" : [],
            "Att_M_INTRA_GLO_CX_" : [],
            "Att_M_INTER_GLO_MX_" : [],
            "Att_M_INTER_GLO_CX_" : [],
            "Att_M_COMBINE_GLO_MX_" : [],
            "Att_M_COMBINE_GLO_CX_": []
        }
        extracts_p = {
            "Att_DEGREE_PER" : [],      # Degré
            "Att_COMBINE_PER_CX_" : [],
            "Att_COMBINE_PER_MX_" : [],
            "YN_COMBINE_PER" : [],      # Classe négative
            "YP_COMBINE_PER" : [],      # Classe positive
            "Att_INTER_PER_MX_" : [],
            "Att_INTER_PER_CX_" : [],
            "YN_INTER_PER" : [],
            "YP_INTER_PER" : [],
            "Att_INTRA_PER_CX_" : [],
            "Att_INTRA_PER_MX_" : [],
            "YN_INTRA_PER" : [],
            "YP_INTRA_PER" : [],
            "Att_M_COMBINE_PER_CX_" : [],
            "Att_M_COMBINE_PER_MX_" : [],
            "Att_M_INTER_PER_CX_" : [],
            "Att_M_INTER_PER_MX_" : [],
            "Att_M_INTRA_PER_CX_" : [],
            "Att_M_INTRA_PER_MX_": []
        }

        # Copie profonde pour les descripteurs de test
        extracts_g_t = copy.deepcopy(extracts_g)
        extracts_p_t = copy.deepcopy(extracts_p)


        ##################################
        ####### Training Descriptor ######
        ##################################
        for borrower in PERSONS:
            # Extraction des descripteurs pour l'emprunteur courant
            # removeEdge: retire l'arête vers la classe (leave-one-out)
            removeEdge(MLN, 1, copT.loc[borrower, target_variable], borrower)

            current = extract_descriptors_from_graph_model(
                graph=MLN,
                graphWithClass=graphWithClass,
                alpha=alpha,
                borrower=borrower,
                layers=1
            )

            # Restaure l'arête sur MLN
            addEdge(MLN, 1, copT.loc[borrower, target_variable], borrower)

            # Répartition des descripteurs entre globaux et personnalisés
            for key in list(current.keys()):
                if 'GLO' in key:
                    extracts_g[key].append(current[key])
                else:
                    extracts_p[key].append(current[key])

            # Libération mémoire
            current = None


        ##################################
        ####### Test Descriptor     ######
        ##################################
        for borrower in PERSONS_T:
            # Ajout de l'emprunteur test au graphe
            borrower_nodes, new_modality_nodes = add_specific_loan_in_mlg(
                MLN,
                x_test.loc[[borrower]],
                [OHE[i]]
            )

            # Extraction des descripteurs
            current = extract_descriptors_from_graph_model(
                graph=MLN,
                graphWithClass=graphWithClass,
                alpha=alpha,
                borrower=borrower,
                layers=1
            )
            remove_specific_loan_from_mlg(MLN, borrower_nodes, new_modality_nodes)

            # Répartition des descripteurs
            for key in list(current.keys()):
                if 'GLO' in key:
                    extracts_g_t[key].append(current[key])
                else:
                    extracts_p_t[key].append(current[key])
            current = None

        ########################################
        ####### Descriptors Normalisation ######
        ########################################

        # Nettoyage des descripteurs non utilisés selon le mode
        if graphWithClass is False:
            # Mode sans classe: supprimer descripteurs Y et CX
            for key in list(extracts_g.keys()):
                if 'Y' in key or 'CX_' in key:
                    del extracts_g_t[key]
                    del extracts_g[key]
            for key in list(extracts_p.keys()):
                if 'Y' in key or 'CX_' in key:
                    del extracts_p[key]
                    del extracts_p_t[key]

        if graphWithClass is True:
            # Mode avec classe: supprimer descripteurs MX
            for key in list(extracts_g_t.keys()):
                if 'MX_' in key:
                    del extracts_g_t[key]
                    del extracts_g[key]
            for key in list(extracts_p_t.keys()):
                if 'MX_' in key:
                    del extracts_p[key]
                    del extracts_p_t[key]

        # print(extracts_g.keys(),'++', extracts_p.keys(),'++',
        #       extracts_g_t.keys(),'++', extracts_p_t.keys())

        # Normalisation: mise à l'échelle [0, 1]
        # Récupération des valeurs maximales de l'ensemble d'entraînement
        maxGDesc = standard_extraction(extracts_g, extracts_g.keys())
        maxPDesc = standard_extraction(extracts_p, extracts_p.keys())
        # print(f"{maxGDesc} <------> {maxPDesc}")

        # Application de la même normalisation au test
        standard_extraction(extracts_g_t, extracts_g.keys(),maxGDesc)
        standard_extraction(extracts_p_t, extracts_p.keys(),maxPDesc)


        ##########################################
        ####### Génération Configuration #########
        ##########################################
        config_df = generate_config_df(
            cwd=cwd,
            root=root,
            graphWithClass=graphWithClass,
            mlnL='/mlna_1',
            domain=domain,
            extracts_g=extracts_g,
            extracts_p=extracts_p,
            extracts_g_t=extracts_g_t,
            extracts_p_t=extracts_p_t,
            name=nominal_factor_colums[i]
        )


# ============================================================================
# FONCTION: make_mlna_k_variable_v2
# ============================================================================

def make_mlna_k_variable_v2(
        x_traini,                    # Features d'entraînement
        x_testi,                     # Features de test
        y_traini,                    # Labels d'entraînement
        y_testi,                     # Labels de test
        OHE,                         # Liste des encodages one-hot
        nominal_factor_colums,       # Noms des colonnes
        cwd,                         # Répertoire de travail
        root,                        # Répertoire racine
        domain,                      # Nom du dataset
        alpha,                       # Facteur PageRank
        target_variable,             # Variable cible
        graphWithClass=True          # Inclure classes
):
    """
    Construction MLNA-K: Analyse multicouche combinatoire (k=2 variables).

    Cette fonction explore toutes les combinaisons possibles de 2 variables
    pour construire des graphes multicouches à 2 couches. Cela permet d'identifier
    les interactions entre paires de variables.

    Algorithme:
    -----------
    1. Fixer k=2 (nombre de couches)
    2. Pour chaque combinaison de 2 variables parmi OHE:
        a. Construire le graphe multicouche avec 2 couches
        b. Sauvegarder le graphe
        c. Extraire descripteurs pour train et test
        d. Normaliser et sauvegarder

    Différences avec MLNA-1:
    -------------------------
    - Graphes à 2 couches au lieu de 1
    - Exploration combinatoire: C(n,2) graphes où n=nombre de variables
    - Descripteurs capturent les interactions entre variables
    - Utilise removeEdge différemment (k=2 au lieu de k=1)

    Paramètres:
    -----------
    (Mêmes paramètres que make_mlna_1_variable_v2)

    Exemple de combinaisons:
    ------------------------
    Si OHE contient 4 variables [A, B, C, D]:
    Combinaisons générées:
    - (A, B), (A, C), (A, D)
    - (B, C), (B, D)
    - (C, D)
    Total: C(4,2) = 6 graphes

    Structure de sortie:
    --------------------
    cwd/mlna_2/variable1_variable2/
        ├── variable1_variable2_mln.graphml
        ├── global/withClass/descriptors/
        ├── personalized/withClass/descriptors/
        ├── mixed/withClass/descriptors/
        └── config_df_for_variable1_variable2.conf

    Notes:
    ------
    - Par défaut, graphWithClass=True pour capturer les effets de classe
    - La liste [2] peut être étendue à [2,3,4,...] pour k>2
    - Temps de calcul O(n²) où n=nombre de variables
    """

    ## Copie des données
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)
    PERSONS_T = get_persons(x_test)
    # Préparation des données
    copT = x_train.copy(deep=True)
    copT[target_variable] = y_train.copy(deep=True)

    # Boucle sur les valeurs de k (ici fixé à k=2)
    for k in list([2]): # Peut être étendu: range(2, len(OHE)+1)

        # Génération de toutes les combinaisons de k variables
        # get_combinations(range(len(OHE)), k) retourne les indices des variables
        for layer_config in get_combinations(range(len(OHE)), k):  # create subsets of k index of OHE and fetch it



            # Nom de la configuration: concaténation des noms de variables
            # Exemple: "home_ownership_loan_purpose"
            col_targeted = [f'{nominal_factor_colums[i]}' for i in layer_config]
            case_k = '_'.join(col_targeted)

            # Vérification si déjà traité
            if sum([
                       f'config_df_for_{case_k}_{"withClass" if graphWithClass else "withoutClass"}' in file
                       for _, _, files in
                       os.walk(cwd + f'/mlna_{k}/{case_k}/') for
                       file in files]) > 0:
                continue

            # Construction du graphe multicouche avec k=2 couches
            # [OHE[i] for i in layer_config]: sélection des colonnes one-hot
            MLN = build_mlg_with_class(
                copT,
                [OHE[i] for i in layer_config], # k couches
                target_variable
            ) if (graphWithClass is True) else build_mlg(
                copT,
                [OHE[i] for i in layer_config]
            )

            # Sauvegarde du graphe
            save_graph(
                cwd=cwd + f'/mlna_{k}/{case_k}',
                graph=MLN,
                name=f'{case_k}_mln',
                rows_len=pd.concat([x_train, y_train], axis=1).shape[0],
                prefix=domain,
                cols_len=len(OHE)
            )

            # Initialisation des dictionnaires de descripteurs (même structure que MLNA-1)
            extracts_g = {
                "Att_DEGREE_GLO": [],
                "Att_INTRA_GLO_CX_": [],
                "Att_INTRA_GLO_MX_": [],
                "Att_INTER_GLO_CX_": [],
                "Att_INTER_GLO_MX_": [],
                "Att_COMBINE_GLO_CX_": [],
                "Att_COMBINE_GLO_MX_": [],
                "Att_M_INTRA_GLO_MX_": [],
                "Att_M_INTRA_GLO_CX_": [],
                "Att_M_INTER_GLO_MX_": [],
                "Att_M_INTER_GLO_CX_": [],
                "Att_M_COMBINE_GLO_MX_": [],
                "Att_M_COMBINE_GLO_CX_": []
            }
            extracts_p = {
                "Att_DEGREE_PER": [],  # GLO
                "Att_COMBINE_PER_CX_": [],
                "Att_COMBINE_PER_MX_": [],
                "YN_COMBINE_PER": [],
                "YP_COMBINE_PER": [],  # COM
                "Att_INTER_PER_MX_": [],
                "Att_INTER_PER_CX_": [],
                "YN_INTER_PER": [],
                "YP_INTER_PER": [],  # INTER
                "Att_INTRA_PER_CX_": [],
                "Att_INTRA_PER_MX_": [],
                "YN_INTRA_PER": [],
                "YP_INTRA_PER": [],  # INTRA
                "Att_M_COMBINE_PER_CX_": [],
                "Att_M_COMBINE_PER_MX_": [],
                "Att_M_INTER_PER_CX_": [],
                "Att_M_INTER_PER_MX_": [],
                "Att_M_INTRA_PER_CX_": [],
                "Att_M_INTRA_PER_MX_": []
            }
            extracts_g_t = copy.deepcopy(extracts_g)
            extracts_p_t = copy.deepcopy(extracts_p)

            ##################################
            ####### Descripteurs Train #######
            ##################################
            for borrower in PERSONS:
                # Extraction avec layers=k
                removeEdge(MLN, k, copT.loc[borrower, target_variable], borrower)
                current = extract_descriptors_from_graph_model(
                    graph=MLN,
                    graphWithClass=graphWithClass,
                    alpha=alpha,
                    borrower=borrower,
                    layers=k
                )
                # Restaure l'arête sur MLN
                addEdge(MLN, k, copT.loc[borrower, target_variable], borrower)

                # Répartition des descripteurs
                for key in list(current.keys()):
                    if 'GLO' in key:
                        extracts_g[key].append(current[key])
                    else:
                        extracts_p[key].append(current[key])

            ##################################
            ####### Descripteurs Test ########
            ##################################
            for borrower in PERSONS_T:
                # Ajout de l'emprunteur au graphe avec les k couches
                borrower_nodes, new_modality_nodes = add_specific_loan_in_mlg(
                    MLN,
                    x_test.loc[[borrower]],
                    [OHE[i] for i in layer_config]
                )

                current = extract_descriptors_from_graph_model(
                    graph=MLN,
                    graphWithClass=graphWithClass,
                    alpha=alpha,
                    borrower=borrower,
                    layers=k
                )
                remove_specific_loan_from_mlg(MLN, borrower_nodes, new_modality_nodes)


                for key in list(current.keys()):
                    if 'GLO' in key:
                        extracts_g_t[key].append(current[key])
                    else:
                        extracts_p_t[key].append(current[key])


            ########################################
            ####### Normalisation Descripteurs #####
            ########################################
            if graphWithClass is False:
                for key in list(extracts_g.keys()):
                    if 'Y' in key or 'CX_' in key:
                        del extracts_g_t[key]
                        del extracts_g[key]
                for key in list(extracts_p.keys()):
                    if 'Y' in key or 'CX_' in key:
                        del extracts_p[key]
                        del extracts_p_t[key]

            if graphWithClass is True:
                for key in list(extracts_g_t.keys()):
                    if 'MX_' in key:
                        del extracts_g_t[key]
                        del extracts_g[key]
                for key in list(extracts_p_t.keys()):
                    if 'MX_' in key:
                        del extracts_p[key]
                        del extracts_p_t[key]

            # Normalisation
            maxGDesc = standard_extraction(extracts_g, extracts_g.keys())
            maxPDesc = standard_extraction(extracts_p, extracts_p.keys())
            standard_extraction(extracts_g_t, extracts_g.keys(), maxGDesc)
            standard_extraction(extracts_p_t, extracts_p.keys(), maxPDesc)

            ##########################################
            ####### Génération Configuration #########
            ##########################################
            config_df = generate_config_df(
                cwd=cwd,
                root=root,
                graphWithClass=graphWithClass,
                mlnL=f'/mlna_{k}',
                domain=domain,
                extracts_g=extracts_g,
                extracts_p=extracts_p,
                extracts_g_t=extracts_g_t,
                extracts_p_t=extracts_p_t,
                name=case_k
            )

# ============================================================================
# FONCTION: make_mlna_top_k_variable_v2
# ============================================================================

def make_mlna_top_k_variable_v2(
        x_traini,                    # Features d'entraînement
        x_testi,                     # Features de test
        y_traini,                    # Labels d'entraînement
        y_testi,                     # Labels de test
        OHE,                         # Liste des encodages one-hot
        nominal_factor_colums,       # Noms des colonnes
        cwd,                         # Répertoire de travail
        root,                        # Répertoire racine
        domain,                      # Nom du dataset
        target_variable,             # Variable cible
        alpha,                       # Facteur PageRank
        graphWithClass=False,        # Inclure classes
        topR=[]                      # Liste ordonnée des meilleures variables
):
    """
     Construction MLNA-TOP-K: Analyse des meilleures k variables sélectionnées.

     Cette fonction construit des graphes multicouches en ajoutant progressivement
     les k meilleures variables identifiées par le protocole de sélection MNIFS.
     Elle permet d'évaluer l'effet cumulatif des meilleures variables.

     Algorithme:
     -----------
     1. Récupérer les indices des variables dans topR (ordre décroissant d'importance)
     2. Pour k de 2 à len(topR):
         a. Sélectionner les k premières variables de topR
         b. Construire graphe multicouche avec ces k variables
         c. Extraire descripteurs pour train et test
         d. Normaliser et sauvegarder

     Différences avec MLNA-K:
     -------------------------
     - Utilise un ordre SPÉCIFIQUE de variables (topR)
     - Construction incrémentale: k=2, k=3, k=4, ...
     - Pas de combinaisons: toujours les k meilleures
     - Sauvegarde dans '/mlna_k_b' (b = best)

     Paramètres:
     -----------
     topR : list of int
         Indices des variables ordonnées par importance décroissante
         Exemple: [3, 0, 5, 1] signifie var3 > var0 > var5 > var1
         Obtenu depuis MNIFS (protocole de sélection)

     (Autres paramètres identiques aux fonctions précédentes)

     Exemple d'utilisation:
     ----------------------
     Si topR = [2, 5, 1, 4] (4 meilleures variables):

     Itération k=2:
         Variables: [2, 5]
         Graphe: mlna_2_b/var2_var5/

     Itération k=3:
         Variables: [2, 5, 1]
         Graphe: mlna_3_b/var2_var5_var1/

     Itération k=4:
         Variables: [2, 5, 1, 4]
         Graphe: mlna_4_b/var2_var5_var1_var4/

     Structure de sortie:
     --------------------
     cwd/select/mlna_k_b/top_k_vars/
         ├── top_k_vars_mln.graphml
         ├── global/withClass/descriptors/
         ├── personalized/withClass/descriptors/
         ├── mixed/withClass/descriptors/
         └── config_df_for_top_k_vars.conf

     Notes:
     ------
     - Permet d'analyser la performance en fonction du nombre de variables
     - Identifie le k optimal (elbow point)
     - Plus efficace que MLNA-K car évite les combinaisons non pertinentes
     - Utilisé pour la modélisation finale
     """

    ## Copie des données
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)
    PERSONS_T = get_persons(x_test)
    # Préparation des données
    copT = x_train.copy(deep=True)
    copT[target_variable] = y_train.copy(deep=True)

    # Boucle sur k de 2 à len(topR)
    # Construction incrémentale des graphes
    for k in range(2, len(topR) + 1):  # for 1<k<|OHE[i]|+2

        # Sélection des k premières variables de topR
        # topR[:k] donne les indices des k meilleures variables
        layer_config = topR[:k]



        # Construction du nom: concaténation des k variables
        col_targeted = [f'{nominal_factor_colums[i]}' for i in layer_config]
        case_k = '_'.join(col_targeted)

        # Vérification si déjà traité
        if sum([
            f'config_df_for_{case_k}_{"withClass" if graphWithClass else "withoutClass"}' in file
            for _, _, files in
            os.walk(cwd + f'/mlna_{k}_b/{case_k}/') for
            file in files]) > 0:
            continue

        # Construction du graphe multicouche avec k couches
        MLN = build_mlg_with_class(
            copT,
            [OHE[i] for i in layer_config],
            target_variable) if (graphWithClass is True) else build_mlg(
            copT,
            [OHE[i] for i in layer_config]
        )

        # Sauvegarde du graphe
        save_graph(
            cwd=cwd + f'/mlna_{k}_b/{case_k}',
            graph=MLN,
            name=f'{case_k}_mln',
            rows_len=copT.shape[0],
            prefix=domain,
            cols_len=len(OHE)
        )

        # Initialisation des dictionnaires de descripteurs
        extracts_g = {
            "Att_DEGREE_GLO": [],
            "Att_INTRA_GLO_CX_": [],
            "Att_INTRA_GLO_MX_": [],
            "Att_INTER_GLO_CX_": [],
            "Att_INTER_GLO_MX_": [],
            "Att_COMBINE_GLO_CX_": [],
            "Att_COMBINE_GLO_MX_": [],
            "Att_M_INTRA_GLO_MX_": [],
            "Att_M_INTRA_GLO_CX_": [],
            "Att_M_INTER_GLO_MX_": [],
            "Att_M_INTER_GLO_CX_": [],
            "Att_M_COMBINE_GLO_MX_": [],
            "Att_M_COMBINE_GLO_CX_": []
        }
        extracts_p = {
            "Att_DEGREE_PER": [],  # GLO
            "Att_COMBINE_PER_CX_": [],
            "Att_COMBINE_PER_MX_": [],
            "YN_COMBINE_PER": [],
            "YP_COMBINE_PER": [],  # COM
            "Att_INTER_PER_MX_": [],
            "Att_INTER_PER_CX_": [],
            "YN_INTER_PER": [],
            "YP_INTER_PER": [],  # INTER
            "Att_INTRA_PER_CX_": [],
            "Att_INTRA_PER_MX_": [],
            "YN_INTRA_PER": [],
            "YP_INTRA_PER": [],  # INTRA
            "Att_M_COMBINE_PER_CX_": [],
            "Att_M_COMBINE_PER_MX_": [],
            "Att_M_INTER_PER_CX_": [],
            "Att_M_INTER_PER_MX_": [],
            "Att_M_INTRA_PER_CX_": [],
            "Att_M_INTRA_PER_MX_": []
        }
        extracts_g_t = copy.deepcopy(extracts_g)
        extracts_p_t = copy.deepcopy(extracts_p)

        ##################################
        ####### Descripteurs Train #######
        ##################################
        for borrower in PERSONS:
            # Extraction avec layers=k
            removeEdge(MLN, k, copT.loc[borrower, target_variable], borrower)
            current = extract_descriptors_from_graph_model(
                graph=MLN,
                graphWithClass=graphWithClass,
                alpha=alpha,
                borrower=borrower,
                layers=k
            )
            # Restaure l'arête sur MLN
            addEdge(MLN, k, copT.loc[borrower, target_variable], borrower)

            # Répartition des descripteurs
            for key in list(current.keys()):
                if 'GLO' in key:
                    extracts_g[key].append(current[key])
                else:
                    extracts_p[key].append(current[key])

        ##################################
        ####### Descripteurs Test ########
        ##################################
        for borrower in PERSONS_T:
            # Ajout de l'observation au graphe avec k couches
            borrower_nodes, new_modality_nodes = add_specific_loan_in_mlg(
                MLN,
                x_test.loc[[borrower]],
                [OHE[i] for i in layer_config]
            )

            current = extract_descriptors_from_graph_model(
                graph=MLN,
                graphWithClass=graphWithClass,
                alpha=alpha,
                borrower=borrower,
                layers=k
            )
            remove_specific_loan_from_mlg(MLN, borrower_nodes, new_modality_nodes)


            for key in list(current.keys()):
                if 'GLO' in key:
                    extracts_g_t[key].append(current[key])
                else:
                    extracts_p_t[key].append(current[key])

        ########################################
        ####### Normalisation Descripteurs #####
        ########################################
        if graphWithClass is False:
            for key in list(extracts_g_t.keys()):
                if 'Y' in key or 'CX_' in key:
                    del extracts_g_t[key]
                    del extracts_g[key]
            for key in list(extracts_p_t.keys()):
                if 'Y' in key or 'CX_' in key:
                    del extracts_p[key]
                    del extracts_p_t[key]

        if graphWithClass is True:
            for key in list(extracts_g_t.keys()):
                if 'MX_' in key:
                    del extracts_g_t[key]
                    del extracts_g[key]
            for key in list(extracts_p_t.keys()):
                if 'MX_' in key:
                    del extracts_p[key]
                    del extracts_p_t[key]

        # Normalisation
        maxGDesc = standard_extraction(extracts_g, extracts_g.keys())
        maxPDesc = standard_extraction(extracts_p, extracts_p.keys())
        standard_extraction(extracts_g_t, extracts_g.keys(), maxGDesc)
        standard_extraction(extracts_p_t, extracts_p.keys(), maxPDesc)


        ##########################################
        ####### Génération Configuration #########
        ##########################################
        config_df = generate_config_df(
            cwd=cwd,
            root=root,
            graphWithClass=graphWithClass,
            mlnL=f'/mlna_{k}_b',
            domain=domain,
            extracts_g=extracts_g,
            extracts_p=extracts_p,
            extracts_g_t=extracts_g_t,
            extracts_p_t=extracts_p_t,
            name=case_k
        )

# ============================================================================
# FONCTION PRINCIPALE: main
# ============================================================================

def main():
    """
    Point d'entrée principal du script de construction de graphes.

    Cette fonction gère l'exécution du pipeline MLNA en 3 tours (turns):

    TURN 1: Construction MLNA-1
        - Analyse monocouche (une variable à la fois)
        - Génère les graphes de base
        - Extrait les descripteurs initiaux
        - Fichier flag: graph_turn_1_completed.dtvni

    TURN 2: Construction MLNA-TOP-K
        - Utilise les résultats de MNIFS (sélection de variables)
        - Construit les graphes des meilleures k variables
        - Inclut les classes (graphWithClass=True)
        - Fichier flag: graph_turn_2_completed.dtvni

    TURN 3: Construction MLNA-2 (Combinatoire)
        - Explore toutes les paires de variables
        - Analyse les interactions
        - Fichier flag: graph_turn_3_completed.dtvni

    Arguments ligne de commande:
    ----------------------------
    --cwd : str
        Répertoire de travail courant (chemin absolu)
    --dataset_folder : str
        Nom du dossier du dataset (ex: 'german_credit')
    --alpha : float
        Facteur d'amortissement PageRank (0.85 recommandé)
    --turn : int
        Numéro du tour à exécuter (1, 2 ou 3)
    --graph_with_class : flag
        Si présent, inclut les nœuds de classe dans les graphes

    Fichiers de configuration requis:
    ----------------------------------
    - configs/{dataset_folder}/config.ini
    - processed/{dataset_folder}/preprocessing_*.pkl
    - splits/{dataset_folder}/{type}/train.csv et test.csv

    Exemples d'utilisation:
    -----------------------
    # Tour 1: Construction de base
    python 03_graph_construction.py \\
        --cwd /path/to/project \\
        --dataset_folder german_credit \\
        --alpha 0.85 \\
        --turn 1

    # Tour 2: Variables sélectionnées avec classes
    python 03_graph_construction.py \\
        --cwd /path/to/project \\
        --dataset_folder german_credit \\
        --alpha 0.85 \\
        --turn 2 \\
        --graph_with_class

    # Tour 3: Analyse combinatoire
    python 03_graph_construction.py \\
        --cwd /path/to/project \\
        --dataset_folder german_credit \\
        --alpha 0.85 \\
        --turn 3

    Flux de données:
    ----------------
    1. Chargement configuration (config.ini)
    2. Chargement données train/test
    3. Chargement config prétraitement
    4. Sélection OHE approprié selon target_columns_type
    5. Exécution du tour demandé
    6. Création fichier flag de complétion
    """

    # ========================================================================
    # PARSING DES ARGUMENTS
    # ========================================================================
    import argparse

    # Arguments obligatoires
    parser = argparse.ArgumentParser(description='Exécution du pipeline MLNA - SPLIT DATASET')
    parser.add_argument('--cwd', type=str, required=True, help='Répertoire de travail courant')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Nom du dataset')
    parser.add_argument('--alpha', type=float, required=True, help='Valeur d\'alpha')
    parser.add_argument('--turn', type=int, required=True, help='Valeur du tour')

    # Argument optionnel
    parser.add_argument('--graph_with_class', action="store_true", required=False, help='integrant les classes?')
    parser.add_argument('--metric', type=str, required=False, help='Nom de la metrique à analyser')

    # Récupération des arguments
    args = parser.parse_args()

    # ========================================================================
    # CHARGEMENT DE LA CONFIGURATION
    # ========================================================================
    config = load_config(f"{args.cwd}/configs/{args.dataset_folder}/config.ini")

    # Extraction des paramètres de configuration
    domain = config["DATA"]["domain"]  # Nom du dataset
    encoding = config["PREPROCESSING"]["encoding"]  # Encodage fichiers
    dataset_delimiter = config["SPLIT"]["dataset_delimiter"]  # Séparateur CSV
    target_variable = config["DATA"]["target"]  # Variable cible

    # Répertoires
    processed_dir = config["GENERAL"]["processed_dir"]  # Données prétraitées
    split_dir = config["GENERAL"]["split_dir"]  # Données splitées
    results_dir = config["GENERAL"]["results_dir"]  # Résultats
    target_columns_type = config["GENERAL"]["target_columns_type"]  # Type: cat/num/all

    # Options d'affichage
    verbose = config.getboolean("GENERAL", "verbose")

    # Colonne d'index (None si non spécifiée)
    index_col = None if config["SPLIT"]["index_col"] in ["None", ""] \
        else config.getint("SPLIT", "index_col")

    # ========================================================================
    # CHARGEMENT DES DONNÉES TRAIN/TEST
    # ========================================================================

    # Vérification de l'existence des fichiers split
    if sum([f'{domain}_train' in file for _, _, files in
            os.walk(args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}') for
            file in files]) == 0:
        print("❌ Unable to access splits data")
        exit(1)

    # Récupération des chemins des fichiers train et test
    # Recherche du fichier contenant '_train' dans son nom
    xtrain_path = args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}/' + [file for _, _, files in
         os.walk(
             args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}')
         for
         file in files][
        [f'{domain}_train' in file for _, _, files in
         os.walk(args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}') for
         file in files].index(True)
    ]

    # Recherche du fichier contenant '_test' dans son nom
    xtest_path = args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}/' + [file for _, _, files in
        os.walk(
            args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}')
        for
        file in files][
        [f'{domain}_test' in file for _, _, files in
         os.walk(args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}') for
         file in files].index(True)
    ]

    # Chargement des fichiers CSV
    if verbose:
        print("the xtrain path is ", xtrain_path)
        print("the xtest path is ", xtest_path)

    X_train = load_data_set_from_url(
        path=xtrain_path,
        sep=dataset_delimiter,
        encoding=encoding,
        index_col=index_col,
        na_values=None
    )

    # Séparation features/target pour train
    y_traini = X_train[target_variable]
    x_traini = X_train.drop(columns=[target_variable])

    X_test = load_data_set_from_url(
        path=xtest_path,
        sep=dataset_delimiter,
        encoding=encoding,
        index_col=index_col,
        na_values=None
    )

    # Séparation features/target pour test
    y_testi = X_test[target_variable]
    x_testi = X_test.drop(columns=[target_variable])

    if verbose:
        print("the x_train shape is ", X_train.shape)
        print("the x_test shape is ", X_test.shape)

    # ========================================================================
    # CHARGEMENT DE LA CONFIGURATION DE PRÉTRAITEMENT
    # ========================================================================

    # Vérification de l'existence du fichier de prétraitement
    if sum([f'preprocessing' in file for _, _, files in
            os.walk(args.cwd + f'/{processed_dir}{args.dataset_folder}') for
            file in files]) == 0:
        print("❌ Unable to access preprocessing config")
        exit(1)

    # Récupération du chemin du fichier de configuration prétraitement
    prepro_path = args.cwd + f'/{processed_dir}{args.dataset_folder}/' + [file for _, _, files in
       os.walk(
           args.cwd + f'/{processed_dir}{args.dataset_folder}/')
       for
       file in files][
        [f'preprocessing' in file for _, _, files in
         os.walk(args.cwd + f'/{processed_dir}{args.dataset_folder}/') for
         file in files].index(True)
    ]

    # Chargement de la configuration (format pickle)
    prepro_config = read_model(path=prepro_path)


    # ========================================================================
    # SÉLECTION DES VARIABLES SELON LE TYPE
    # ========================================================================

    # Sélection des encodages One-Hot et noms de colonnes selon le type
    if target_columns_type == "cat":
        # Uniquement variables catégorielles
        OHE = prepro_config["OHE"]
        columns = prepro_config["categorial_col"]

    elif target_columns_type == "num":
        # Uniquement variables numériques
        OHE = [*prepro_config["OHE_2"]]
        columns = list(
            {*prepro_config["numeric_with_outliers_columns"],
             *prepro_config["numeric_uniform_colums"]} - {
                target_variable})
    else:
        # Toutes les variables (cat + num)
        OHE = [*prepro_config["OHE"], *prepro_config["OHE_2"]]
        columns = list(
            {*prepro_config["categorial_col"], *prepro_config["numeric_with_outliers_columns"], *prepro_config["numeric_uniform_colums"]} - {
                target_variable})

    # ========================================================================
    # EXÉCUTION DU TOUR DEMANDÉ
    # ========================================================================

    # ------------------------------------------------------------------------
    # TOUR 1: Construction MLNA-1 (monocouche)
    # ------------------------------------------------------------------------
    if args.turn == 1:
        if sum(['graph_turn_1_completed.dtvni' == file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_1') for
                file in files]) > 0 and args.graph_with_class is False:
            print("✅ MLNA 1 Graph already completed")
        else:
            # Exécution MLNA-1
            make_mlna_1_variable_v2(
                x_traini=x_traini,
                x_testi=x_testi,
                y_traini=y_traini,
                y_testi=y_testi,
                OHE=OHE,
                nominal_factor_colums=columns,
                cwd = args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}',
                root= args.cwd,
                domain= domain,
                target_variable= target_variable,
                alpha= args.alpha,
                graphWithClass=args.graph_with_class
            )

            # Création du fichier flag de complétion
            with open(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_1/graph_turn_1_completed.dtvni', "a") as fichier:
                fichier.write("")

    # ------------------------------------------------------------------------
    # TOUR 2: Construction MLNA-TOP-K (variables sélectionnées)
    # ------------------------------------------------------------------------
    if args.turn == 2:
        # Vérification de l'existence des résultats MNIFS
        if sum([f"MNIFS_{domain}_best_features" in file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}') for
                file in files]) == 0:
            print("❌ Unable to access selection protocol results")
            exit(1)

        # Vérification si déjà complété
        if sum(['graph_turn_2_completed.dtvni' == file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/select{"/"+args.metric if args.metric.strip() else ""}/') for
                file in files]) > 0:
            print("✅ MLNA 1 Graph already completed")
        else:

            # Chargement des résultats de sélection MNIFS
            mnifs_path = args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/' + [file for _, _, files in
                os.walk(
                  args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}')
                for
                file in files][
                [f"MNIFS_{domain}_best_features" in file for _, _, files in
                 os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}') for
                 file in files].index(True)
            ]
            mnifs_config = read_model(path=mnifs_path)

            # Exécution MLNA-TOP-K
            make_mlna_top_k_variable_v2(
                x_traini=x_traini,
                x_testi=x_testi,
                y_traini=y_traini,
                y_testi=y_testi,
                OHE=OHE,
                nominal_factor_colums=columns,
                cwd=args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/select{"/"+args.metric if args.metric.strip() else ""}/',
                root=args.cwd,
                domain=domain,
                target_variable=target_variable,
                alpha=args.alpha,
                graphWithClass=args.graph_with_class,
                topR=list(mnifs_config['model'][args.metric].keys() if args.metric.strip() else mnifs_config['model'].keys())
            )

            # Création du fichier flag de complétion
            with open(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/select{"/"+args.metric if args.metric.strip() else ""}/graph_turn_2_completed.dtvni', "a") as fichier:
                fichier.write("")

    # ------------------------------------------------------------------------
    # TOUR 3: Construction MLNA-2 (combinatoire)
    # ------------------------------------------------------------------------
    if args.turn == 3:
        # Vérification si déjà complété
        if sum(['graph_turn_3_completed.dtvni' == file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_2/') for
                file in files]) > 0:
            print("✅ COMBINATORY MLNA 2 Graph  already completed")
        else:
            # Exécution MLNA-K (k=2)
            make_mlna_k_variable_v2(
                x_traini=x_traini,
                x_testi=x_testi,
                y_traini=y_traini,
                y_testi=y_testi,
                OHE=OHE,
                nominal_factor_colums=columns,
                cwd = args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}',
                root= args.cwd,
                domain= domain,
                target_variable= target_variable,
                alpha= args.alpha,
                graphWithClass=args.graph_with_class
            )

            # Création du fichier flag de complétion
            with open(
                    args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_2/graph_turn_3_completed.dtvni',
                    "a") as fichier:
                fichier.write("")

    print("Descripteurs extraits et sauvegardés.")

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    main()

