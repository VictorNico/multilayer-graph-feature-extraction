"""
===============================================================================
FILE: 03_graph_construction.py
===============================================================================
Author: DJIEMBOU TIENTCHEU Victor Nico
Created: 28/11/2023
Last modified: 11/06/2025

DESCRIPTION:
    Builds and analyses multilayer graphs for descriptor extraction within
    the MLNA supervised machine-learning pipeline.  Three construction modes
    are implemented:
    - MLNA-1:     Monolayer analysis — one categorical variable per layer.
    - MLNA-K:     Combinatorial multilayer analysis — k=2 variables per layer.
    - MLNA-TOP-K: Top-k selected variables per layer (framework mode).

DEPENDENCIES:
    - networkx:              Graph construction and manipulation.
    - pandas / numpy:        Data manipulation and numerical computation.
    - modules.preprocessing: Combination generation utilities.
    - modules.file:          File I/O helpers.
    - modules.graph:         Multilayer graph building and PageRank extraction.

STRUCTURE:
    1. Descriptor extraction from graphs (PageRank — batched power iteration)
    2. Descriptor configuration generation and saving
    3. MLNA-1 construction (monolayer)
    4. MLNA-K construction (combinatorial multilayer)
    5. MLNA-TOP-K construction (selected variables)
    6. main() entry point

USAGE:
    python -m scripts.03_graph_construction --cwd <dir> --dataset_folder <name>
           --alpha <value> --turn <number> [--graph_with_class] [--metric <name>]
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
    """Extract graph-based descriptors for a single instance.

    Computes two families of descriptors:

    1. **GLO** (global) — PageRank computed over the entire graph with a
       uniform or type-restricted personalisation vector.
    2. **PER** (personalised) — PageRank personalised towards the neighbours
       of *borrower* in the graph.

    Descriptor types:
        - ``DEGREE``: Number of instances sharing the same modality values.
        - ``INTRA``:  PageRank over intra-layer modality nodes (``-M-`` nodes).
        - ``INTER``:  PageRank over inter-layer instance nodes (``-U-`` nodes).
        - ``COMBINE``: PageRank over the full graph.
        - ``M_*``:    Maximum PageRank score of modality nodes connected to
                      *borrower*.
        - ``YN_* / YP_*``: Class-node scores (only when *graphWithClass* is True).

    PageRank is computed via a single batched power-iteration over a
    6-column personalisation matrix, replacing 6 sequential
    ``nx.pagerank()`` calls.

    Args:
        graph (nx.DiGraph): Multilayer graph encoding instance-modality relations.
        graphWithClass (bool): If True, include class-node descriptors (CX suffix
            and YN/YP keys).
        alpha (float): PageRank damping factor.  Default 0.85.
        borrower (int): Row index of the instance to extract descriptors for.
        layers (int): Number of graph layers (used for DEGREE computation).
            Default 1.

    Returns:
        dict: Descriptor dictionary.  Key format: ``'Att_<TYPE>_<CONTEXT>_<MODE>_'``
            where TYPE ∈ {DEGREE, INTRA, INTER, COMBINE, M_INTRA, M_INTER,
            M_COMBINE}, CONTEXT ∈ {GLO, PER}, MODE ∈ {MX, CX}.
            Example::

                {
                    'Att_DEGREE_GLO': 10,
                    'Att_INTRA_GLO_MX_': 0.0234,
                    'Att_M_COMBINE_PER_CX_': 0.0456,
                    'YN_COMBINE_PER': 0.678,
                    'YP_COMBINE_PER': 0.322,
                }
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
    """Organise extracted descriptors into DataFrames and save them to disk.

    Assembles train and test descriptor DataFrames for each descriptor family
    (GLO_MX, GLO_CX, PER_MX, PER_CX, PER_CY, PER_CXY) and writes them as CSV
    files under the appropriate sub-directory.  Also saves a ``.conf`` pickle
    containing the complete configuration for later reconstruction.

    Output directory layout::

        cwd/mlna_X/<name>/
            ├── global/withClass/descriptors/        (GLO_CX)
            ├── global/withoutClass/descriptors/     (GLO_MX)
            ├── personalized/withClass/descriptors/  (PER_CX, PER_CY, PER_CXY)
            ├── personalized/withoutClass/descriptors/ (PER_MX)
            └── config_df_for_<name>_<with|without>Class.conf

    Args:
        graphWithClass (bool): If True, also produce CY and CXY descriptor
            DataFrames.
        mlnL (str): MLNA level path segment (e.g. ``'/mlna_1'``,
            ``'/mlna_k_b'``).
        cwd (str): Working directory path (results root for the current alpha).
        root (str): Project root directory.
        domain (str): Dataset domain name used in filenames.
        extracts_g (dict): Global descriptors for the training set.
            Format: ``{'Att_DEGREE_GLO': [v1, v2, ...], ...}``.
        extracts_p (dict): Personalised descriptors for the training set.
        extracts_g_t (dict): Global descriptors for the test set.
        extracts_p_t (dict): Personalised descriptors for the test set.
        name (str): Variable/configuration name used in directory and file names.
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
    """Build MLNA-1 graphs and extract monolayer descriptors for each categorical variable.

    For each variable in *nominal_factor_colums*, constructs a one-layer
    multilayer graph, then iterates over train instances (leave-one-out: the
    edge to the class node is temporarily removed) and test instances (the
    instance is temporarily added) to extract GLO and PER PageRank descriptors.
    Descriptors are normalised with :func:`standard_extraction` and saved via
    :func:`generate_config_df`.  Already-processed variables are skipped.

    Algorithm per variable:
        1. Build the multilayer graph (with or without class nodes).
        2. Save the graph in GraphML format.
        3. For each training instance: remove class edge → extract → restore edge.
        4. For each test instance: add instance → extract → remove instance.
        5. Remove unused descriptor keys (MX or CX depending on *graphWithClass*).
        6. Normalise with ``standard_extraction``; apply same scale to test.
        7. Call ``generate_config_df`` to save DataFrames and ``.conf`` file.

    Args:
        x_traini (pd.DataFrame): Training feature matrix.
        x_testi (pd.DataFrame): Test feature matrix.
        y_traini (pd.Series): Training labels.
        y_testi (pd.Series): Test labels.
        OHE (list[np.ndarray]): One-hot-encoded column arrays, one per variable.
        nominal_factor_colums (list[str]): Original categorical column names
            (one-to-one with *OHE*).
        cwd (str): Results directory for the current (dataset, alpha, type).
        root (str): Project root directory.
        domain (str): Dataset name used in saved filenames.
        target_variable (str): Name of the target column.
        alpha (float): PageRank damping factor.
        graphWithClass (bool): If True, include class nodes in the graph.
            Default False.
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
    """Build MLNA-K graphs and extract combinatorial multilayer descriptors (k=2).

    Enumerates all C(n, 2) pairs of categorical variables, builds a two-layer
    multilayer graph for each pair, and extracts descriptors following the same
    leave-one-out / temporary-addition protocol as :func:`make_mlna_1_variable_v2`.
    Results are saved under ``cwd/mlna_2/<var1>_<var2>/``.

    Algorithm:
        For each pair of variables (k=2 fixed):
        1. Build a 2-layer MLN graph.
        2. Extract and normalise GLO/PER descriptors for train and test.
        3. Save via :func:`generate_config_df`.

    Args:
        x_traini (pd.DataFrame): Training feature matrix.
        x_testi (pd.DataFrame): Test feature matrix.
        y_traini (pd.Series): Training labels.
        y_testi (pd.Series): Test labels.
        OHE (list[np.ndarray]): One-hot-encoded column arrays.
        nominal_factor_colums (list[str]): Original categorical column names.
        cwd (str): Results directory for the current (dataset, alpha, type).
        root (str): Project root directory.
        domain (str): Dataset name.
        alpha (float): PageRank damping factor.
        target_variable (str): Name of the target column.
        graphWithClass (bool): If True, include class nodes.  Default True.

    Note:
        Complexity is O(n²) in the number of categorical variables.  The inner
        loop is currently fixed to k=2 but the structure supports extension to
        higher k values.
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
    """Build MLNA-TOP-K graphs for the top-k variables selected by the MNIFS protocol.

    Incrementally builds k-layer multilayer graphs for k = 2 … len(topR),
    always using the first k indices from *topR* (ranked by importance).
    Unlike :func:`make_mlna_k_variable_v2`, no combinatorial search is
    performed: the variable order is fixed by *topR*.  Results are saved under
    ``cwd/mlna_k_b/<top_k_vars>/``.

    Algorithm:
        For k from 2 to len(topR):
        1. Select the top-k variable indices: ``layer_config = topR[:k]``.
        2. Build a k-layer MLN graph.
        3. Extract and normalise GLO/PER descriptors for train and test.
        4. Save via :func:`generate_config_df` with mlnL=``'/mlna_k_b'``.

    Example:
        If topR = [2, 5, 1, 4]:
        - k=2: layers=[2,5],   saved to ``mlna_2_b/var2_var5/``
        - k=3: layers=[2,5,1], saved to ``mlna_3_b/var2_var5_var1/``
        - k=4: layers=[2,5,1,4], saved to ``mlna_4_b/var2_var5_var1_var4/``

    Args:
        x_traini (pd.DataFrame): Training feature matrix.
        x_testi (pd.DataFrame): Test feature matrix.
        y_traini (pd.Series): Training labels.
        y_testi (pd.Series): Test labels.
        OHE (list[np.ndarray]): One-hot-encoded column arrays.
        nominal_factor_colums (list[str]): Original categorical column names.
        cwd (str): Results directory (typically ``…/select[/metric]/``).
        root (str): Project root directory.
        domain (str): Dataset name.
        target_variable (str): Name of the target column.
        alpha (float): PageRank damping factor.
        graphWithClass (bool): If True, include class nodes.  Default False.
        topR (list[int]): Variable indices ordered by decreasing importance,
            as produced by the MNIFS selection protocol.  Default [].
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
    """Entry point for the MLNA graph-construction pipeline (script 03).

    Dispatches to one of three construction modes depending on ``--turn``:

    Turn 1 — MLNA-1 (monolayer):
        Builds one graph per categorical variable.
        Completion flag: ``mlna_1/graph_turn_1_completed.dtvni``.

    Turn 2 — MLNA-TOP-K (framework mode):
        Loads MNIFS selection results and incrementally builds graphs for the
        top-k variables.  Requires ``--graph_with_class`` in practice.
        Completion flag: ``select[/<metric>]/graph_turn_2_completed.dtvni``.

    Turn 3 — MLNA-K (combinatorial, k=2):
        Builds one graph for every pair of categorical variables.
        Completion flag: ``mlna_2/graph_turn_3_completed.dtvni``.

    CLI arguments:
        --cwd (str): Absolute path to the working directory.
        --dataset_folder (str): Dataset sub-folder name.
        --alpha (float): PageRank damping factor.
        --turn (int): Construction mode to execute (1, 2, or 3).
        --graph_with_class (flag): If present, include class nodes in graphs.
        --metric (str): Optional metric sub-directory used in turn 2 path.

    Required inputs:
        - ``configs/<dataset_folder>/config.ini``
        - ``<processed_dir>/<dataset_folder>/preprocessing_*.pkl``
        - ``<split_dir>/<dataset_folder>/<type>/<domain>_train/test.csv``
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

