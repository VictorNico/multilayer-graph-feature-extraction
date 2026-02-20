"""
    This file contains all methods useful to make a statistical analysis of the environment reports

    based on the exploration ideas, we'll need to analyse the following aspects:
        - compare the gain of models using descriptor extract with the personalisation per loan and the global one
        - compare the gain of models with taking account of class or decision attributes multilayer network analysis
        - compare the gain of models using both global personalisation and personalisation per loan descriptors
        - compare the gain of applying features selection in place of the random one
        - compare the models using the customs cost metric
        - evaluate the gain of class distance base descriptor in learning process

"""
import re
from copy import deepcopy

# coding: utf-8

## 0.------- Module loading
from .file import *
from .report import *
import json
from colorama import init, Fore, Style
import statistics
import math
from scipy.spatial.distance import euclidean


# Initialize colorama for Windows compatibility
init()

# 1.--------Constant declaration

MlC_F = lambda x: (('MlC_' in x))  # find metric of model where mln were added to classic Att
MCA_F = lambda x: (('MCA_' in x))  # where mln attribut were removed first and mln were added to the rest
GAP_F = lambda x: (('GAP_' in x))  # where mln attribut were removed first and mln were added to the rest
C_F = lambda x: (('classic_' in x))  # classic metrics
GLO_F = lambda x: (('GLO' in x))  # classic metrics
PER_F = lambda x: (('PER' in x))  # classic metrics
Y_F = lambda x: (('YP' in x) or ('YN' in x))  # classic metrics

GLO_MX_F = lambda x: (('GLO_MX_' in x))
GLO_CX_F = lambda x: (('GLO_CX_' in x))
PER_MX_F = lambda x: (('PER_MX_' in x))
PER_CX_F = lambda x: (('PER_CX_' in x))
PER_CY_F = lambda x: (('PER_CY_' in x))
PER_CXY_F = lambda x: (('PER_CXY_' in x))
GAP_MX_F = lambda x: (('GAP_MX_' in x))
GAP_CX_F = lambda x: (('GAP_CX_' in x))
GAP_CY_F = lambda x: (('GAP_CY_' in x))
GAP_CXY_F = lambda x: (('GAP_CXY_' in x))
BOT_CXY_F = lambda x: (('BOT_CXY_' in x))

GLO_INTER_F = lambda x: (not ('_M_' in x) and ('INTER' in x) and ('GLO' in x))
GLO_INTRA_F = lambda x: (not ('_M_' in x) and ('INTRA' in x) and ('GLO' in x))
GLO_COMBINE_F = lambda x: (not ('_M_' in x) and ('COMBINE' in x) and ('GLO' in x))
PER_INTER_F = lambda x: (not ('_M_' in x) and ('INTER' in x) and ('PER' in x))
PER_INTRA_F = lambda x: (not ('_M_' in x) and ('INTRA' in x) and ('PER' in x))
PER_COMBINE_F = lambda x: (not ('_M_' in x) and ('COMBINE' in x) and ('PER' in x))
GLO_INTER_M_F = lambda x: (('_M_' in x) and ('INTER' in x) and ('GLO' in x))
GLO_INTRA_M_F = lambda x: (('_M_' in x) and ('INTRA' in x) and ('GLO' in x))
GLO_COMBINE_M_F = lambda x: (('_M_' in x) and ('COMBINE' in x) and ('GLO' in x))
PER_INTER_M_F = lambda x: (('_M_' in x) and ('INTER' in x) and ('PER' in x))
PER_INTRA_M_F = lambda x: (('_M_' in x) and ('INTRA' in x) and ('PER' in x))
PER_COMBINE_M_F = lambda x: (('_M_' in x) and ('COMBINE' in x) and ('PER' in x))
DEGREE_F = lambda x: (('DEGREE' in x))

YN_PER_INTER_F = lambda x: (not ('_M_' in x) and ('INTER' in x) and ('PER' in x) and ('YN' in x))
YN_PER_INTRA_F = lambda x: (not ('_M_' in x) and ('INTRA' in x) and ('PER' in x) and ('YN' in x))
YN_PER_COMBINE_F = lambda x: (not ('_M_' in x) and ('COMBINE' in x) and ('PER' in x) and ('YN' in x))
YP_PER_INTER_F = lambda x: (not ('_M_' in x) and ('INTER' in x) and ('PER' in x) and ('YP' in x))
YP_PER_INTRA_F = lambda x: (not ('_M_' in x) and ('INTRA' in x) and ('PER' in x) and ('YP' in x))
YP_PER_COMBINE_F = lambda x: (not ('_M_' in x) and ('COMBINE' in x) and ('PER' in x) and ('YP' in x))

svg = "<?xml version='1.0' ?><!DOCTYPE svg  PUBLIC '-//W3C//DTD SVG 1.1//EN'  'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd'><svg enable-background='new 0 0 32 32' height='12px' id='Layer_1' version='1.1' viewBox='0 0 32 32' width='12px' xml:space='preserve' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'><path d='M18.221,7.206l9.585,9.585c0.879,0.879,0.879,2.317,0,3.195l-0.8,0.801c-0.877,0.878-2.316,0.878-3.194,0  l-7.315-7.315l-7.315,7.315c-0.878,0.878-2.317,0.878-3.194,0l-0.8-0.801c-0.879-0.878-0.879-2.316,0-3.195l9.587-9.585  c0.471-0.472,1.103-0.682,1.723-0.647C17.115,6.524,17.748,6.734,18.221,7.206z' fill='#515151'/></svg>"

header = """
    \\documentclass[11pt, oneside]{article}      % use "amsart" instead of "article" for AMSLaTeX format
    \\usepackage{geometry}                       % See geometry.pdf to learn the layout options. There are lots.
    \\geometry{letterpaper}                          % ... or a4paper or a5paper or ... 
    %\\geometry{landscape}                       % Activate for rotated page geometry
    %\\usepackage[parfill]{parskip}          % Activate to begin paragraphs with an empty line rather than an indent
    \\usepackage{graphicx}               % Use pdf, png, jpg, or eps§ with pdflatex; use eps in DVI mode
                                    % TeX will automatically convert eps --> pdf in pdflatex        
    \\usepackage{amssymb}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Modules
    \\usepackage{pgfplots}
    \\usepackage{xcolor}
    \\usepackage{tikz}
    \\pgfplotsset{compat=1.17}
    \\usetikzlibrary{matrix, positioning}

    \\usepackage{array}
    \\usepackage{float}
    \\usepackage{pgfplots}
    \\usepackage[utf8]{inputenc}
    \\usepackage[T1]{fontenc}
    \\usepackage{fancyhdr}
    \\pagestyle{fancy}
    \\usepackage{multirow}
    \\usepackage{hyperref}
    \\usepackage[babel]{csquotes}
    \\usepackage{rotating}
    \\usepackage{
    multirow, 
    multicol, 
    xcolor, 
    % fontspec, 
    graphicx, 
    booktabs, 
    subfigure, 
    fancyhdr, 
    tikz, 
    algorithm, 
    algpseudocode, 
    acronym
    }
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %SetFonts

    %SetFonts


    \\title{Class-based multilayer graph feature extraction for supervised Machine Learning}
    \\author{DJIEMBOU TIENTCHEU Victor Nico}
    %\\date{}                            % Activate to display a given date or no date

    \\begin{document}
    \\maketitle
    \\newpage
    """

bchar_header="""
    \\documentclass[varwidth=true, border=2pt]{standalone}
    \\usepackage[margin=1in]{geometry}
    \\usepackage{multicol}
    \\usepackage{bchart}
    \\usepackage{array}

    \\usetikzlibrary{fit}

    \\makeatletter
    \\newdimen\\legendxshift
    \\newdimen\\legendyshift
    \\newcount\\legendlines
    % distance of frame to legend lines
    \\newcommand{\\bclldist}{1mm}
    \\newcommand{\\bclegend}[3][10mm]{%
        % initialize
        \\legendxshift=0pt\\relax
        \\legendyshift=0pt\\relax
        \\xdef\\legendnodes{}%
        % get width of longest text and number of lines
        \\foreach \\lcolor/\\ltext [count=\\ll from 1] in {#3}%
            {\\global\\legendlines\\ll\\pgftext{\\setbox0\\hbox{\bcfontstyle\\ltext}\\ifdim\\wd0>\\legendxshift\\global\\legendxshift\\wd0\\fi}}%
        % calculate xshift for legend; \\bcwidth: from bchart package; \\bclldist: from node frame, inner sep=\\bclldist (see below)
        % \\@tempdima: half width of bar; 0.72em: inner sep from text nodes with some manual adjustment
        \\@tempdima#1\\@tempdima0.5\\@tempdima
        \\pgftext{\\bcfontstyle\\global\\legendxshift\\dimexpr\\bcwidth-\\legendxshift-\\bclldist-\\@tempdima-0.72em}
        % calculate yshift; 5mm: heigt of bar
        \\legendyshift\\dimexpr5mm+#2\\relax
        \\legendyshift\\legendlines\\legendyshift
        % \\bcpos-2.5mm: from bchart package; \\bclldist: from node frame, inner sep=\\bclldist (see below)
        \\global\\legendyshift\\dimexpr\\bcpos-2.5mm+\\bclldist+\\legendyshift
        % draw the legend
        \\begin{scope}[shift={(\\legendxshift,\\legendyshift)}]
        \\coordinate (lp) at (0,0);
        \\foreach \\lcolor/\\ltext [count=\\ll from 1] in {#3}%
        {
            \\node[anchor=north, minimum width=#1, minimum height=5mm,fill=\\lcolor] (lb\\ll) at (lp) {};
            \\node[anchor=west] (l\\ll) at (lb\\ll.east) {\\bcfontstyle\\ltext};
            \\coordinate (lp) at ($(lp)-(0,5mm+#2)$);
            \\xdef\\legendnodes{\\legendnodes (lb\\ll)(l\\ll)}
        }
        % draw the frame
        \\node[draw, inner sep=\\bclldist,fit=\\legendnodes] (frame) {};
        \\end{scope}
    }
    \\makeatother

    \\begin{document}
    \\resizebox{\\textwidth}{!}{
    """


plot_header = """\\documentclass[varwidth=true, border=2pt]{standalone}
    \\usepackage[margin=1in]{geometry}
    \\usepackage{multicol}
    \\usepackage{array}
    \\usepackage{pgfplots}

    \\pgfplotsset{compat=1.18}

    \\begin{document}
    \\resizebox{\\textwidth}{!}{
    """


footer = """
    \\end{document}  """

# add cols for metric
dicto = {'accuracy':'Acc', 'f1-score':'F1', 'precision':'P', 'recall':'R', 'financial-cost':'Cost'}

############# End declaration
########################################################
# --------------------------------------------------------------------------------------MixingUtilsFunction---------------------------------------------------------


def pretty_print(data):
    def format_value(item):
        if isinstance(item, list):
            return [f"{Fore.CYAN}{val}{Style.RESET_ALL}" for val in item]
        return f"{Fore.CYAN}{item}{Style.RESET_ALL}"

    formatted = json.dumps(data, indent=2, default=format_value)
    formatted = formatted.replace('"', '')

    for i, line in enumerate(formatted.split('\n')):
        if ':' in line:
            key, value = line.split(':', 1)
            print(f"{Fore.GREEN}{key}{Style.RESET_ALL}:{value}")
        else:
            if i == 0:
                print(f"{Fore.YELLOW}{line}{Style.RESET_ALL}")
            else:
                print(line)

def get_filenames(root_dir, func, verbose=False):
    """ Get all filenames
        Args:
            root_dir (string): Directory where all files are stored
            func (function): Function that takes one argument
            verbose (boolean): If true, prints out the filenames
        Returns
            a list of all files in `root_dir`
    """
    data_filenames = []
    # Walk through the directories and files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # dirpath: current directory path
        # dirnames: list of directories in the current directory
        # filenames: list of files in the current directory

        # Print the current directory
        print('Directory:', dirpath) if verbose else None
        # Print all the subdirectories
        if verbose:
            for dirname in dirnames:
                print('Subdirectory:', os.path.join(dirpath, dirname))

        # Print all the files
        for filename in filenames:
            if func(filename) and not ('_x_' in filename or '_y_' in filename) and ('_metric_' in filename):
                print('File:', os.path.join(dirpath, filename)) if verbose else None
                data_filenames.append(os.path.join(dirpath, filename))

        # Print an empty line to separate directories
        print() if verbose else None
    return data_filenames


"""
    Lambda Function
    Get all qualitative features from operation results
    During the process, we specify the partern '__' like an indicator
"""
get_qualitative_from_cols = lambda x: (list(set([
    var.split("__",1) for var in  x if ("__" in var)
    
]
)))

"""
    Lambda Function
    Get all quantitative features from operation results
    During the process, we specify the partern '___' like an indicator
"""
get_quantitative_from_cols = lambda x: (list(set([
    var.split("___",1) for var in x if ("___" in var)
    ]
)))

# Valeur de epsilon
epsilon = 1e-8
# --------------------------------------------------------------------------------------RessourcesLoading---------------------------------------------------------

# ################################################################
# ############## Files Ressources Loading ########################
# ################################################################
def load_results(
        outputs_path,
        _type,
        k,
        alpha,
        per=True,
        glo=True,
        mix=True,
        bot=False,
        isRand=False,
        match=lambda x: True,
        attributs=None,
        isBest=False,
        dataset_delimiter=None,
        encoding=None,
        index_col=None,
        metric=''
):
    """
    Load results from a results folder.

    OPTIMISATION : au lieu de 22×N_attributs appels os.walk() (un par (approach,logic,config)),
    on fait 1 seul os.walk() du répertoire racine puis on filtre en mémoire.
    Gain mesuré : O(22×N) → O(1) en I/O disque, critique pour grande échelle (DMKD).
    """

    if attributs is None:
        attributs = []
    # ── Calcul des chemins racine ────────────────────────────────────────────
    select_part = (
        f"/select/{metric.strip()}" if isBest and metric.strip()
        else ("/select" if isBest else "")
    )
    layer_part = (
        f"mlna_{k}" if k == 1 or isRand
        else f"mlna_{k}_b"
    )
    root = f"{outputs_path}/{alpha}/{_type}{select_part}/{layer_part}"

    # Racine distincte pour BOT (toujours sous /select/mlna_k_b ou /mlna_1)
    bot_root = f'{outputs_path}/{alpha}/{_type}{("/select"+(metric+"/" if metric.strip() else ""))*(k>1)}/mlna_{k}{"_b"*(k>1)}'

    # ── Index disque : 1 seul os.walk() pour tous les attributs ─────────────
    # Filtre immédiat : on ne garde que les fichiers *_metric_* (pas _x_ ni _y_)
    _index = {}
    for dirpath, _, filenames in os.walk(root):
        hits = [
            f for f in filenames
            if '_metric_' in f and '_x_' not in f and '_y_' not in f
        ]
        if hits:
            _index[dirpath] = hits

    _bot_index = {}
    if bot:
        for dirpath, _, filenames in os.walk(bot_root):
            hits = [
                f for f in filenames
                if '_metric_' in f and '_x_' not in f and '_y_' not in f
            ]
            if hits:
                _bot_index[dirpath] = hits

    # ── Helper : lookup dans l'index + chargement CSV ───────────────────────
    def collect(approach_fn, logic_fn, subpath, use_bot=False):
        idx = _bot_index if use_bot else _index
        base = bot_root if use_bot else root
        result = []
        for attribut in attributs:
            dir_path = os.path.join(base, attribut, subpath, 'evaluation')
            for fname in idx.get(dir_path, []):
                if approach_fn(fname) and logic_fn(fname) and match(fname):
                    result.append(
                        load_data_set_from_url(
                            path=os.path.join(dir_path, fname),
                            sep=dataset_delimiter,
                            encoding=encoding,
                            index_col=index_col,
                            na_values=None
                        )
                    )
        return result

    # ── Construction du dictionnaire de résultats (même structure qu'avant) ─
    files = {
        'MlC': {
            'GLO': {
                'MX': collect(MlC_F, GLO_MX_F, 'global/withoutClass'),
                'CX': collect(MlC_F, GLO_CX_F, 'global/withClass'),
            } if glo else None,
            'PER': {
                'MX':  collect(MlC_F, PER_MX_F,  'personalized/withoutClass'),
                'CX':  collect(MlC_F, PER_CX_F,  'personalized/withClass'),
                'CY':  collect(MlC_F, PER_CY_F,  'personalized/withClass'),
                'CXY': collect(MlC_F, PER_CXY_F, 'personalized/withClass'),
            } if per else None,
            'GAP': {
                'MX':  collect(MlC_F, GAP_MX_F,  'mixed/withoutClass'),
                'CX':  collect(MlC_F, GAP_CX_F,  'mixed/withClass'),
                'CY':  collect(MlC_F, GAP_CY_F,  'mixed/withClass'),
                'CXY': collect(MlC_F, GAP_CXY_F, 'mixed/withClass'),
            } if mix else None,
            'BOT': {
                'CXY': collect(MlC_F, BOT_CXY_F, 'mixed/both', use_bot=True),
            } if bot else None,
        },
        'MCA': {
            'GLO': {
                'MX': collect(MCA_F, GLO_MX_F, 'global/withoutClass'),
                'CX': collect(MCA_F, GLO_CX_F, 'global/withClass'),
            } if glo else None,
            'PER': {
                'MX':  collect(MCA_F, PER_MX_F,  'personalized/withoutClass'),
                'CX':  collect(MCA_F, PER_CX_F,  'personalized/withClass'),
                'CY':  collect(MCA_F, PER_CY_F,  'personalized/withClass'),
                'CXY': collect(MCA_F, PER_CXY_F, 'personalized/withClass'),
            } if per else None,
            'GAP': {
                'MX':  collect(MCA_F, GAP_MX_F,  'mixed/withoutClass'),
                'CX':  collect(MCA_F, GAP_CX_F,  'mixed/withClass'),
                'CY':  collect(MCA_F, GAP_CY_F,  'mixed/withClass'),
                'CXY': collect(MCA_F, GAP_CXY_F, 'mixed/withClass'),
            } if mix else None,
            'BOT': {
                'CXY': collect(MCA_F, BOT_CXY_F, 'mixed/both', use_bot=True),
            } if bot else None,
        }
    }
    return files


def build_compare_feature_selection_protocole(
        store
):
    # add the resize box to ensure the scale of the table will be contain's inside the width space avalable.
    # start setting up the tabular dimensions setting
    datasets = list(store.keys())
    methods = list(store[datasets[0]].keys())
    alphas = sorted(list({el for data in datasets for el in store[data][methods[0]].keys()}))
    table_header = """
        %\\begin{sidewaystable}
        \\resizebox{\\textwidth}{!}{

        \\begin{tabular}{|c|c|""" + ("c|" * len(datasets))
    # setup information columns headears
    nbMCol = len(datasets)
    # add col for total results
    table_header += "c|} "
    # add separator clines
    nb_cols = (2 + nbMCol + 1)
    table_header += " \\cline{1-" + str(nb_cols) + "}"  # corresponding to the number of columns

    # build the first line: metrics' line
    lines = ''
    # add the blank block
    lines += """
    \\multirow{2}{*}{\\textbf{$\\alpha$}} & \\multirow{2}{*}{\\textbf{Method}} & \\multicolumn{"""+str(len(datasets))+"""}{c|}{Dataset} & \\multirow{2}{*}{\\textbf{Score}}
    """
    lines += """\\\\
     \\cline{3-""" + str(nb_cols-1) + """}
      & 
    """


    # add alpha for metric
    for fold in datasets:
        lines += f" & {fold}"
    # add the total name
    lines += """ & \\\\ 
    \\hline
    \\hline
    """
    # pretty_print(methods)
    for ai, alpha in enumerate(alphas):
        lines += """
        \\multirow{3}{*}{""" + str(alpha) + """}"""
        for mi, meth in enumerate(methods):
            lines += f""" & {meth}"""
            for folder in datasets:
                # print('---------------',store[folder][meth][alpha])
                lines += (
                    f""" & {store[folder][meth][alpha]}"""
                    if store[folder][meth][alpha] < store[folder]['réel'][alpha] or ( 'réel' in meth)
                    else f""" & \\textbf{{{store[folder][meth][alpha]}}}"""
                ) if alpha in list(store[folder][meth].keys()) else " & "
            lines += (""" & \\\\ """ + """ \\cline{2-""" + str(nb_cols) + """}

                                        """) if mi != len(methods) - 1 else (
                        """ & \\\\ """)
        lines += """ 
        \\hline 
        \\hline
        
        """

    lines += """

    \\end{tabular}
    }
    %\\end{sidewaystable}"""

    table = table_header + lines
    return table



def bestThreshold(numbers):
    """
    find the optimal threshold
    Parameters
    ----------
    data

    Returns
    -------
    limit
    """
    diffs = [numbers[i] - numbers[i + 1] for i in range(len(numbers) - 1)]
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)
    result = len(numbers) - 1
    for i, diff in enumerate(diffs):
        if abs(diff - mean_diff) > std_diff:
            result = i
            break  # Sortir de la boucle dès qu'un écart significatif est trouvé
    return result

def cumulative_difference_threshold(accuracies, threshold_percent=0.8):
    sorted_accuracies = sorted(accuracies, reverse=True)
    diffs = [sorted_accuracies[i] - sorted_accuracies[i+1] for i in range(len(sorted_accuracies)-1)]
    total_diff = sum(diffs)
    cumulative_diff = 0
    for i, diff in enumerate(diffs):
        cumulative_diff += diff
        if cumulative_diff / total_diff >= threshold_percent:
            return i + 1
    return len(accuracies)

def cusum_threshold_v1(accuracies, k=0.5, h=5):
    sorted_accuracies = sorted(accuracies, reverse=True)
    mean_acc = np.mean(sorted_accuracies)

    cusum = 0
    for i, acc in enumerate(sorted_accuracies):
        # Détecter une baisse significative
        cusum = max(0, cusum + (mean_acc - acc - k))
        if cusum > h:
            return i
    return len(accuracies)

def cusum_threshold_v2(accuracies, h_threshold=1.0, drift=0.01):
    """
    Détecte le point de stabilisation (changement de régime) via CUSUM.

    Parameters:
    - accuracies: Liste des scores de précision.
    - h_threshold: Seuil de décision (sensibilité).
    - drift: Marge de changement tolérée (souvent la moyenne des variations attendues).
    """
    n = len(accuracies)
    if n < 2: return 0

    # Calcul des différences
    diffs = np.diff(accuracies)
    # On centre sur une cible (généralement on veut détecter quand le gain tombe vers 0)
    target = np.mean(diffs)

    s_high = 0
    # Liste pour stocker les cumuls
    s_pos = []

    for d in diffs:
        # On accumule les écarts positifs par rapport à la cible + dérive
        s_high = max(0, s_high + (d - target - drift))
        s_pos.append(s_high)

    # Le seuil est l'indice où la somme cumulée s'écarte de manière significative
    for i, val in enumerate(s_pos):
        if val > h_threshold:
            return i + 1  # Point de rupture détecté

    return n - 1  # Pas de rupture franche détectée


def elbow_method(accuracies):
    """
    Détecte le point de coude (elbow) dans une courbe de précisions triées.

    Cette méthode géométrique identifie le point qui s'éloigne le plus de la droite
    reliant le premier et le dernier point de la courbe. Ce point représente le
    compromis optimal entre le nombre de features et la performance.

    Principe mathématique :
    ----------------------
    1. Trace une droite entre le premier point (meilleure précision) et le dernier
    2. Pour chaque point, calcule sa distance perpendiculaire à cette droite
    3. Le coude est le point avec la distance maximale

    La distance perpendiculaire est calculée via projection orthogonale :
    - Projection du vecteur point-origine sur la droite
    - Calcul du vecteur de rejet (composante perpendiculaire)
    - Distance = norme euclidienne du vecteur de rejet

    Args:
        accuracies (list[float]): Liste des précisions (non triées)

    Returns:
        int: Indice du coude dans la liste triée (décroissante)
             Retourne len(accuracies) - 1 si le coude est aux positions 0 ou 1

    Example:
        >>> accuracies = [0.95, 0.94, 0.92, 0.85, 0.80, 0.78]
        >>> elbow_method(accuracies)
        3  # Correspond à la précision 0.85 dans la liste triée

    Notes:
        - Les précisions sont automatiquement triées par ordre décroissant
        - La fonction utilise l'indice dans la liste triée, pas la liste originale
        - Si le coude détecté est à l'indice 0 ou 1, retourne le dernier indice
          (cas où il n'y a pas de coude significatif)
    """
    # Trier les précisions par ordre décroissant
    sorted_accuracies = sorted(accuracies, reverse=True)

    # Créer des coordonnées (index, précision) pour chaque point
    coords = [(i, acc) for i, acc in enumerate(sorted_accuracies)]

    # Vecteur de la droite reliant le premier au dernier point
    line_vec = (coords[-1][0] - coords[0][0],
                coords[-1][1] - coords[0][1])

    # Norme (longueur) du vecteur de la droite
    line_vec_norm = math.sqrt(sum(x * x for x in line_vec))

    # Lambda: vecteur du premier point au point donné
    vec_from_first = lambda coord: (coord[0] - coords[0][0],
                                    coord[1] - coords[0][1])

    # Lambda: projection scalaire d'un vecteur sur la droite
    # Formule: (v · l) / ||l||
    scalar_proj = lambda vec: (vec[0] * line_vec[0] + vec[1] * line_vec[1]) / line_vec_norm

    # Lambda: projection vectorielle d'un vecteur sur la droite
    # Formule: ((v · l) / ||l||²) * l
    vec_proj = lambda vec: ((scalar_proj(vec) / line_vec_norm) * line_vec[0],
                            (scalar_proj(vec) / line_vec_norm) * line_vec[1])

    # Lambda: composante perpendiculaire (rejet) d'un vecteur par rapport à la droite
    # Formule: v - proj_l(v)
    vec_reject = lambda vec: (vec[0] - vec_proj(vec)[0],
                              vec[1] - vec_proj(vec)[1])

    # Calculer la distance perpendiculaire de chaque point à la droite
    # Distance = norme euclidienne du vecteur de rejet
    dists_from_line = [euclidean((0, 0), vec_reject(vec_from_first(coord)))
                       for coord in coords]

    # Trouver l'indice du point avec la distance maximale augmenté de 1 pour l'acces en list
    indice = dists_from_line.index(max(dists_from_line)) + 1

    # Retourner l'indice du coude, ou le deuxième indice si le coude est trop tôt
    # (indice <= 1 indique qu'il n'y a pas de coude significatif)
    return indice # if indice > 1 else 2


def otsu_method(metrics):
    from skimage.filters import threshold_otsu

    # Votre liste de points (vecteur 1D)
    data = np.array(metrics)

    # Calcul du seuil optimal
    seuil = threshold_otsu(data)

    # Trouver l'index de la première valeur >= seuil dans les données triées
    sorted_metrics = np.sort(data)[::-1]  # Tri décroissant
    index_seuil = np.searchsorted(-sorted_metrics, -seuil)  # Truc pour tri décroissant

    # S'assurer que l'index est >= 1 (au moins 1 élément dans "Utiles")
    index_seuil = max(1, index_seuil.item())
    return index_seuil


def jenkspy_method(metrics):
    import jenkspy

    if len(metrics) < 3:  # Besoin d'au moins 3 valeurs pour 2 classes
        # Retourner un index par défaut
        return 1  # ou len(metrics) // 2

    # Vérifier s'il y a de la variance
    if len(set(metrics)) == 1:
        # Toutes les valeurs sont identiques
        return len(metrics) // 2

    scores = sorted(metrics, reverse=True)

    try:
        breaks = jenkspy.jenks_breaks(scores, n_classes=2)
        seuil_critique = breaks[1]

        # Trouver l'index
        index_seuil = sum(1 for score in scores if score >= seuil_critique)
        index_seuil = max(1, index_seuil)

        return index_seuil

    except ValueError as e:
        print(f"Erreur Jenkspy: {e}")
        print(f"Nombre de métriques: {len(metrics)}")
        print(f"Valeurs uniques: {len(set(metrics))}")
        # Fallback : retourner le milieu
        return len(metrics) // 2

def elbow_method_v2(points):
    """
    Détecte le point de coude (elbow) dans une courbe de valeurs.

    Cette implémentation vectorisée utilise NumPy pour calculer efficacement
    les distances perpendiculaires de tous les points à la droite reliant
    le premier et le dernier point. Le coude est le point le plus éloigné
    de cette droite.

    Principe mathématique :
    ----------------------
    Pour chaque point P_i, on calcule :
    1. Le vecteur v_i du premier point à P_i
    2. La projection de v_i sur la droite (premier-dernier point)
    3. La distance perpendiculaire = ||v_i - proj(v_i)||

    Formule de projection :
        proj_l(v) = (v · l_norm) * l_norm
    où l_norm est le vecteur directeur normalisé de la droite.

    Args:
        points (list ou array): Liste de valeurs numériques représentant
                                une courbe (ex: précisions, inerties, scores)
                                Les points n'ont pas besoin d'être triés.

    Returns:
        int: Indice du point de coude dans la liste
             - Si moins de 3 points : retourne len(points) - 1
             - Si le coude est à l'indice 0 ou 1 : retourne len(points) - 1
               (indique l'absence de coude significatif)

    Complexité:
        Temps: O(n) où n = len(points)
        Espace: O(n) pour les tableaux temporaires

    Example:
        >>> accuracies = [0.95, 0.92, 0.88, 0.75, 0.72, 0.70]
        >>> elbow_method_v2(accuracies)
        3  # Le coude est à l'indice 3 (valeur 0.75)

        >>> # Visualisation typique d'un coude
        >>> # Précision
        >>> #   |
        >>> # 1.0|●
        >>> # 0.9| ●
        >>> # 0.8|  ●___
        >>> # 0.7|     ●●●  <- Après le coude, la courbe s'aplatit
        >>> #   +----------> Nombre de features

    Notes:
        - Version optimisée avec opérations vectorisées NumPy
        - Plus performante que les boucles Python pour n > 100
        - Les coordonnées x sont les indices (0, 1, 2, ..., n-1)
        - Les coordonnées y sont les valeurs de la liste points

    Voir aussi:
        - elbow_method() : Version avec lambdas et euclidean()
        - KneeLocator (kneed package) : Implémentation plus sophistiquée
    """
    n_points = len(points)

    # Cas limite : moins de 3 points, pas de coude possible
    if n_points < 3:
        return n_points - 1

    # Créer les coordonnées (x, y) de tous les points
    # x : indices [0, 1, 2, ..., n-1]
    # y : valeurs de la liste points
    x = np.arange(n_points)
    y = np.array(points)

    # Vecteur directeur de la droite reliant le premier au dernier point
    # vecteur_ligne = (Δx, Δy) = (n-1, y[-1] - y[0])
    vecteur_ligne = np.array([x[-1] - x[0], y[-1] - y[0]])

    # Normaliser le vecteur pour obtenir un vecteur unitaire
    # l_norm = l / ||l||
    # Nécessaire pour les calculs de projection
    vecteur_ligne_norm = vecteur_ligne / np.sqrt(np.sum(vecteur_ligne ** 2))

    # Créer les vecteurs du premier point (x[0], y[0]) à tous les autres points
    # vecteurs_points[i] = (x[i] - x[0], y[i] - y[0])
    # Shape: (n_points, 2) - chaque ligne est un vecteur 2D
    vecteurs_points = np.vstack([x - x[0], y - y[0]]).T

    # Calcul vectorisé des distances perpendiculaires
    # -------------------------------------------------
    # Pour chaque vecteur v_p :
    # 1. Produit scalaire : v_p · l_norm (résultat: scalaire pour chaque point)
    # 2. Projection : (v_p · l_norm) * l_norm (vecteur dans la direction de la droite)
    # 3. Rejet : v_p - proj(v_p) (composante perpendiculaire)
    # 4. Distance : ||rejet||
    #
    # np.outer() crée une matrice où chaque ligne est la projection du point correspondant
    # Formule finale : distances[i] = ||v_p[i] - proj_l(v_p[i])||
    distances = np.linalg.norm(
        vecteurs_points - np.outer(
            np.dot(vecteurs_points, vecteur_ligne_norm),  # Projections scalaires
            vecteur_ligne_norm  # Vecteur directeur normalisé
        ),
        axis=1  # Calculer la norme de chaque ligne (= chaque vecteur de rejet)
    )

    # Trouver l'indice du point avec la distance maximale
    indice = np.argmax(distances)

    # Retourner l'indice du coude
    # Si indice <= 1, cela indique qu'il n'y a pas de coude clair
    # (le point le plus éloigné est au début), donc on retourne le dernier indice
    return indice if indice > 1 else n_points - 1


def proto_precision_tikz(
        tolerances,
        elbow_results,
        cusum_results,
        datasets,
        layout_config=None
):
    """
    Génère un document TikZ avec des matrices pour comparer les précisions Elbow vs CUSUM

    Args:
        tolerances: liste des tolérances (ex: [0.0, 0.01, 0.02, ...])
        elbow_results: dict {tolerance: {dataset: precision}}
        cusum_results: dict {tolerance: {dataset: precision}}
        datasets: liste des noms de datasets
        output_file: nom du fichier de sortie
        layout_config: dict avec configuration du layout (optionnel)
    """

    # Configuration par défaut du layout
    default_config = {
        'matrices_per_row': len(datasets),
        'matrix_spacing_x': 3.7,
        'matrix_spacing_y': 5,
        'cell_width': '1.7cm',
        'cell_height': '0.6cm',
        'header_width': '3.4cm',
        'header_height': '0.6cm'
    }

    if layout_config:
        default_config.update(layout_config)

    config = default_config

    # Début du document LaTeX
    latex_content = f"""
\\begin{{tikzpicture}}

"""

    first_x = -1.0
    # Calculer positions des matrices
    positions = []
    matrices_per_row = config['matrices_per_row']
    spacing_x = config['matrix_spacing_x']
    spacing_y = config['matrix_spacing_y']

    for i, dataset in enumerate(datasets):
        row = i // matrices_per_row
        col = i % matrices_per_row
        x = col * spacing_x
        y = -row * spacing_y
        positions.append((x, y))

    # Générer chaque matrice pour chaque dataset
    for i, dataset in enumerate(datasets):
        x, y = positions[i]
        matrix_name = dataset.lower().replace(' ', '').replace('-', '')

        # Créer la matrice
        latex_content += f"""% Matrice pour {dataset}
\\matrix ({matrix_name}) [matrix of nodes, nodes in empty cells,
    nodes={{draw, minimum width={config['cell_width']}, minimum height={config['cell_height']}, anchor=center, text centered}}] at ({x if i != 0 else first_x},{y})
{{
"""

        # Ligne d'en-tête (si c'est la première colonne, inclure la colonne des labels)
        if i % matrices_per_row == 0:
            latex_content += "        & Elbow & CUSUM \\\\\n"
        else:
            latex_content += "        Elbow & CUSUM \\\\\n"

        # Lignes de données
        for j, tol in enumerate(tolerances):
            if i % matrices_per_row == 0:  # Première colonne : inclure les labels de précision
                precision_label = f"P ({int(tol * 100)}\\%)"
                row_content = f"        {precision_label} & "
            else:  # Autres colonnes : pas de label
                row_content = "        "

            # Ajouter les valeurs Elbow et CUSUM
            elbow_val = elbow_results[tol][dataset]
            cusum_val = cusum_results[tol][dataset]

            if i % matrices_per_row == 0:
                row_content += f"{elbow_val:.1f} & {cusum_val:.1f}"
            else:
                row_content += f"{elbow_val:.1f} & {cusum_val:.1f}"

            row_content += " \\\\\n"
            latex_content += row_content

        latex_content += "};\n\n"

        # Ajouter le titre du dataset
        header_y = y + (len(tolerances)/2+.4) * 0.6 + 0.57  # Ajuster selon la hauteur des cellules
        latex_content += f"""% Titre pour {dataset}
\\node[draw, rectangle, minimum width={config['header_width']}, minimum height={config['header_height']}] ({matrix_name}_title) at ({(x if i != 0 else first_x)+ 0.85 if x == 0 else x},{header_y}) {{{dataset}}};

"""

    # Fermer le document
    latex_content += """\\end{tikzpicture}
"""

    return latex_content


def proto_precision_tab(
        tolerances,
        datasets,
        store
):
    # add the resize box to ensure the scale of the table will be contain's inside the width space avalable.
    # start setting up the tabular dimensions setting
    methods = list(store.keys())
    table_header = """
        %\\begin{sidewaystable}
        \\resizebox{\\textwidth}{!}{

        \\begin{tabular}{|c|""" + ("c|" * (len(datasets)*len(methods)))
    # setup information columns headears
    nbMCol = len(datasets)*len(methods)
    # add col for total results
    table_header += "} "
    # add separator clines
    nb_cols = (1 + nbMCol)
    table_header += " \\cline{1-" + str(nb_cols) + "}"  # corresponding to the number of columns

    # build the first line: metrics' line
    lines = " & "
    # add the blank block
    lines+= " & ".join(["""\\multicolumn{""" + str(
        len(methods)) + """}{c|}{"""+str(Dat)+"""}""" for Dat in datasets])
    lines += """\\\\
     \\cline{2-""" + str(nb_cols) + """}
    """
    lines += (" & "+" & ".join(methods))*len(datasets)
    lines += """\\\\
         \\cline{1-""" + str(nb_cols) + """}
        """

    # pretty_print(methods)
    for t, tol in enumerate(tolerances):
        lines += f"P ({tol*100}\\%)"
        for folder in datasets:
            for mi, meth in enumerate(methods):

                lines += f""" & {store[meth][tol][folder]:.2f}"""
        lines += ("""\\\\ """ + """ \\cline{2-""" + str(nb_cols) + """}

                                        """) if t != len(tolerances) - 1 else (
                """\\\\ """)
        # lines += """
        # \\hline
        # \\hline
        #
        # """

    lines += """

    \\end{tabular}
    }
    %\\end{sidewaystable}"""

    table = table_header + lines
    return table

def vector_matching_precision(v1, v2, tolerance=0):
    """
    Calcule la précision de correspondance entre deux vecteurs.

    :param v1: Premier vecteur
    :param v2: Deuxième vecteur
    :param tolerance: Tolérance pour considérer deux valeurs comme correspondantes
    :return: Pourcentage de correspondance entre les vecteurs
    """
    if len(v1) != len(v2):
        raise ValueError("Les vecteurs doivent avoir la même longueur")

    v1, v2 = np.array(v1), np.array(v2)
    matches = np.abs(v1 - v2) <= tolerance
    precision = np.mean(matches) * 100

    return precision



def selection_proto(records, output_path, metric='accuracy'):
    # result structure
    resultDict = {
        'dataset': [],
        'alpha': [],
        # 'QuartileThreshold':[],
        'elbowThreshold': [],
        'cumulative_difference_threshold': [],
        'otsu': [],
        'jenkspy':[],
        # 'variance_explained_threshold':[],
        'realThreshold': []
    }
    res = {}
    p = 2
    getTheBestAcc = lambda store, k: round(max([acc for layer, _, acc in store if k == layer]), p)
    real_values = {}
    elbow_values = {}
    cusum_values = {}
    otsu_values = {}
    jenkspy_values = {}
    # walk on the datasets
    for dataset in records.keys():
        # walk on alphas
        res[dataset] = {key: {} for key in ['CUSUM', 'Elbow', 'otsu', 'jenkspy', 'réel']}
        real_values[dataset] = []
        elbow_values[dataset] = []
        cusum_values[dataset] = []
        otsu_values[dataset] = []
        jenkspy_values[dataset] = []
        for alpha in records[dataset].keys():
            if not isinstance(records[dataset][alpha]['predicted_best_k'], list):
                resultDict['dataset'].append(dataset)
                resultDict['alpha'].append(alpha)
                elb = elbow_method(records[dataset][alpha][metric])
                resultDict['elbowThreshold'].append(getTheBestAcc(records[dataset][alpha]['list'], elb))
                res[dataset]['Elbow'][alpha] = getTheBestAcc(records[dataset][alpha]['list'], elb)
                elbow_values[dataset].append(res[dataset]['Elbow'][alpha])

                cum = cusum_threshold_v1(records[dataset][alpha][metric])
                cum = cum if cum < max([layer for layer, _, _ in records[dataset][alpha]['list']]) else max(
                    [layer for layer, _, _ in records[dataset][alpha]['list']])
                resultDict['cumulative_difference_threshold'].append(
                    getTheBestAcc(records[dataset][alpha]['list'], cum))
                res[dataset]['CUSUM'][alpha] = getTheBestAcc(records[dataset][alpha]['list'], cum)
                cusum_values[dataset].append(res[dataset]['CUSUM'][alpha])

                otsu = otsu_method(records[dataset][alpha][metric])
                # print(otsu)
                otsu = otsu if otsu < max([layer for layer, _, _ in records[dataset][alpha]['list']]) else max(
                    [layer for layer, _, _ in records[dataset][alpha]['list']])
                resultDict['otsu'].append(
                    getTheBestAcc(records[dataset][alpha]['list'], otsu))
                res[dataset]['otsu'][alpha] = getTheBestAcc(records[dataset][alpha]['list'], otsu)
                otsu_values[dataset].append(res[dataset]['otsu'][alpha])

                jenkspy = jenkspy_method(records[dataset][alpha][metric])
                jenkspy = jenkspy if jenkspy < max([layer for layer, _, _ in records[dataset][alpha]['list']]) else max(
                    [layer for layer, _, _ in records[dataset][alpha]['list']])
                resultDict['jenkspy'].append(
                    getTheBestAcc(records[dataset][alpha]['list'], jenkspy))
                res[dataset]['jenkspy'][alpha] = getTheBestAcc(records[dataset][alpha]['list'], jenkspy)
                jenkspy_values[dataset].append(res[dataset]['jenkspy'][alpha])

                # resultDict['variance_explained_threshold'].append(getTheBestAcc(records[dataset][alpha]['list'],variance_explained_threshold(list(records[dataset][alpha]['accuracies'].values()))))
                resultDict['realThreshold'].append(
                    round(max([acc for _, _, acc in records[dataset][alpha]['list']]), p))
                res[dataset]['réel'][alpha] = round(max([acc for _, _, acc in records[dataset][alpha]['list']]), p)
                real_values[dataset].append(round(max([acc for _, _, acc in records[dataset][alpha]['list']]), p))
                #
                # print(f"----elbow {dataset}, {alpha}, {res[dataset]['réel'][alpha]}, {elb}")
                # print(round(max([acc for layer, _, acc in records[dataset][alpha]['list'] if elb == layer]), p) == res[dataset]['réel'][alpha])

    dat = pd.DataFrame(resultDict)
    # pretty_print(res)
    return dat, build_compare_feature_selection_protocole(res), (real_values,elbow_values,cusum_values, otsu_values, jenkspy_values)


def analyse_files(
        models_name,
        metrics,
        files,
        classic_f,
        macro_metrics,
        result_folder,
        list_of_accuracy,
        layer,
        metricA='accuracy'

):
    for index4, model in enumerate(models_name):
        # now fetch ours results to store
        for metric in list(set(classic_f.columns) & set(metrics)):  # accuracy and f1-score and/or financial cost
            for approach in list(files.keys()):  # Mlc or MCA
                for logic in list(files[approach].keys()):  # GLO, PER or GAP
                    for config in list(files[approach][logic].keys() if files[approach][logic] is not None else {}):  # MX, CX, CY, CXY
                        # print(result_folder, layer, approach, logic, config, len(files[approach][logic][config]) )
                        for result in list(range(len(
                                files[approach][logic][config]))):  # each result file's containing evaluation metrics
                            # print(files[approach][logic][config][result].columns)
                            # exit(0)
                            # valu = round(
                            #     (
                            #             (
                            #                     round(
                            #                         files[approach][logic][config][result].loc[model, metric],
                            #                         4
                            #                     )
                            #                     -
                            #                     round(
                            #                         classic_f.loc[model, metric],
                            #                         4
                            #                     )
                            #             )
                            #             /
                            #             (round(classic_f.loc[model, metric], 4) if round(classic_f.loc[model, metric],
                            #                                                              4) > 0 else epsilon)
                            #     ) * 100,
                            #     1
                            # )

                            def calculate_gain_percentage(new_value, baseline_value, epsilon=1e-8):
                                """Calcule le pourcentage de gain entre deux valeurs."""
                                new_rounded = round(new_value, 4)
                                baseline_rounded = round(baseline_value, 4)

                                # Éviter la division par zéro
                                denominator = baseline_rounded if baseline_rounded != 0 else epsilon

                                gain = ((new_rounded - baseline_rounded) / denominator) * 100
                                return gain

                            # Usage
                            valu1 = calculate_gain_percentage(
                                files[approach][logic][config][result].loc[model, metric],
                                classic_f.loc[model, metric]
                            )

                            # main_key = [kjj for kjj in list(macro_metrics['gain'].keys()) if
                            #             (logic in kjj) and (approach in kjj) and (config in kjj)]
                            main_key = [kjj for kjj in list(macro_metrics['gain'].keys()) if
                                        kjj == f"{approach}_{logic}_{config}"]
                            # met = 'Acc' if 'acc' in metric else ('F1-score' if 'f1' in metric else 'Cost')
                            macro_metrics['gain'][main_key[0]][metric][result_folder].append(
                                (
                                    valu1 if 'finan' not in metric else (valu1 if valu1 == 0 else -1 * valu1)
                                    , model)
                            )
                            macro_metrics['real'][main_key[0]][metric][result_folder].append(
                                (round(
                                    files[approach][logic][config][result].loc[model, metric],
                                    4
                                ),model)
                            )
                            macro_metrics['classic'][metric][result_folder].append((round(classic_f.loc[model, metric], 4),model))

                            # if 'LD4' in result_folder:
                            #     print(result_folder, layer, config)
                            if metricA in metric and 'MX' in config:
                                # if 'LD4' in result_folder:
                                # print(result_folder, layer) list_of_f1
                                list_of_accuracy.append((layer, model, files[approach][logic][config][result].loc[model, metric]))
                            # if 'f1' in metric and 'MX' in config:
                            #     # if 'LD4' in result_folder:
                            #     # print(result_folder, layer) list_of_f1
                            #     list_of_f1.append((layer, model, files[approach][logic][config][result].loc[model, metric]))
    # print(layer,list_of_accuracy)
    # print(layer)
    return (list_of_accuracy, macro_metrics)

def count_pattern_matches(dataframe, pattern):
    """Count how many feature names in dataframe match the regex pattern."""
    return sum(1 for key in dataframe.index if re.match(pattern, key))


def sum_absolute_matches(dataframe, pattern):
    """Calcule la somme des valeurs absolues des colonnes pour les lignes correspondant au pattern."""
    # On filtre les index qui correspondent au regex pattern
    matched_df = dataframe[dataframe.index.str.contains(pattern, regex=True)]

    # On calcule la somme des valeurs absolues pour chaque ligne (axis=1)
    # puis on retourne la somme totale ou la série selon votre besoin
    return matched_df.abs().sum(axis=1)


def sum_absolute_matches_total(dataframe, pattern):
    """Retourne la somme globale des valeurs absolues pour les lignes dont l'index matche le pattern."""
    # 1. Sélectionner les lignes via l'index avec le regex
    matched_df = dataframe[dataframe.index.str.contains(pattern, regex=True, na=False)]

    # 2. Somme de toutes les valeurs absolues (toutes lignes et colonnes confondues)
    return matched_df.abs().to_numpy().sum()

def analyse_files_for_shap_value(
        models_name,
        files,
        result_folders,
        top=10,
        n=2
):
    template_descripteurs = {
        'INTER_GLO_CX': [],
        'INTER_GLO_MX': [],
        'INTRA_GLO_CX': [],
        'INTRA_GLO_MX': [],
        'COMBINE_GLO_CX': [],
        'COMBINE_GLO_MX': [],
        'INTER_PER_CX': [],
        'INTER_PER_MX': [],
        'INTRA_PER_CX': [],
        'INTRA_PER_MX': [],
        'COMBINE_PER_CX': [],
        'COMBINE_PER_MX': [],
        'INTER_M_GLO_CX': [],
        'INTER_M_GLO_MX': [],
        'INTRA_M_GLO_CX': [],
        'INTRA_M_GLO_MX': [],
        'COMBINE_M_GLO_CX': [],
        'COMBINE_M_GLO_MX': [],
        'INTER_M_PER_CX': [],
        'INTER_M_PER_MX': [],
        'INTRA_M_PER_CX': [],
        'INTRA_M_PER_MX': [],
        'COMBINE_M_PER_CX': [],
        'COMBINE_M_PER_MX': [],

        'YN_COMBINE_PER': [],
        'YP_COMBINE_PER': [],
        'YN_INTER_PER': [],
        'YP_INTER_PER': [],
        'YN_INTRA_PER': [],
        'YP_INTRA_PER': [],
        'DEGREE': []

    }
    global_details_metrics_depth_1 = {}
    p=3
    # pretty_print(files)
    for index4, model in enumerate(models_name):
        global_details_metrics_depth_1[model] = {key: deepcopy(template_descripteurs) for key in
                                                 result_folders}
        for result_folder in list(files.keys()):
            for result in list(range(len(files[result_folder]["MlC"]["BOT"]["CXY"]))):  # each result file's containing evaluation metrics
                # print(result, len(files[result_folder]["MlC"]["BOT"]["CXY"]), files[result_folder]["MlC"]["BOT"]["CXY"][result].columns)
                if sum([k in files[result_folder]["MlC"]["BOT"]["CXY"][result].columns for k in ["precision", "accuracy", "recall", "f1-score"]]) == 4:
                    files[result_folder]["MlC"]["BOT"]["CXY"][result].drop(["precision", "accuracy", "recall", "f1-score"], axis=1, inplace=True)
                colo = files[result_folder]["MlC"]["BOT"]["CXY"][result].columns
                # print(colo)
                for att in colo:
                    if not (att in ["accuracy", "precision", "recall", "f1-score", "financial-cost"]):
                        if YN_PER_INTER_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'YN_INTER_PER'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif YP_PER_INTER_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'YP_INTER_PER'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif YN_PER_INTRA_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'YN_INTRA_PER'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif YP_PER_INTRA_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'YP_INTRA_PER'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif YN_PER_COMBINE_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'YN_COMBINE_PER'].append(
                                [abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif YP_PER_COMBINE_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'YP_COMBINE_PER'].append(
                                [abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_INTER_F(att) and GLO_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_GLO_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_INTER_F(att) and GLO_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_GLO_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_INTRA_F(att) and GLO_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_GLO_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_INTRA_F(att) and GLO_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_GLO_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_COMBINE_F(att) and GLO_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_GLO_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_COMBINE_F(att) and GLO_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_GLO_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTER_F(att) and PER_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_PER_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTER_F(att) and PER_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_PER_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTRA_F(att) and PER_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_PER_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTRA_F(att) and PER_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_PER_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_COMBINE_F(att) and PER_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_PER_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_COMBINE_F(att) and PER_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_PER_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_INTER_M_F(att) and GLO_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_M_GLO_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_INTER_M_F(att) and GLO_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_M_GLO_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_INTRA_M_F(att) and GLO_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_M_GLO_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_INTRA_M_F(att) and GLO_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_M_GLO_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_COMBINE_M_F(att) and GLO_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_M_GLO_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_COMBINE_M_F(att) and GLO_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_M_GLO_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTER_M_F(att) and PER_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_M_PER_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTER_M_F(att) and PER_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_M_PER_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTRA_M_F(att) and PER_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_M_PER_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTRA_M_F(att) and PER_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_M_PER_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_COMBINE_M_F(att) and PER_MX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_M_PER_MX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_COMBINE_M_F(att) and PER_CX_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_M_PER_CX'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif DEGREE_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'DEGREE'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        else:
                            var = att
                            if '__' in att:
                                _, var = att.split('__', 1)
                            if not (var in global_details_metrics_depth_1[model][result_folder].keys()):
                                global_details_metrics_depth_1[model][result_folder][
                                    var] = []
                                # print(global_details_metrics_depth_1[model][result_folder])
                            global_details_metrics_depth_1[model][result_folder][
                                var].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])

    # pretty_print(global_details_metrics_depth_1)
    for model in global_details_metrics_depth_1.keys():
        for folder in global_details_metrics_depth_1[model].keys():
            for att in global_details_metrics_depth_1[model][folder].keys():
                # print(global_details_metrics_depth_1[model][folder][att], model, folder, att)
                # print(model, folder)
                # pretty_print(global_details_metrics_depth_1[model][folder])
                # pretty_print(np.mean(global_details_metrics_depth_1[model][folder][att], axis=0))
                global_details_metrics_depth_1[model][folder][att] = np.mean(global_details_metrics_depth_1[model][folder][att], axis=0)

    tab_shapes = { key: [] for key in models_name }


    # pretty_print(global_details_metrics_depth_1)
    summary = {model: {folder: {'Classic':0, 'GLO_MX': 0,'PER_MX': 0, 'GLO_CX': 0, 'PER_CX': 0,  'PER_CY': 0} for folder in result_folders} for model in models_name}
    counter1 = {'Classic':0, 'GLO_MX': 0,'PER_MX': 0, 'GLO_CX': 0, 'PER_CX': 0,  'PER_CY': 0}
    for i, model in enumerate(models_name):
        tt = []
        # shapes += """
        # """
        for fold in files.keys():
            # print('>>>>>>>>>', fold, model)
            agg = pd.DataFrame(global_details_metrics_depth_1[model][fold]).T
            agg.columns = [f'Classe {i}' for i in range(agg.shape[1])]

            # Calculer l'importance totale et trier (DÉCROISSANT = plus important en premier)
            agg['total_importance'] = agg.abs().sum(axis=1)
            agg = agg.sort_values('total_importance', ascending=False)
            agg = agg.drop('total_importance', axis=1)
            summary[model][fold]['all'] = sum_absolute_matches_total(agg.head(top), r'.*')
            summary[model][fold]['GLO_MX'] = round(sum_absolute_matches_total(agg.head(top), r'^.*_GLO_MX$')/summary[model][fold]['all'], p)
            summary[model][fold]['PER_MX'] = round(sum_absolute_matches_total(agg.head(top), r'^.*_PER_MX$')/summary[model][fold]['all'], p)
            summary[model][fold]['GLO_CX'] = round(sum_absolute_matches_total(agg.head(top), r'^.*_GLO_CX$')/summary[model][fold]['all'], p)
            summary[model][fold]['PER_CX'] = round(sum_absolute_matches_total(agg.head(top), r'^.*_PER_CX$')/summary[model][fold]['all'], p)
            summary[model][fold]['PER_CY'] = round(sum_absolute_matches_total(agg.head(top), r'^Y.*_PER$')/summary[model][fold]['all'], p)
            summary[model][fold]['Classic'] = round(sum_absolute_matches_total(agg.head(top), r'^(?!.*(?:_GLO_CX|_GLO_MX|_PER_MX|_PER_CX|Y.*_PER)).*$')/summary[model][fold]['all'], p)
            summary[model][fold]['nbAtt'] = len(agg.index)

            tt.append(f"""{{{create_standalone_shap_plot(agg.head(top), model, fold, top)}}}""")

        tab_shapes[model] = tt
        # pretty_print(summary)
        # ----------------------------- print the summary ----------------------------
        table_header = """
            %\\begin{sidewaystable}
            \\resizebox{\\textwidth}{!}{

            \\begin{tabular}{|c|c|"""

        # setup information columns headears
        nbMCol = 1+len(global_details_metrics_depth_1)
        table_header += "r|" * nbMCol
        # add col for total results
        table_header += "} "
        # add separator clines
        nb_cols = (2 + nbMCol)
        table_header += " \\cline{1-" + str(nb_cols) + "}"  # corresponding to the number of columns

        # build the first line: metrics' line
        lines = ''
        # add the blank block
        lines += """
            \\multicolumn{2}{|c|}{}"""
        # add cols for metric
        for model in models_name:
            lines += " & \\textbf{" + model + "}"
        # add the total name
        lines += " & \\textbf{TOTAL} \\\\ "
        lines += " \\cline{1-" + str(nb_cols) + """}
            """

        # lambda function for sum
        sum_type = lambda store, folder, type: sum([store[model][folder][type] for model in store.keys()])
        max_sum_type = lambda store, folder, types: max(
            [sum([store[model][folder][type] for model in store.keys()]) for type in types])
        sum_mod = lambda store, model, types: sum(
            [store[model][folder][type] for folder in store[model].keys() for type in types])
        max_sum_mod = lambda store, types: max(
            [sum([store[model][folder][type] for folder in store[model].keys() for type in types]) for model in
             store.keys()])
        # fetch folders
        for folder in files.keys():
            # print(summary)
            lines += """
                \\multirow{3}{*}{\\textbf{""" + folder + """ (""" + str(summary[list(models_name.keys())[0]][folder]['nbAtt']) + """)}}
                """
            # fetch descriptors type
            for di, desc in enumerate(counter1.keys()):
                lines += """& """ + str(desc).replace('_', '\\_')
                # fetch models
                for model in models_name:
                    # add desc info for each model
                    lines += """& """ + str(summary[model][folder][desc])
                # add total of the current
                lines += """& """ + str(round(sum_type(summary, folder, desc),p)) if max_sum_type(summary, folder,
                                                                                         counter1.keys()) != sum_type(
                    summary, folder, desc) else "& \\textbf{" + str(round(sum_type(summary, folder, desc),p)) + "}"
                # back to next line
                lines += " \\\\ "
                lines += " \\cline{1-" + str(nb_cols) + """}
                    """ if di == len(counter1.keys()) - 1 else " \\cline{2-" + str(nb_cols) + """}
                    """
        # add total line
        lines += """
            \\multicolumn{2}{|c|}{\\textbf{TOTAL}}"""
        for model in models_name:
            lines += " & " + str(round(sum_mod(summary, model, list({*list(counter1.keys())}-{'Classic'})),p)) if sum_mod(summary, model,
                                                                                      list({*list(counter1.keys())}-{'Classic'})) != max_sum_mod(
                summary, list({*list(counter1.keys())}-{'Classic'})) else "& \\textbf{" + str(round(sum_mod(summary, model, list({*list(counter1.keys())}-{'Classic'})),p)) + "}"
        lines += " & "
        lines += " \\\\ "
        lines += " \\cline{1-" + str(nb_cols) + """}
            \\end{tabular}}"""



    sub_shapes = []
    items = list(tab_shapes.items())
    for i in range(0, len(items), n):
        groupe = dict(items[i:i + n])
        # print(groupe)
        sous_dict = dict(groupe)  # Si vous avez besoin de manipuler un dictionnaire
        shapes = """
        \\begin{table}[H]
        \\center
        \\resizebox{\\textwidth}{!}{
        \\begin{tabular}{"""+("|c"*len(sous_dict.keys()))+"""|}
        \\hline
        
        """+"&".join(["""\\textbf{"""+model+"""}""" for model in sous_dict.keys()])

        for i in range(len(sous_dict[list(sous_dict.keys())[0]])):
            shapes += """\\\\
            \\hline
            """
            store = []
            for model in sous_dict.keys():
                store.append(sous_dict[model][i])
            shapes += """ 
            &
             """.join(store)
        shapes+="""\\\\ 
        \\hline
        \\end{tabular}}
        \\label{tab:"""+("-".join(sous_dict.keys()))+"""}
        \\end{table}
        """
        sub_shapes.append(shapes)

    res = """
          """.join(sub_shapes)+ """
            \\begin{table}[H]
            \\center"""+ table_header + lines +"""
            \\caption{Nombre d'attributs de type GLO, PER et PER\\_Y des graphes multicouches qui sont présents dans le Top-20 des attributs qui contribuent le plus aux décisions des modèles de prédiction dans les différents jeux de données. Entre parenthèse, c'est le nombre total d'attributs du jeu de données associé.}
            \\label{tab:recapTab}
            \\end{table}"""

    return res

# === 4. Définir les couleurs pour chaque classe ===
class_colors = {
    'Classe 0': '{RGB}{255,20,147}',  # Rose/Magenta
    'Classe 1': '{RGB}{34,139,34}',  # Vert
    'Classe 2': '{RGB}{255,69,0}',  # Rouge
    'Classe 3': '{RGB}{30,144,255}',  # Bleu
}

def get_additional_colors(n_classes):
    """Génère des couleurs supplémentaires si plus de 4 classes"""
    additional_colors = [
        '{RGB}{148,0,211}',  # Violet
        '{RGB}{255,140,0}',  # Orange foncé
        '{RGB}{0,191,255}',  # Bleu ciel
        '{RGB}{50,205,50}',  # Vert clair
    ]

    all_colors = {}
    base_colors = list(class_colors.values())

    for i in range(n_classes):
        if i < len(base_colors):
            all_colors[f'Classe {i}'] = base_colors[i]
        else:
            all_colors[f'Classe {i}'] = additional_colors[(i - len(base_colors)) % len(additional_colors)]

    return all_colors
def create_standalone_shap_plot(df_shap, model_name, fd, top):
    """Crée un SHAP summary plot standalone pour un modèle"""

    n_features, n_classes = df_shap.shape
    feature_names = [name.replace("_", "\\_") for name in df_shap.index.tolist()]
    class_names = df_shap.columns.tolist()

    # Obtenir les couleurs pour toutes les classes
    colors = get_additional_colors(n_classes)

    # Calculer les valeurs maximales pour l'axe X
    max_val = df_shap.abs().max().max()
    x_max = max_val * 1.1

    latex_content = f"""
% Définition des couleurs pour chaque classe
"""

    # Ajouter les définitions de couleurs
    for class_name, color_def in colors.items():
        color_key = class_name.lower().replace(' ', '')
        latex_content += f"\\definecolor{{{color_key}}}{color_def}\n"

    latex_content += f"""
%\\begin{{figure}}[H]
%\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    xbar stacked,
    width=14cm,
    height={max(8, n_features * 0.4)}cm,
    xlabel={{Average importance |SHAP| by class}},
    ylabel={{Variables}},
    xmin=0,
    xmax={x_max:.3f},
    ytick=data,
    yticklabels={{%
        {', '.join(feature_names)}
    }},
    bar width={max(6, 20 - n_features)}pt,
    legend style={{
        at={{(0.98,0.02)}},
        anchor=south east,
        legend columns=1
    }},
    grid=major,
    grid style={{gray!30}},
    title={{{model_name} - {{{fd}}} - SHAP Summary Plot}},
    y dir=reverse,
    yticklabel style={{font=\small}}
]

"""

    # Ajouter les données pour chaque classe
    for i, class_name in enumerate(class_names):
        color_key = class_name.lower().replace(' ', '')
        values = df_shap[class_name].abs().values

        # PAS d'inversion ici - on garde l'ordre du DataFrame (déjà trié par importance décroissante)
        coordinates = ' '.join([f"({val:.4f},{j})" for j, val in enumerate(values)])

        latex_content += f"""% {class_name}
\\addplot[fill={color_key}, draw=none] coordinates {{
    {coordinates}
}};

"""

    # Ajouter la légende
    legend_entries = ', '.join(class_names)
    latex_content += f"""\\legend{{{legend_entries}}}
\\end{{axis}}
\\end{{tikzpicture}}
%\\caption{{Importance des variables SHAP pour le modèle {model_name} (ordre décroissant d'importance top {top})}}
%\\label{{fig:shap_{fd}_{model_name.replace(' ', '_').lower()}}}
%\\end{{figure}}
"""

    return latex_content

# === 3. Convertir les vecteurs SHAP de string en liste de float ===
def parse_vector(text):
    try:
        # Enlever les crochets et séparer par les espaces
        # print(type(text),text, "//")
        nombres_str = text.strip('[]').split()
        # Convertir en float
        # print(nombres_str, "..")
        liste = [float(x) for x in nombres_str]
        # print(liste, ".-.")
        return liste
    except Exception as e:
        print(f"Erreur parsing: {text} → {e}")
        return np.nan