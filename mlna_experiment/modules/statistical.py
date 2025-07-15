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
        match= lambda x: True,
        attributs= [],
        isBest=False,
        dataset_delimiter=None,
        encoding=None,
        index_col=None

):
    """
    Load results from a results folder
    Parameters
    ----------
    outputs_path: path of results folder
    _type: type of results
    k: layer
    per: flag to indicate whether to load per-layer results
    glo: flag to indicate whether to load glo results
    mix: flag to indicate whether to load mix results
    bot: flag to indicate whether to load both results
    isRand: flag to indicate if it's a random combinaison layer
    match: lambda func to a specific kind of file
    attributs: names of the actual processing attributs



    Returns
    -------

    """
    files = {
        'MlC':{
            'GLO':{
                'MX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/global/withoutClass/evaluation',
                        func=lambda x: (MlC_F(x) and GLO_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/global/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and GLO_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            } if glo is True else None,
            'PER':{
                'MX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withoutClass/evaluation',
                        func=lambda x: (MlC_F(x) and PER_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and PER_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and PER_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and PER_CXY_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            } if per is True else None,
            'GAP':{
                'MX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withoutClass/evaluation',
                        func=lambda x: (MlC_F(x) and GAP_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and GAP_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and GAP_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and GAP_CXY_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            } if mix is True else None,
            'BOT':{
                'CXY': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col,
                                           na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/select/mlna_{k}_b/{attribut}/mixed/both/evaluation',
                        func=lambda x: (MlC_F(x) and BOT_CXY_F(x) and (match(x))),
                        verbose=False
                    )]
            } if bot is True else None
        },
        'MCA':{
            'GLO':{
                'MX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/global/withoutClass/evaluation',
                        func=lambda x: (MCA_F(x) and GLO_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/global/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and GLO_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            } if glo is True else None,
            'PER':{
                'MX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withoutClass/evaluation',
                        func=lambda x: (MCA_F(x) and PER_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and PER_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and PER_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and PER_CXY_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            } if per is True else None,
            'GAP':{
                'MX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withoutClass/evaluation',
                        func=lambda x: (MCA_F(x) and GAP_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and GAP_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and GAP_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}{"/select" if isBest is True else ""}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and GAP_CXY_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            } if mix is True else None,
            'BOT': {
                'CXY': [
                    load_data_set_from_url(path=file, sep=dataset_delimiter, encoding=encoding, index_col=index_col,
                                           na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/select/mlna_{k}_b/{attribut}/mixed/both/evaluation',
                        func=lambda x: (MCA_F(x) and BOT_CXY_F(x) and (match(x))),
                        verbose=False
                    )]
            } if bot is True else None
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

        \\begin{tabular}{|c|c|""" + ("c|" * len(alphas))
    # setup information columns headears
    nbMCol = 10
    # add col for total results
    table_header += "} "
    # add separator clines
    nb_cols = (2 + nbMCol)
    table_header += " \\cline{1-" + str(len(alphas)+2) + "}"  # corresponding to the number of columns

    # build the first line: metrics' line
    lines = ''
    # add the blank block
    lines += """
    \\multicolumn{2}{|c|}{}"""


    # add alpha for metric
    for alpha in alphas:
        lines += f" & {alpha}"
    # add the total name
    lines += " \\\\ "
    lines += " \\cline{1-" + str(len(alphas)+2) + """}
    """
    for folder in datasets:
        # fetch on model
        lines += """
            \\multirow{3}{*}{""" + folder + """}"""
        for mi, meth in enumerate(methods):
            lines += f""" & {meth}"""
            for ai, alpha in enumerate(alphas):  # MlC, MCA
                lines += f""" & {store[folder][meth][alpha]}""" if alpha in list(store[folder][meth].keys()) else " & "
            lines += ("""\\\\ """ + """ \\cline{2-""" + str(len(alphas)+2) + """}

                    """) if mi != len(methods) - 1 else ("""\\\\ """ + """ \\cline{1-""" + str(len(alphas)+2) + """}

                    """)

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

def elbow_method(accuracies):
    sorted_accuracies = sorted(accuracies, reverse=True)
    coords = [(i, acc) for i, acc in enumerate(sorted_accuracies)]
    line_vec = coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1]
    line_vec_norm = math.sqrt(sum(x*x for x in line_vec))
    vec_from_first = lambda coord: (coord[0] - coords[0][0], coord[1] - coords[0][1])
    scalar_proj = lambda vec: (vec[0]*line_vec[0] + vec[1]*line_vec[1]) / line_vec_norm
    vec_proj = lambda vec: ((scalar_proj(vec) / line_vec_norm) * line_vec[0], (scalar_proj(vec) / line_vec_norm) * line_vec[1])
    vec_reject = lambda vec: (vec[0] - vec_proj(vec)[0], vec[1] - vec_proj(vec)[1])
    dists_from_line = [euclidean((0,0), vec_reject(vec_from_first(coord))) for coord in coords]
    return dists_from_line.index(max(dists_from_line)) + 1

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
        'matrices_per_row': 2,
        'matrix_spacing_x': 4,
        'matrix_spacing_y': 5,
        'cell_width': '1.5cm',
        'cell_height': '0.6cm',
        'header_width': '3.05cm',
        'header_height': '0.6cm'
    }

    if layout_config:
        default_config.update(layout_config)

    config = default_config

    # Début du document LaTeX
    latex_content = f"""
\\begin{{tikzpicture}}

"""

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
    nodes={{draw, minimum width={config['cell_width']}, minimum height={config['cell_height']}, anchor=center, text centered}}] at ({x},{y})
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
\\node[draw, rectangle, minimum width={config['header_width']}, minimum height={config['header_height']}] ({matrix_name}_title) at ({x + 0.75 if x == 0 else x},{header_y}) {{{dataset}}};

"""

    # Fermer le document
    latex_content += """\\end{tikzpicture}
"""

    return latex_content
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

def selection_proto(records, output_path):
    # result structure
    resultDict = {
        'dataset': [],
        'alpha': [],
        # 'QuartileThreshold':[],
        'elbowThreshold': [],
        'cumulative_difference_threshold': [],
        # 'variance_explained_threshold':[],
        'realThreshold': []
    }
    res = {}
    getTheBestAcc = lambda store, k: round(max([acc for layer, _, acc in store if k == layer]), 4)
    real_values = {}
    elbow_values = {}
    cusum_values = {}
    # walk on the datasets
    for dataset in records.keys():
        # walk on alphas
        res[dataset] = {key: {} for key in ['CUSUM', 'Elbow', 'réel']}
        real_values[dataset] = []
        elbow_values[dataset] = []
        cusum_values[dataset] = []
        for alpha in records[dataset].keys():
            if not isinstance(records[dataset][alpha]['predicted_best_k'], list):
                resultDict['dataset'].append(dataset)
                resultDict['alpha'].append(alpha)
                # resultDict['QuartileThreshold'].append(getTheBestAcc(records[dataset][alpha]['list'],records[dataset][alpha]['predicted_best_k'][0]))
                elb = elbow_method(list(records[dataset][alpha]['accuracies']))
                elb = elb if elb < max([layer for layer, _, _ in records[dataset][alpha]['list']]) else max(
                    [layer for layer, _, _ in records[dataset][alpha]['list']])
                resultDict['elbowThreshold'].append(getTheBestAcc(records[dataset][alpha]['list'], elb))
                res[dataset]['Elbow'][alpha] = getTheBestAcc(records[dataset][alpha]['list'], elb)
                elbow_values[dataset].append(res[dataset]['Elbow'][alpha])

                cum = cumulative_difference_threshold(list(records[dataset][alpha]['accuracies']))
                cum = cum if cum < max([layer for layer, _, _ in records[dataset][alpha]['list']]) else max(
                    [layer for layer, _, _ in records[dataset][alpha]['list']])
                resultDict['cumulative_difference_threshold'].append(
                    getTheBestAcc(records[dataset][alpha]['list'], cum))
                res[dataset]['CUSUM'][alpha] = getTheBestAcc(records[dataset][alpha]['list'], cum)
                cusum_values[dataset].append(res[dataset]['CUSUM'][alpha])

                # resultDict['variance_explained_threshold'].append(getTheBestAcc(records[dataset][alpha]['list'],variance_explained_threshold(list(records[dataset][alpha]['accuracies'].values()))))
                resultDict['realThreshold'].append(
                    round(max([acc for _, _, acc in records[dataset][alpha]['list']]), 4))
                res[dataset]['réel'][alpha] = round(max([acc for _, _, acc in records[dataset][alpha]['list']]), 4)
                real_values[dataset].append(res[dataset]['réel'][alpha])

    dat = pd.DataFrame(resultDict)
    return dat, build_compare_feature_selection_protocole(res), (real_values,elbow_values,cusum_values)


def analyse_files(
        models_name,
        metrics,
        files,
        classic_f,
        macro_metrics,
        result_folder,
        list_of_accuracy,
        layer
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
                            valu = round(
                                (
                                        (
                                                round(
                                                    files[approach][logic][config][result].loc[model, metric],
                                                    4
                                                )
                                                -
                                                round(
                                                    classic_f.loc[model, metric],
                                                    4
                                                )
                                        )
                                        /
                                        (round(classic_f.loc[model, metric], 4) if round(classic_f.loc[model, metric],
                                                                                         4) > 0 else epsilon)
                                ) * 100,
                                1
                            )

                            valu1 = (
                                            (
                                                    round(
                                                        files[approach][logic][config][result].loc[model, metric],
                                                        4
                                                    )
                                                    -
                                                    round(
                                                        classic_f.loc[model, metric],
                                                        4
                                                    )
                                            )
                                            /
                                            (round(classic_f.loc[model, metric], 4) if round(
                                                classic_f.loc[model, metric], 4) > 0 else epsilon)
                                    ) * 100

                            main_key = [kjj for kjj in list(macro_metrics.keys()) if
                                        (logic in kjj) and (approach in kjj) and (config in kjj)]
                            # met = 'Acc' if 'acc' in metric else ('F1-score' if 'f1' in metric else 'Cost')
                            macro_metrics[main_key[0]][metric][result_folder].append(
                                (
                                    valu1 if 'finan' not in metric else (valu1 if valu1 == 0 else -1 * valu1)
                                    , model)
                            )

                            # if 'LD4' in result_folder:
                            #     print(result_folder, layer, config)
                            if 'acc' in metric and 'MX' in config:
                                # if 'LD4' in result_folder:
                                # print(result_folder, layer)
                                list_of_accuracy.append((layer, model, files[approach][logic][config][result].loc[model, metric]))
    return (list_of_accuracy, macro_metrics)

def analyse_files_for_shap_value(
        models_name,
        files,
        result_folders,
        top=10
):
    template_descripteurs = {
        'INTER_GLO': [],
        'INTRA_GLO': [],
        'COMBINE_GLO': [],
        'INTER_PER': [],
        'INTRA_PER': [],
        'COMBINE_PER': [],
        'INTER_M_GLO': [],
        'INTRA_M_GLO': [],
        'COMBINE_M_GLO': [],
        'INTER_M_PER': [],
        'INTRA_M_PER': [],
        'COMBINE_M_PER': [],

        'YN_COMBINE_PER': [],
        'YP_COMBINE_PER': [],
        'YN_INTER_PER': [],
        'YP_INTER_PER': [],
        'YN_INTRA_PER': [],
        'YP_INTRA_PER': [],
        'DEGREE': []

    }
    global_details_metrics_depth_1 = {}
    for index4, model in enumerate(models_name):
        global_details_metrics_depth_1[model] = {key: deepcopy(template_descripteurs) for key in
                                                 result_folders}
        for result_folder in result_folders:
            # print(result_folder)
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
                        elif GLO_INTER_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_GLO'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_INTRA_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_GLO'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_COMBINE_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_GLO'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTER_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_PER'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTRA_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_PER'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_COMBINE_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_PER'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_INTER_M_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_M_GLO'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_INTRA_M_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_M_GLO'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif GLO_COMBINE_M_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_M_GLO'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTER_M_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTER_M_PER'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_INTRA_M_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'INTRA_M_PER'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
                        elif PER_COMBINE_M_F(att):
                            global_details_metrics_depth_1[model][result_folder][
                                'COMBINE_M_PER'].append([abs(x) for x in parse_vector(files[result_folder]["MlC"]["BOT"]["CXY"][result].loc[model, att])])
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

    for model in global_details_metrics_depth_1.keys():
        for folder in global_details_metrics_depth_1[model].keys():
            for att in global_details_metrics_depth_1[model][folder].keys():
                # print(global_details_metrics_depth_1[model][folder][att], model, folder, att)
                global_details_metrics_depth_1[model][folder][att] = np.mean(global_details_metrics_depth_1[model][folder][att], axis=0)

    shapes = f"""
    """
    for model in models_name:
        tt = []
        shapes += """
        """
        for fold in result_folders:
            agg = pd.DataFrame(global_details_metrics_depth_1[model][fold]).T
            agg.columns = [f'Classe {i}' for i in range(agg.shape[1])]

            # Calculer l'importance totale et trier (DÉCROISSANT = plus important en premier)
            agg['total_importance'] = agg.abs().sum(axis=1)
            agg = agg.sort_values('total_importance', ascending=False)
            agg = agg.drop('total_importance', axis=1)

            tt.append(f"{{{create_standalone_shap_plot(agg.head(top), model, fold, top)}}}")
        shapes += """
        """.join(tt)

    return shapes

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
\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    xbar stacked,
    width=14cm,
    height={max(8, n_features * 0.4)}cm,
    xlabel={{Importance moyenne |SHAP| par classe}},
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
\\caption{{Importance des variables SHAP pour le modèle {model_name} (ordre décroissant d'importance top {top})}}
\\label{{fig:shap_{fd}_{model_name.replace(' ', '_').lower()}}}
\\end{{figure}}"""

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