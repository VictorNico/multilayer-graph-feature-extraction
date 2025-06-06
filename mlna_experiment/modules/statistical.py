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

# coding: utf-8

## 0.------- Module loading
import statistics
import os
from .file import *
from .report import *
from unidecode import unidecode
import re
import time
from copy import deepcopy
from collections import Counter
import json
from colorama import init, Fore, Style

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
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %SetFonts

    %SetFonts


    \\title{Prediction du risque de credit bancaire sensible aux coûts financiers en intégrant des descripteurs extraits des graphes \\ Tableaux récapitulatifs}
    \\author{The Author}
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
        isRand=False,
        match= lambda x: True,
        attributs= []

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
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/global/withoutClass/evaluation',
                        func=lambda x: (MlC_F(x) and GLO_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/global/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and GLO_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            },
            'PER':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withoutClass/evaluation',
                        func=lambda x: (MlC_F(x) and PER_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and PER_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and PER_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and PER_CXY_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            },
            'GAP':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withoutClass/evaluation',
                        func=lambda x: (MlC_F(x) and GAP_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and GAP_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and GAP_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MlC_F(x) and GAP_CXY_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            }
        },
        'MCA':{
            'GLO':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/global/withoutClass/evaluation',
                        func=lambda x: (MCA_F(x) and GLO_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/global/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and GLO_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            } if glo is True else None,
            'PER':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withoutClass/evaluation',
                        func=lambda x: (MCA_F(x) and PER_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and PER_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and PER_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/personalized/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and PER_CXY_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            } if per is True else None,
            'GAP':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withoutClass/evaluation',
                        func=lambda x: (MCA_F(x) and GAP_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and GAP_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and GAP_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for attribut in attributs for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/{attribut}/mixed/withClass/evaluation',
                        func=lambda x: (MCA_F(x) and GAP_CXY_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            } if mix is True else None,
        }
    }
    return files

# --------------------------------------------------------------------------------------TablesPrinting-------------------------------------------------------------
# ################################################################
# ############## Tables Printing Functions #######################
# ################################################################
"""
    Cette fonction compare les impacts de la valeur d'alpha 
"""
def print_compare_v3_3(
    store,
    output_path,
    alpha,
    metrics,
    configs,
    result,
    valll=False,
    total_digits= 4,
    decimal_digits= 1
):
    
    alphas = sorted(list(store.keys()))
    models = list(store[alphas[0]].keys())
    folders = list(store[alphas[0]][models[0]].keys())
    # fetch model name
    tables = {folder:'' for folder in folders}
    counter = {metric: {alpha:0 for alpha in alphas} for metric in metrics }
    resume = []
    for folder in list(tables.keys()):
        # add the resize box to ensure the scale of the table will be contain's inside the width space avalable.
        # start setting up the tabular dimensions setting
        table_header = """
        %\\begin{sidewaystable}
        \\resizebox{\\textwidth}{!}{

        \\begin{tabular}{|c|c|"""

        # setup information columns headears
        nbMCol =  (len(alphas)+1)*len(metrics)
        table_header+= "r|"*nbMCol
        # add col for total results
        table_header+= "} "
        # add separator clines
        nb_cols = (2+nbMCol)
        table_header+= " \\cline{1-"+str(nb_cols)+"}" # corresponding to the number of columns


        # build the first line: metrics' line
        lines = ''
        # add the blank block
        lines += """
        \\multicolumn{2}{|c|}{"""+folder+"""}"""
        # add cols for metric
        for metric in metrics:
            lines+= " & \\multicolumn{"+str(nbMCol//len(metrics))+"}{|c|}{"+dicto[metric]+"}"
        # add the total name
        lines+= " \\\\ "
        lines+= " \\cline{3-"+str(nb_cols)+"""}
        """

        # build the second line: logics' line
        # add the blank block
        lines += """\\multicolumn{2}{|c|}{} """
        # add alphas
        for metric in metrics:
            lines+= " & Classic"
            for alpha in alphas:
                lines+= f" & {alpha}"

        # add the total name
        lines+= " \\\\ "
        lines+= " \\cline{1-"+str(nb_cols)+"""}

        """

        

        # here we need to define a structure that will keep track of the number of times a config has the best result for a given metric
        counterStore= {}
        for metric in metrics:
            counterStore[metric] = {}
            for alpha in alphas:
                counterStore[metric][alpha] = []

        counterStore_1 = deepcopy(counterStore)
        counterStore_2 = deepcopy(counterStore)

        # having done this, we now need a function that will return max of the metrics in a config with a given approach
        max_config = lambda store, alpha, mod, folder, metric, app, act: (
        act(
            [
            round(act(store[alpha][mod][folder][metric][app][logic][config]),2)
            # for app in store[alpha][mod][folder][metric].keys() 
            for logic in store[alpha][mod][folder][metric][app].keys() 
            for config in store[alpha][mod][folder][metric][app][logic].keys()
            ]
            )
        )

        get_all_max_config = lambda store, mod, folder, metric, app, act: (
        (
            [
            round(act(store[alpha][mod][folder][metric][app][logic][config]),2)
            for alpha in store.keys()
            for logic in store[alpha][mod][folder][metric][app].keys() 
            for config in store[alpha][mod][folder][metric][app][logic].keys()
            ]
            )
        )

        # fetch on model
        for mi, model in enumerate(models): 
            # print(store[alphas[0]].keys(),[alphas[0]],[model],[folder],[metrics[0]], "=----------=\n")
            lines+= """
            \\multirow{"""+str(len(list(store[alphas[0]][model][folder][metrics[0]].keys())))+"""}{*}{"""+model+"""}"""
            for ai, approach in enumerate(list(store[alphas[0]][model][folder][metrics[0]].keys())): # MlC, MCA
                lines+= f""" & {approach}"""
                for i, metric in enumerate(metrics):
                    lines+= """ & \\multirow{2}{*}{"""+str(round(store[alphas[0]][model][folder]['classic'].loc[model, metric],4))+"""}""" if ai == 0 else """ &"""
                    for ali,alpha in enumerate(alphas):
                        act = max if valll is False else (min if "financial-cost" in metric else max)
                        act_b = sorted(list(set(get_all_max_config(store, model, folder, metric, approach, act))), reverse= (False if ((valll == True) and ("financial-cost" in metric )) else True))
                        pos = act_b.index(max_config(store, alpha, model, folder, metric, approach, act))

                        # Adjust integer part to the required digits
                        int_part = str(int(max_config(store, alpha, model, folder, metric, act))).zfill(total_digits - decimal_digits) if total_digits != None else None
                        form = f"{int_part}.{str(max_config(store, alpha, model, folder, metric, act)).split('.')[1][:decimal_digits]}" if total_digits != None else str(round(max_config(store, alpha, model, folder, metric, approach, act),2))
                        val = "\\textbf{"+form+"}" if (pos == 0) and (max_config(store, alpha, model, folder, metric, approach, act) != 0.0) else (
                            "\\underline{"+form+"}" if (pos == 1) and (max_config(store, alpha, model, folder, metric, approach, act) != 0.0) else (
                                "\\textit{"+form+"}" if (pos == 2) and (max_config(store, alpha, model, folder, metric, approach, act) != 0.0) else 
                                form
                                )
                            )
                        if (pos == 0) and (max_config(store, alpha, model, folder, metric, approach, act) != 0.0) and (valll is True):
                                resume.extend([model,approach,alpha])
                        lines+= f"& {val}"
                        bests = get_all_max_config(store, model, folder, metric, approach, act)
                        counterStore[metric][alpha].append(1/(bests.count(max_config(store, alpha, model, folder, metric, approach, act))) if pos == 0 else 0)
                        counterStore_1[metric][alpha].append(1/(bests.count(max_config(store, alpha, model, folder, metric, approach, act))) if pos == 1 else 0)
                        counterStore_2[metric][alpha].append(1/(bests.count(max_config(store, alpha, model, folder, metric, approach, act))) if pos == 2 else 0)
                # ---------------------------------------
                if (ai != len(store[alphas[0]][model][folder][metrics[0]].keys())-1):
                    start = 2
                    prog = (nbMCol//len(metrics))
                    # print(start)
                    lines+= """\\\\ """
                    for _ in range(len(metrics)):
                        lines+= """ \\cline{"""+str(start)+"-"+str(start)+"""} \\cline{"""+str(start+2)+"""-"""+str(start+(nbMCol//len(metrics)))+"""}"""
                        start = start+prog
                    lines+= """

                    """
                else:
                    lines+= """\\\\ """+ """ \\cline{1-"""+str(nb_cols)+"""}

                        """ 

        # total's line
        # print(result)
        maxim = lambda dictio, metric: (max([ sum(dictio[metric][alpha]) for alpha in dictio[metric].keys() ]))
        
        if 1 in result:
            lines += """
            \\multicolumn{2}{|c|}{\\textbf{Total}}"""

            for j, metric in enumerate(metrics):
                lines += """ & """
                for alpha in alphas:
                    counter[metric][alpha]+= round(sum(counterStore[metric][alpha]),1)
                    lines += """ & """+str(round(sum(counterStore[metric][alpha]),1)) if round(sum(counterStore[metric][alpha]),1) != round(maxim(counterStore, metric),1)else """ & \\textbf{"""+str(round(sum(counterStore[metric][alpha]),1))+"}"

            lines+= " \\\\ "
            lines+= " \\cline{1-"+str(nb_cols)+"}"
        
        if 2 in result:
            lines += """
            \\multicolumn{2}{|c|}{\\textbf{Total 2nd Place}}"""

            for j, metric in enumerate(metrics):
                lines += """ & """
                for alpha in alphas:
                    lines += """ & """+str(round(sum(counterStore_1[metric][alpha]),1)) if sum(counterStore_1[metric][alpha]) != maxim(counterStore_1, metric) else """ & \\textbf{"""+str(round(sum(counterStore_1[metric][alpha]),1))+"}"

            lines+= " \\\\ "
            lines+= " \\cline{1-"+str(nb_cols)+"}"

        if 3 in result:
            lines += """
            \\multicolumn{2}{|c|}{\\textbf{Total 3th Place}}"""

            for j, metric in enumerate(metrics):
                lines += """ & """
                for alpha in alphas:
                    lines += """ & """+str(round(sum(counterStore_2[metric][alpha]),1)) if sum(counterStore_2[metric][alpha]) != maxim(counterStore_2, metric) else """ & \\textbf{"""+str(round(sum(counterStore_2[metric][alpha]),1))+"}"

            lines+= " \\\\ "
            lines+= " \\cline{1-"+str(nb_cols)+"}"

        lines+= """

        \\end{tabular}
        }
        %\\end{sidewaystable}"""

        table = table_header + lines
        tables[folder] = table
        create_domain(f"{output_path}/alpha/{folder}")
        filename1 = f"{output_path}/alpha/{folder}/tab_alpha_{folder}.tex"
        _file = open(filename1, "w")
        _file.write(table)
        _file.close()
    # print(counter)
    if (valll is True):
        print(f"""
            {valll}, 
            ///////////////////////
            {alpha}, 
            =======================
            {dict(Counter(resume))}
            -----------------------
            """)
    return tables

def print_g_b_impact_table(
    store,
    output_path,
    alpha,
    valll=False
):
    # fetch metric name
    tables = """
        
        """
    for metric in list(store.keys()):
        # add the resize box to ensure the scale of the table will be contain's inside the width space avalable.
        # start setting up the tabular dimensions setting
        table_header = f"%{metric}"+"""
        %\\begin{sidewaystable}
        \\resizebox{\\textwidth}{!}{
        \\begin{tabular}{|c|c|r|r|r|r|r|r|r|r|r|}"""

        # add separator clines
        nb_cols = 11
        table_header+= """ 
        \\cline{1-"""+str(nb_cols)+"}" # corresponding to the number of columns


        # build the first line
        lines = ""
        # add the metric block
        lines += """
        \\multicolumn{2}{|c|}{"""+metric+"""}"""

        # add cols coupling
        lines+= " & \\multicolumn{3}{|c|}{(Good MLN 1, Good MLN 1)} & \\multicolumn{3}{|c|}{(Bad MLN 1, Good MLN 1)} & \\multicolumn{3}{|c|}{(Bad MLN 1, Bad MLN 1)}"
        
        # add the total name
        lines+= " \\\\ "
        lines+= " \\cline{3-"+str(nb_cols)+"}"

        # build the second line
        # add the metric block
        lines += """
        \\multicolumn{2}{|c|}{}"""

        # add cols coupling
        lines+= "& Classic $>$ &  Classic $=$ & Classic $<$"*3
        
        # add the total name
        lines+= " \\\\ "
        lines+= " \\cline{1-"+str(nb_cols)+"}"


        # fetch store to fullfil the tables with content

        # fetch logics
        for li, logic in enumerate(list(store[metric]['MlC'].keys())): # GLO, PER or GAP
            lines+= """
            \\multirow{2}{*}{"""+logic+"""}"""

            # fetch approachs
            for ai, approach in enumerate(list(store[metric].keys())): # Mlc or MCA
                lines+= f"""& {approach}"""
                # fetch coupling
                best_f = lambda x : max(list(x.values()))
                for ci, couple in enumerate(store[metric][approach][logic]): #(Good MLN 1, Good MLN 1), (Bad MLN 1, Good MLN 1), (Bad MLN 1, Bad MLN 1)
                    best = best_f(couple)
                    for value in couple.values():
                        lines+= "& \\textbf{"+str(value)+"}" if value == best else f"& {value}"

                if ai == (len(list(store[metric].keys())) - 1):
                    lines+= " \\\\ "
                    lines+= " \\cline{1-"+str(nb_cols)+"""}
                    """
                else:
                    lines+= " \\\\ "
                    lines+= " \\cline{2-"+str(nb_cols)+"""}
                    """

        lines+= """

        \\end{tabular}}"""

        table = table_header + lines
        tables += """

        """+ table

    create_domain(f"{output_path}/coupling")
    filename1 = f"{output_path}/coupling/coupling.tex"
    _file = open(filename1, "w")
    _file.write(header+tables+footer)
    _file.close()
    return tables

# ################################################################
# ############## Tables Generation Logic Functions ###############
# ################################################################
"""
    Impact performance analytic for metrics: accuracy, f1-score and financial-cost for all configurations
"""

def extract_evaluations(
    outputs_name=None, 
    cwd=None, 
    outputPath=None,
    layers=[1], 
    glo=True,
    per=True,
    gap=True,
    logics=['GLO','PER','GAP'],
    approachs=['MlC','MCA'],
    configs=['MX','CX','CY','CXY'],
    metrics=['accuracy','f1-score','precision','recall','financial-cost'],
    metrics1=[],
    _type='cat',
    alphas=[0.85],
):
    """
    Analyzer Launcher
    Parameters
    ----------
    outputs_path
    cwd
    layers
    type

    Returns
    -------

    """
    ## identify the name of results folder
    data_result_folder_name = [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{outputs_name}')][0]
    data_result_folder_name =  list(set(data_result_folder_name)-set(['LFD']))
    # enregistrer les améliorations améliorations
    metrics = ["Acc", "F1-score", "Cost"]
    categories = ["CRD", "GER", "LD4", "LDD"]
    list_values = {metric: {cat: [] for cat in categories} for metric in metrics}
    best_mlna_k_per_alpha = {
        folder: {key: {'real_best_k': [], 'predicted_best_k':[], 'value':[], 'model':[]} for key in alphas} for folder in data_result_folder_name
    }
    ## fetch on alpha values
    for index, alpha in enumerate(alphas):
        ## fetch on name of folders
        for index2, result_folder in enumerate(data_result_folder_name):
            # load predicted best mlnk
            print(result_folder)
            name = \
                [file for _, _, files in os.walk(
                    f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage')
                 for file in files if
                 '_best_features' in file][0] if len([file for _, _, files in os.walk(
                    f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage')
                 for file in files if
                 '_best_features' in file]) != 0 else None
            best_mlna_k_per_alpha[result_folder][alpha]['accuracies'] = read_model(f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage/{name}')['model'] if name != None else None
            best_mlna_k_per_alpha[result_folder][alpha]['predicted_best_k'].append(read_model(f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage/{name}')["bestK"]) if name != None else None
            ## get classic model
            classic_f = [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir= f'{os.getcwd()}/{outputs_name}/{result_folder}/data_selection_storage',
                    func= lambda x :C_F(x) and ('v2_' in x),
                    verbose=False
                )
            ][-1]
            # define a list which will receive all possible value of accuracy for each layer in aims tp sorted it later and extra, the best best layer and model
            list_of_accuracy = []
            ## get model list on classic results
            models_list = classic_f.index.values.tolist()
            ## get model dictionary
            models = model_desc()
            ## save only the ones use during the classic learning
            models_name = {key: models[key] for key in models.keys() if key in models_list}
            print(models_name)
            ## identify the number of existing layer storage
            mlna_folders_names = [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}')][0]
            mlna_folders_names = sorted([int(el.split("_")[1]) for el in mlna_folders_names if "mlna" in el])
            print(result_folder, mlna_folders_names,"//", alpha, list(set(mlna_folders_names)&set(layers)))

            ## fetch on each mlna layer resultts
            for index3, layer in enumerate(list(set(mlna_folders_names)&set(layers))):

                ## get files results of alpha
                files = load_results(
                    f'{outputs_name}/{result_folder}',
                    _type,
                    layer,
                    alpha,
                    per=per,
                    glo=glo,
                    mix=gap
                )

                ## aggregate results belong to metrics specific to the current results
                for index4, model in enumerate(models_name):
                    # now fetch ours results to store
                    for metric in metrics: # accuracy and f1-score and/or financial cost
                        for approach in approachs: # Mlc or MCA
                            for logic in logics: # GLO, PER or GAP
                                for config in configs: # MX, CX, CY, CXY

                                    # print(result_folder, layer, approach, logic, config, len(files[approach][logic][config]) )
                                    for result in list(range(len(files[approach][logic][config]))): # each result file's containing evaluation metrics
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
                                                    (round(classic_f.loc[model, metric],4) if round(classic_f.loc[model, metric],4) > 0 else epsilon)
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
                                                    (round(classic_f.loc[model, metric],4) if round(classic_f.loc[model, metric],4) > 0 else epsilon)
                                                    ) * 100

                                        main_key = [ kjj for kjj in list(macro_metrics.keys()) if (logic in kjj) and (approach in kjj) and (config in kjj)]
                                        met = 'Acc' if 'acc' in metric else ('F1-score' if 'f1' in metric else 'Cost')
                                        macro_metrics[main_key[0]][met][result_folder].append(
                                            (
                                                valu1 if 'finan' not in metric else  (valu1 if valu1 ==0 else -1*valu1)
                                                , model)
                                            )

                                        # if 'LD4' in result_folder:
                                        #     print(result_folder, layer, config)
                                        if 'acc' in metric and 'MX' in config:
                                            # if 'LD4' in result_folder:
                                            #     print(result_folder, layer)
                                            list_of_accuracy.append((layer, model,files[approach][logic][config][result].loc[model, metric]))

            # analyse impact of layers and identify the best mlna as k
            if not(name is None) and 'MX' in configs:
                best_mlna_k_per_alpha[result_folder][alpha]['list']= list_of_accuracy
                list_of_accuracy = sorted(list_of_accuracy, key=lambda x: abs(x[2]), reverse=False) # best will be at position 0
                best_mlna_k_per_alpha[result_folder][alpha]['real_best_k'].append(list_of_accuracy[0][0])
                best_mlna_k_per_alpha[result_folder][alpha]['model'].append(list_of_accuracy[0][1])
                best_mlna_k_per_alpha[result_folder][alpha]['value'].append(list_of_accuracy[0][2])



"""
    Impact performance analytic for MLN1 and MLN2
    Q: How to select features used to build the graph?

    The goal here is to count the number of times where distingiush couple of performance (G,B) in MLN1 improve in MLN2
"""
def generate_g_b_impact_table(
    outputs_name=None, 
    cwd=None, 
    outputPath='result_lts',
    layers=[1], 
    _type='qualitative',
    alphas=[0.85]
):
    day = time.strftime("%Y_%m_%d_%H") # actual datetime
    data_result_folder_name = [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{outputs_name}')][0] # result folder name
    data_result_folder_name =  list(set(data_result_folder_name)-set(['LFD']))
    # define a structures
    metrics_impact = {
        'accuracy': {
            'MlC': {
                'GLO': [
                {
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [
                {
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            },
            'MCA': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            }
        },
        'f1-score': {
            'MlC': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            },
            'MCA': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            }
        },
        'precision1': {
            'MlC': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            },
            'MCA': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            }
        },
        'recall1': {
            'MlC': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            },
            'MCA': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            }
        },
        'f1-score1': {
            'MlC': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            },
            'MCA': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            }
        },
        'precision0': {
            'MlC': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            },
            'MCA': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            }
        },
        'recall0': {
            'MlC': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            },
            'MCA': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            }
        },
        'f1-score0': {
            'MlC': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            },
            'MCA': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            }
        },
        'financial-cost': {
            'MlC': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            },
            'MCA': {
                'GLO': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }],
                'PER': [{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            },{
                's': 0,
                'e':0,
                'i':0
            }]
            }
        }
    }

    # Loop on folder names
    for i, folder_name in enumerate(data_result_folder_name):
        # get classics values
        classic_f = [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{os.getcwd()}/{outputs_name}/{folder_name}/data_selection_storage',
                    func=lambda x :C_F(x) and ('v2_' in x),
                    verbose=False
                )
            ][-1]
        models_list = classic_f.index.values.tolist() # get list of used models
        cols = get_qualitative_from_cols(list(classic_f.columns)) # get resultat qualitative columns
        # Loop on alphas
        for alpha in alphas:    
            clusters = {key : deepcopy({
            'accuracy': {
                'MlC': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                },
                'MCA': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                }
            },
            'f1-score': {
                'MlC': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                },
                'MCA': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                }
            },
            'precision1': {
                'MlC': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                },
                'MCA': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                }
            },
            'recall1': {
                'MlC': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                },
                'MCA': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                }
            },
            'f1-score1': {
                'MlC': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                },
                'MCA': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                }
            },
            'precision0': {
                'MlC': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                },
                'MCA': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                }
            },
            'recall0': {
                'MlC': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                },
                'MCA': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                }
            },
            'f1-score0': {
                'MlC': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                },
                'MCA': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                }
            },
            'financial-cost': {
                'MlC': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                },
                'MCA': {
                    'GLO': {
                        'Good': [],
                        'Bad': []
                        },
                    'PER': {
                        'Good': [],
                        'Bad': []
                        },
                    'GAP': {
                        'Good': [],
                        'Bad': []
                        }
                }
            }
        }) for key in models_list} # structure to store good and bad features
            # Loop on MLN1 and MLN2
            for k in [1,2]:
                # loop on features combination
                for layer_config in get_combinations(range(len(cols)),k): # create subsets of k index of OHE and fetch it
                    col_targeted= [f'{cols[i]}' for i in layer_config]
                    case_k= '±'.join(col_targeted) if len(layer_config)>1 else col_targeted[0]
                    
                    ### get files for distincts logic
                    match= lambda x: (
                        sum(
                            [
                                re.sub(r'[^\w\s]', '', unidecode(partern)) in re.sub(r'[^\w\s]', '', unidecode(x))
                                for partern in case_k.split("±")
                                ]
                            ) == k if k > 1 else re.sub(r'[^\w\s]', '', unidecode(case_k)) in re.sub(r'[^\w\s]', '', unidecode(x))
                        )
                    
                    # Load belong files
                    files = load_results(
                        f'{outputs_name}/{folder_name}',
                        _type,
                        k,
                        alpha,
                        per=True,
                        glo=True,
                        mix=False,
                        isRand= True if k != 1 else False,
                        match=match
                    )

                    # Loop on approach
                    for approach in  ['MlC','MCA']:
                        # Loop on Logic
                        for logic in ['GLO','PER']: #,'PER'
                            # Loop on config
                            for config in ['MX']: # ['MX','CX']
                                # print(f"{folder_name}, {k}, {case_k}, {alpha}, {approach}, {logic}, {len(files[approach][logic][config])}")
                                # Loop on files inside
                                for file in range(len(files[approach][logic][config])):
                                    # Loop on models
                                    for model in models_list:
                                        # Loop on metrics
                                        for metric in metrics_impact.keys():
                                            if k == 1:
                                                status = 'Good' if round(files[approach][logic][config][file].loc[model,metric],4) > round(classic_f.loc[model,metric],4) else 'Bad'
                                                clusters[model][metric][approach][logic][status].append(case_k)
                                            else:
                                                indice = -1
                                                if sum([partern in clusters[model][metric][approach][logic]['Good'] for partern in case_k.split("±")]) == k:
                                                   indice = 0
                                                elif sum([partern in clusters[model][metric][approach][logic]['Bad'] for partern in case_k.split("±")]) == k:
                                                   indice = 2
                                                else:
                                                   indice = 1
                                                metrics_impact[metric][approach][logic][indice]["s"] += 1 if round(files[approach][logic][config][file].loc[model,metric],) < round(classic_f.loc[model,metric],) else 0
                                                metrics_impact[metric][approach][logic][indice]["e"] += 1 if round(files[approach][logic][config][file].loc[model,metric],) == round(classic_f.loc[model,metric],) else 0
                                                metrics_impact[metric][approach][logic][indice]["i"] += 1 if round(files[approach][logic][config][file].loc[model,metric],) > round(classic_f.loc[model,metric],) else 0
    # call the printer function
    print_g_b_impact_table(
        metrics_impact,
        f'{cwd}/{outputPath}',
        'all',
        True
        )
    return metrics_impact



"""
    Impact performance analytic for GLO, PER
"""
# --------------------------------------------------------------------------------------BarPlots-----------------------------------------------------------------

# ################################################################
# ############## BarPLots Printing Functions #####################
# ################################################################
def print_compare_bchart(
    store,
    output_path,
    outputs_name,
    alpha,
    top
):
    # fetch model name
    models = list(store.keys())
    folders = list(store[models[0]].keys())
    tables = {model:'' for model in models}

    summary = {model:{folder: {'GLO_CX':0,'PER_CX':0,'PER_CY':0} for folder in folders} for model in models}

    lambda_function = lambda dictionary, pattern: sum(1 for key in dictionary.keys() if re.match(pattern, key))
    counter1 = {'GLO_CX':0,'PER_CX':0,'PER_CY':0}
    top_2 = []
    for model in list(store.keys()):
        """
        \\begin{bchart}[min=50,max=100,step=10,unit=\\%]
        \\bcbar[label=1st bar,color=yellow]{-3.4}
            \\smallskip
        \\bcbar{5.6} 
            \\medskip
        \bcbar{7.2}
            \\bigskip
        \\bcbar{9.9}
    \\end{bchart}
        \\begin{tabular}{
        \\end{tabular}
        """

        # fetch folders
        tab = "\\begin{tabular}{"
        for _ in store[model].keys():
            tab+="c"
        tab+="""}
        """
        tables[model]+= tab
        for i, folder in enumerate(store[model].keys()):
            classic_f = [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{os.getcwd()}/{outputs_name}/{folder}/data_selection_storage',
                    func=lambda x :C_F(x) and ('v2_' in x),
                    verbose=False
                )
            ][-1]
            columns = len(list(classic_f.columns))-11 # 11 METRICS
            # sort the dict
            data = dict(sorted(store[model][folder].items(), key=lambda x: abs(x[1]), reverse=True)[:top])
            top_2.extend(list(data.keys())[:5])
            summary[model][folder]['GLO_CX'] = lambda_function(data, r'^.*_GLO$')
            # summary[model][folder]['PER'] = lambda_function(data, r'^.*_PER$')
            summary[model][folder]['PER_CX'] = lambda_function(data, r'^(?!Y).*_PER$')
            summary[model][folder]['PER_CY'] = lambda_function(data, r'^Y.*_PER$')
            summary[model][folder]['nbAtt'] = columns
            counter1['GLO_CX'] += 1 if summary[model][folder]['GLO_CX'] >= 4 else 0
            counter1['PER_CX'] += 1 if summary[model][folder]['PER_CX'] >= 4 else 0
            counter1['PER_CY'] += 1 if summary[model][folder]['PER_CY'] >= 4 else 0
            # identify a max and min importance value
            # print(data.items())
            data1 = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
            max_val = list(data1.items())[0][1]
            min_val = list(data1.items())[-1][1]
            # create a new char for each folder
            bchart = "\\begin{bchart}[max="+f"{(max_val if abs(min_val) < max_val else abs(min_val)):.{6}f}"+""",unit=\\%]
                """

            # use to Dec order dict to fill the bar chart
            for key, val in data.items():
                # identify the logic
                color = "green" if GLO_F(key) else ("yellow" if DEGREE_F(key) else ("red" if PER_F(key) else "blue"))
                bchart += "\\bcbar[value=, label="+str(key).replace('_','\\_')+",color="+color+"]{"+f"{val:.{6}f}"+"""}
                    \\smallskip
                    """
            bchart += ("\\bcxlabel{"+str(folder).replace('_','\\_')+"""}
                            \\end{bchart}
                            """) if i != len(store[model].keys())-1 else ("\\bcxlabel{"+str(folder).replace('_','\\_')+"""}
                            \\bclegend{5pt}{blue/Classic, green/GLO, red/PER, yellow/DEGREE}
                                \\end{bchart}
                    """)

            tables[model]+= (bchart +"""
                            &
                            """) if i != len(store[model].keys())-1 else (bchart +"""
                \\end{tabular}}""")
        
        create_domain(f"{output_path}/bchart/{str(alpha).replace('.','_')}/{model}")
        filename1 = f"{output_path}/bchart/{str(alpha).replace('.','_')}/{model}/{model}.tex"
        _file = open(filename1, "w")
        _file.write(bchar_header+tables[model]+footer)
        _file.close()
    # ----------------------------- print the summary ---------------------------- 
    table_header = """
    %\\begin{sidewaystable}
    \\resizebox{\\textwidth}{!}{

    \\begin{tabular}{|c|c|"""

    # setup information columns headears
    nbMCol =  7
    table_header+= "r|"*nbMCol
    # add col for total results
    table_header+= "} "
    # add separator clines
    nb_cols = (2+nbMCol)
    table_header+= " \\cline{1-"+str(nb_cols)+"}" # corresponding to the number of columns


    # build the first line: metrics' line
    lines = ''
    # add the blank block
    lines += """
    \\multicolumn{2}{|c|}{}"""
    # add cols for metric
    for model in models:
        lines+= " & \\textbf{"+model+"}"
    # add the total name
    lines+= " & \\textbf{TOTAL} \\\\ "
    lines+= " \\cline{1-"+str(nb_cols)+"""}
    """

    # lambda function for sum
    sum_type = lambda store, folder, type: sum([store[model][folder][type] for model in store.keys()])
    max_sum_type = lambda store, folder, types: max([sum([store[model][folder][type] for model in store.keys()]) for type in types])
    sum_mod = lambda store, model, types: sum([store[model][folder][type] for folder in store[model].keys() for type in types])
    max_sum_mod = lambda store, types: max([sum([store[model][folder][type]  for folder in store[model].keys() for type in types]) for model in store.keys()])
    # fetch folders
    for folder in folders:
        lines+= """
        \\multirow{3}{*}{\\textbf{"""+folder+""" ("""+ str(summary[models[0]][folder]['nbAtt'])+""" + 19)}}
        """
        # fetch descriptors type
        for di, desc in enumerate(counter1.keys()):
            lines+= """& """+ str(desc).replace('_','\\_')
            # fetch models
            for model in models:
                # add desc info for each model
                lines+= """& """+ str(summary[model][folder][desc])
            # add total of the current
            lines+= """& """+ str(sum_type(summary, folder, desc)) if max_sum_type(summary, folder, counter1.keys()) != sum_type(summary, folder, desc) else "& \\textbf{"+str(sum_type(summary, folder, desc))+"}"
            # back to next line
            lines+= " \\\\ "
            lines+= " \\cline{1-"+str(nb_cols)+"""}
            """ if di == len(counter1.keys())-1 else " \\cline{2-"+str(nb_cols)+"""}
            """
    # add total line
    lines += """
    \\multicolumn{2}{|c|}{\\textbf{TOTAL}}"""
    for model in models:
        lines+= " & "+ str(sum_mod(summary, model, counter1.keys())) if sum_mod(summary, model, counter1.keys()) != max_sum_mod(summary, counter1.keys()) else "& \\textbf{"+ str(sum_mod(summary, model, counter1.keys()))+"}"
    lines+= " & "
    lines+= " \\\\ "
    lines+= " \\cline{1-"+str(nb_cols)+"""}
    \\end{tabular}}"""

    create_domain(f"{output_path}/bchart/{str(alpha).replace('.','_')}/summary")

    with open(f"{output_path}/bchart/{str(alpha).replace('.','_')}/summary/summary.tex", "w") as fichier:
        fichier.write(
            header+"""
            \\begin{table}[H]
            \\center"""+table_header+lines+"""
            \\caption{Nombre d'attributs de type GLO, PER et PER\\_Y des graphes multicouches qui sont présents dans le Top-20 des attributs qui contribuent le plus aux décisions des modèles de prédiction dans les différents jeux de données. Entre parenthèse, c'est le nombre total d'attributs du jeu de données associé.}
            \\label{tab:recapTab}
            \\end{table}"""+footer
            )
    # ------------------------------------------------------------------------------------
    return (summary,"/////////",counter1, "/////////",dict(Counter(top_2)))

"""
    pour chaque jeu de donneer, presenter l'impact du nombre de couches suivant notre protocole sur les metrics
"""
def print_compare_k_att_selection_v2_2(
    store,
    output_path,
    metrics,
    valll=False
):

    # define the stat function
    act = max if valll is False else (min if "financial-cost" in metric else max)
    # define the tabular of the metric
    gplot = """\\resizebox{\\textwidth}{!}{
    \\begin{tabular}{"""+("c"*(2))+"""}
    """
    # fetch dataset
    for index, dataset in enumerate(set(store.keys())-set(['LFD'])):
        # define lines plots form
        lines = """\\addplot table[x=k,y=Acc,row sep=crcr] {
            k Acc \\\\
        """

        plot = """
            \\begin{tikzpicture}
            \\begin{axis}[
                xlabel=k,
                ylabel={Acc},
                xtick=data, % Get x tick marks from data
                x tick label style={% Change x tick label style
                    /pgf/number format/set thousands separator={}%
                },
                y tick label style={% Change y tick label style
                    /pgf/number format/fixed, % Use fixed-point notation
                    /pgf/number format/precision=3, % Precision to 3 decimal places
                    %/pgf/number format/fixed zerofill % Fill with zeros if needed
                },
                title={"""+dataset+""" },% Plot title
                tick align=outside, % Ticks on the outside
                enlargelimits = upper,
                xmin=0, % Ensure x-axis starts at 0
                %ymin=0, % Ensure y-axis starts at 0
                legend pos = outer north east
            ]
        """
        # fetch alpha
        for alpha in store[dataset].keys():
            if ('list' in store[dataset][alpha]):
                # print(store[dataset])
                sorted_data = sorted(store[dataset][alpha]['list'], key=lambda x: (x[0], -x[2]))
        
                best_tuples = {}
                for nb_couches, _, precision in sorted_data:
                    if nb_couches not in best_tuples:
                        best_tuples[nb_couches] = precision
                
                result = [(nb_couches, precision) for nb_couches, precision in best_tuples.items()]
                
                NbCouches_list = [nb_couches for nb_couches, _ in result]
                precision_list = [precision for _, precision in result]
                # Chart plotting
                print(NbCouches_list, precision_list, dataset, alpha)


                # add line container
                plot +=lines
                # fetch mln k
                for i in range(len(NbCouches_list)):
                    # add line content
                    plot += f"""{NbCouches_list[i]} {precision_list[i]} \\\\
                    """
                # end the model line
                plot+= """};
                    \\addlegendentry{$\\alpha = """+str(alpha)+"""$}
                """   
                # if ((index+1)%2 == 0) else """};
                # """ 
        # close the folder plot 
        gplot+=(plot+"""
            \\end{axis}
            \\end{tikzpicture}
            &
                """) if ((index)%2 == 0) else (plot+"""
            \\end{axis}
            \\end{tikzpicture}
            \\\\
                """) 
    # close the metric
    gplot+="""\\end{tabular}}
            """
    create_domain(f"{output_path}/alpha/mlnkGrowth")
    filename1 = f"{output_path}/alpha/mlnkGrowth/mlnkGrowth.tex"
    _file = open(filename1, "w")
    _file.write(plot_header+gplot+"}"+footer)
    _file.close()




# ################################################################
# ############## BarPlots Generation Logic Functions #############
# ################################################################
def generate_descriptor_ranking(
    outputs_name=None, 
    cwd=None, 
    outputPath='result_lts',
    layers=[1], 
    _type='qualitative',
    alphas=[0.85]
):
    """
    Analyzer Descriptors' importance
    Parameters
    ----------
    outputs_path
    cwd
    layers
    type

    Returns
    -------

    """
    day = time.strftime("%Y_%m_%d_%H")
    ## identify the name of results folder
    data_result_folder_name = [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{outputs_name}')][0]
    print(data_result_folder_name)
    data_result_folder_name = list(set(data_result_folder_name)-set(['LFD']))
    ## init a global container for containing the results of all results folder

    
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

        'INTER_GLO_': [],
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

        'YN_COMBINE_PER':[],
        'YP_COMBINE_PER':[],
        'YN_INTER_PER':[],
        'YP_INTER_PER':[],
        'YN_INTRA_PER':[],
        'YP_INTRA_PER':[],
        'DEGREE': []

    }
    ######################################
    global_details_metrics_depth_1 = {
    }

    global_tables = ""
    local_tables = {}
    ## fetch on alpha values
    for index, alpha in enumerate(alphas):
        ## fetch on name of folders
        for index2, result_folder in enumerate(data_result_folder_name):
            ## get classic model
            classic_f = [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{os.getcwd()}/{outputs_name}/{result_folder}/data_selection_storage',
                    func=lambda x :C_F(x) and ('v2_' in x),
                    verbose=False
                )
            ][-1]
            # define a list which will receive all possible value of accuracy for each layer in aims tp sorted it later and extra, the best best layer and model
            # list_of_accuracy = []
            ## get model list on classic results
            models_list = classic_f.index.values.tolist()
            ## get model dictionary
            models = model_desc()
            ## save only the ones use suring the classic learning
            models_name = {key: models[key] for key in models.keys() if key in models_list}
            print(models_name)
            ## identify the number of existing layer storage
            mlna_folders_names = [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}')][0]
            mlna_folders_names = sorted([int(el.split("_")[1]) for el in mlna_folders_names if "mlna" in el])
            print(result_folder, mlna_folders_names,"//", alpha, list(set(mlna_folders_names)&set(layers)))

            ## fetch on each mlna layer resultts
            for index3, layer in enumerate(list(set(mlna_folders_names)&set(layers))):
                ## get files results of alpha
                files = load_results(
                    f'{outputs_name}/{result_folder}',
                    _type,
                    layer,
                    alpha,
                    per=True,
                    glo=True,
                    mix=True,
                    match= lambda x: ('v2_' in x) 
                )
                ## aggregate results belong to metrics specific to the current results
                for index4, model in enumerate(models_name):
                    if (index2 == index3) and (index2 == 0) and (index == 0):
                        # global one
                        global_details_metrics_depth_1[model] = {key: deepcopy(template_descripteurs) for key in data_result_folder_name}

                    
                    for result in list(range(len(files["MlC"]["GAP"]["CXY"]))): # each result file's containing evaluation metrics
                        colo = files["MlC"]["GAP"]["CXY"][result].columns
                        for att in colo:
                            if not (att in ["accuracy", "precision", "recall", "f1-score","financial-cost", "precision1", "recall1", "f1-score1", "precision0", "recall0", "f1-score0"]):
                                if YN_PER_INTER_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'YN_INTER_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif YP_PER_INTER_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'YP_INTER_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif YN_PER_INTRA_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'YN_INTRA_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif YP_PER_INTRA_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'YP_INTRA_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif YN_PER_COMBINE_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'YN_COMBINE_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif YP_PER_COMBINE_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'YP_COMBINE_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif GLO_INTER_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTER_GLO'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif GLO_INTRA_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTRA_GLO'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif GLO_COMBINE_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'COMBINE_GLO'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif PER_INTER_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTER_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif PER_INTRA_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTRA_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif PER_COMBINE_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'COMBINE_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif GLO_INTER_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTER_M_GLO'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif GLO_INTRA_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTRA_M_GLO'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif GLO_COMBINE_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'COMBINE_M_GLO'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif PER_INTER_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTER_M_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif PER_INTRA_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTRA_M_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif PER_COMBINE_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'COMBINE_M_PER'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                elif DEGREE_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'DEGREE'].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))
                                else:
                                    if not (att in global_details_metrics_depth_1[model][result_folder].keys()):
                                        global_details_metrics_depth_1[model][result_folder][
                                            att] = []
                                        # print(global_details_metrics_depth_1[model][result_folder])
                                    global_details_metrics_depth_1[model][result_folder][
                                        att].append(abs(files["MlC"]["GAP"]["CXY"][result].loc[model, att]))


    for model in global_details_metrics_depth_1.keys():
        for folder in global_details_metrics_depth_1[model].keys():
            for att in global_details_metrics_depth_1[model][folder].keys():
                print(global_details_metrics_depth_1[model][folder][att], model, folder, att)
                global_details_metrics_depth_1[model][folder][att] = statistics.mean(global_details_metrics_depth_1[model][folder][att])
            # Calculer la somme de toutes les valeurs
            total_sum = sum(global_details_metrics_depth_1[model][folder].values())

            # Normaliser les valeurs
            global_details_metrics_depth_1[model][folder] = {k: v / total_sum for k, v in global_details_metrics_depth_1[model][folder].items()}
    ## print the global container (gain, metric)
    return print_compare_bchart(
        global_details_metrics_depth_1,
        f'{cwd}/{outputPath}/',
        outputs_name,
        "all",
        20
    )
































