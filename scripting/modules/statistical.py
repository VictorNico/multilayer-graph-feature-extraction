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

################################################
############ Module loading
import statistics
import os
from .pipeline import *
from unidecode import unidecode
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
import re
import time
from copy import deepcopy
import tabulate

########### End Module
################################################

################################################
######## Constant declaration

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
layers = [1]
approach = ['MlC','MCA']

style = """<style> 
    table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
    } 
    .wrap-text {
    word-wrap: break-word;
    } 
    .wrap-text {
    overflow-wrap: break-word;
    } 
    .limited-column {
    width: 100px;
    } 
    .dashed-border {
    border: 1px dashed black;
    }
    .dotted-border {
    border: 1px dotted black;
    } 
    td {
    text-align: center;
    } 
    caption {
    margin:0; 
    margin-bottom: 2px; 
    text-align: start; 
    border: 1px dashed black;
    } 
    caption > h2 {
    text-align: center;
    }
    </style>"""

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

############# End declaration
########################################################
# --------------------------------------------------------------------------------------MixingUtilsFunction---------------------------------------------------------

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
            if func(filename) and not ('x_' in filename or 'y_' in filename or 'metric' in filename):
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
    var.split("__")[1] for var in  x if ("__" in var)
    
]
)))

"""
    Lambda Function
    Get all quantitative features from operation results
    During the process, we specify the partern '___' like an indicator
"""
get_quantitative_from_cols = lambda x: (list(set([
    var.split("___")[1] for var in x if ("___" in var)
    ]
)))
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
        match= lambda x: True

):
    """
    Load results from a results folder
    Parameters
    ----------
    outputs_path: path of results folder
    type: type of results
    k: layer
    per: flag to indicate whether to load per-layer results
    glo: flag to indicate whether to load glo results
    mix: flag to indicate whether to load mix results
    isRand: flag to indicate if it's a random combinaison layer
    match: lambda func to a specific kind of file



    Returns
    -------

    """
    files = {
        'MlC':{
            'GLO':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/global/data_selection_storage',
                        func=lambda x: (MlC_F(x) and GLO_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/global/data_selection_storage',
                        func=lambda x: (MlC_F(x) and GLO_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            },
            'PER':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/personalized/data_selection_storage',
                        func=lambda x: (MlC_F(x) and PER_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/personalized/data_selection_storage',
                        func=lambda x: (MlC_F(x) and PER_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/personalized/data_selection_storage',
                        func=lambda x: (MlC_F(x) and PER_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/personalized/data_selection_storage',
                        func=lambda x: (MlC_F(x) and PER_CXY_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            },
            'GAP':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/mixed/data_selection_storage',
                        func=lambda x: (MlC_F(x) and GAP_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/mixed/data_selection_storage',
                        func=lambda x: (MlC_F(x) and GAP_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/mixed/data_selection_storage',
                        func=lambda x: (MlC_F(x) and GAP_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/mixed/data_selection_storage',
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
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/global/data_selection_storage',
                        func=lambda x: (MCA_F(x) and GLO_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/global/data_selection_storage',
                        func=lambda x: (MCA_F(x) and GLO_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            } if glo is True else None,
            'PER':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/personalized/data_selection_storage',
                        func=lambda x: (MCA_F(x) and PER_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/personalized/data_selection_storage',
                        func=lambda x: (MCA_F(x) and PER_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/personalized/data_selection_storage',
                        func=lambda x: (MCA_F(x) and PER_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/personalized/data_selection_storage',
                        func=lambda x: (MCA_F(x) and PER_CXY_F(x) and (match(x))),
                        verbose=False
                    )
                ]
            } if per is True else None,
            'GAP':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/mixed/data_selection_storage',
                        func=lambda x: (MCA_F(x) and GAP_MX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/mixed/data_selection_storage',
                        func=lambda x: (MCA_F(x) and GAP_CX_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/mixed/data_selection_storage',
                        func=lambda x: (MCA_F(x) and GAP_CY_F(x) and (match(x))),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else("mlna_" + str(k) if isRand == True else "mlna_" + str(k) + "_b")}/mixed/data_selection_storage',
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
def print_compare(
    store,
    output_path,
    alpha,
    valll=False
):
    # fetch model name
    tables = {model:'' for model in list(store.keys())}
    for model in list(store.keys()):
        # add the resize box to ensure the scale of the table will be contain's inside the width space avalable.
        # start setting up the tabular dimensions setting
        table_header = """
        \\resizebox{\\textwidth}{!}{

        \\begin{tabular}{|c|c|c|"""
        # identify folder with none financial details
        none_financial_folder = [folder for folder in list(store[model].keys()) if sum(['finan' in mec for mec in list(store[model][folder].keys())]) != 1]
        financial_folder = [folder for folder in list(store[model].keys()) if sum(['finan' in mec for mec in list(store[model][folder].keys())]) == 1]
        # print(none_financial_folder, financial_folder)
        # add cols for financial folders
        for folder in financial_folder:
            table_header+= "r|r|r|"
        # add cols for non financial folder
        for folder in financial_folder:
            table_header+= "r|r|"
        # add col for total results
        table_header+= "r|} "
        # add separator clines
        nb_cols = (3+(3*len(financial_folder))+(2*len(none_financial_folder))+1)
        table_header+= " \\cline{1-"+str(nb_cols)+"}" # corresponding to the number of columns


        # build the first line
        lines = ''
        # add the model block
        lines += """
        
        \\multicolumn{3}{|c|}{"""+model+"("+str(alpha).replace('.','_')+""")}"""
        # add cols for financial folders
        for folder in financial_folder:
            lines+= "& \\multicolumn{3}{|c|}{"+folder+"}"
        # add cols for non financial folder
        for folder in none_financial_folder:
            lines+=  "& \\multicolumn{2}{|c|}{"+folder+"}"
        # add the total name
        lines+= " & \\multirow{2}{*}{Total} \\\\ "
        lines+= " \\cline{4-"+str(nb_cols-1)+"}"

        # build metrics' lines
        lines+= """
         \\multicolumn{3}{|c|}{}

        """
        # add metrics for financial folders
        for folder in financial_folder:
            lines+= "& Acc & F1 & Cost"
        # add metrics' for non financial folder
        for folder in financial_folder:
            lines+=  "& Acc & F1"
        # add the total name
        lines+= " & \\\\ "
        lines+= " \\cline{1-"+str(nb_cols)+"}"

        # add classic metrics
        lines+="""
        \\multicolumn{3}{|c|}{Classic} 

        """
        for folder in financial_folder:
            for metric in ['accuracy','f1-score','financial-cost']:
                # print(model, folder, metric)
                lines+= "& "+str(round(store[model][folder]['classic'].loc[model, metric],4))+""
        # add cols for non financial folder
        for folder in none_financial_folder:
            for metric in ['accuracy','f1-score']:
                lines+= "& "+str(round(store[model][folder]['classic'].loc[model, metric],4))+""
        # add an empty cell for Total
        lines+= "& \\\\ \\cline{1-"+str(nb_cols)+"""}

        """

        is_best = lambda mag,current,act: (act([act(mag[el]) for el in list(mag.keys())]) == current) and (current != 0)
        # fetch store to fullfil the tables with content
        for ai, approach in enumerate(list(store[model][financial_folder[0]]['accuracy'].keys())): # Mlc or MCA
            lines+= """
            \\multirow{10}{*}{"""+approach+"""}&

            """
            for li, logic in enumerate(list(store[model][financial_folder[0]]['accuracy'][approach].keys())): # GLO, PER or GAP
                total_counter = {}
                lines+= f"{'&'*(li != 0)}"+"""
                    \\multirow{"""+str(len(list(store[model][financial_folder[0]]['accuracy'][approach][logic].keys())))+"""}{*}{"""+logic+"""}

                    """
                for ci,config in enumerate(list(store[model][financial_folder[0]]['accuracy'][approach][logic].keys())): # MX, CX, CY, CXY 
                    lines+= f"{'&'*(ci != 0)}  & {config}"
                    for fi, folder in enumerate(list(store[model].keys())):
                        if (ci == 0) and (fi == 0):
                            total_counter = {key:[] for key in list(store[model][folder]['accuracy'][approach][logic].keys())}
                        metrics = (['accuracy','f1-score','financial-cost'] if (folder in financial_folder) else ['accuracy','f1-score'])
                        # metrics = list(set(metrics)-set(['classic']))
                        # print(folder, metrics, config, approach, logic)
                        for metric in metrics:
                            act = max if valll is False else (min if "financial-cost" in metric else max) 
                            print(metric, act, valll)
                            is_sup = is_best(store[model][folder][metric][approach][logic], act(store[model][folder][metric][approach][logic][config]),act)
                            val = "\\textbf{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if is_sup else str(act(store[model][folder][metric][approach][logic][config]))
                            lines+= f"& {val}"
                            total_counter[config].append(is_sup)
                    start = (3 if ci != len(list(store[model][financial_folder[0]]['accuracy'][approach][logic].keys()))-1 else (1 if li == len(list(store[model][financial_folder[0]]['accuracy'][approach].keys()))-1 else 2))
                    # print(start)
                    lines+= f"""& {sum(total_counter[config])} \\\\ """+ """ \\cline{"""+str(start)+"""-"""+str(nb_cols)+"""}

                    
                    """

        lines+= """

        \\end{tabular}
        }"""

        table = table_header + lines
        tables[model] = table
        create_domain(f"{output_path}/tableaux/{str(alpha).replace('.','_')}")
        filename1 = f"{output_path}/tableaux/{str(alpha).replace('.','_')}/tab_alpha_{str(alpha).replace('.','_')}_{model}.tex"
        _file = open(filename1, "w")
        _file.write(table)
        _file.close()
    return tables


def print_compare_v2(
    store,
    output_path,
    alpha,
    valll=False
):
    # fetch model name
    tables = {model:'' for model in list(store.keys())}
    for model in list(store.keys()):
        # add the resize box to ensure the scale of the table will be contain's inside the width space avalable.
        # start setting up the tabular dimensions setting
        table_header = """
        \\resizebox{\\textwidth}{!}{

        \\begin{tabular}{|c|c|c|"""
        # identify folder with none financial details
        none_financial_folder = [folder for folder in list(store[model].keys()) if sum(['finan' in mec for mec in list(store[model][folder].keys())]) != 1]
        financial_folder = [folder for folder in list(store[model].keys()) if sum(['finan' in mec for mec in list(store[model][folder].keys())]) == 1]
        print(none_financial_folder, financial_folder)
        # add cols for financial folders
        for _ in financial_folder:
            table_header+= "r|r|r|"
        # add cols for non financial folder
        for _ in none_financial_folder:
            print(1)
            table_header+= "r|r|"
        # add col for total results
        table_header+= "} "
        # add separator clines
        nb_cols = (3+(3*len(financial_folder))+(2*len(none_financial_folder)))
        table_header+= " \\cline{1-"+str(nb_cols)+"}" # corresponding to the number of columns


        # build the first line
        lines = ''
        # add the model block
        lines += """
        \\multicolumn{3}{|c|}{"""+model+"("+str(alpha).replace('.','_')+""")}"""
        # add cols for financial folders
        for folder in financial_folder:
            lines+= "& \\multicolumn{3}{|c|}{"+folder.replace('_','\\_')+"}"
        # add cols for non financial folder
        for folder in none_financial_folder:
            print(1)
            lines+=  "& \\multicolumn{2}{|c|}{"+folder+"}"
        # add the total name
        lines+= "\\\\ "
        lines+= " \\cline{4-"+str(nb_cols)+"}"

        # build metrics' lines
        lines+= """
         \\multicolumn{3}{|c|}{}"""
        # add metrics for financial folders
        for folder in financial_folder:
            lines+= "& Acc & F1 & Cost"
        # add metrics' for non financial folder
        for folder in none_financial_folder:
            print(1)
            lines+=  "& Acc & F1"
        # add the total name
        lines+= " \\\\ "
        lines+= " \\cline{1-"+str(nb_cols)+"}"

        # add classic metrics
        lines+="""
        \\multicolumn{3}{|c|}{Classic}"""
        for folder in financial_folder:
            for metric in ['accuracy','f1-score','financial-cost']:
                # print(model, folder, metric)
                lines+= "& "+str(round(store[model][folder]['classic'].loc[model, metric],4))+""
        # add cols for non financial folder
        for folder in none_financial_folder:
            print(1)
            for metric in ['accuracy','f1-score']:
                lines+= "& "+str(round(store[model][folder]['classic'].loc[model, metric],4))+""
        # add an empty cell for Total
        lines+= "\\\\ \\cline{1-"+str(nb_cols)+"""}

        """

        is_best = lambda mag,act: (set([act(mag[el][al][il]) for el in list(mag.keys()) for al in list(mag[el].keys()) for il in list(mag[el][al].keys())]))
        # fetch store to fullfil the tables with content
        for ai, approach in enumerate(list(store[model][financial_folder[0]]['accuracy'].keys())): # Mlc or MCA
            lines+= """
            \\multirow{10}{*}{"""+approach+"""}&"""
            for li, logic in enumerate(list(store[model][financial_folder[0]]['accuracy'][approach].keys())): # GLO, PER or GAP
                total_counter = {}
                lines+= f"{'&'*int(li != 0)}"+"""\\multirow{"""+str(len(list(store[model][financial_folder[0]]['accuracy'][approach][logic].keys())))+"""}{*}{"""+logic+"""}"""
                for ci,config in enumerate(list(store[model][financial_folder[0]]['accuracy'][approach][logic].keys())): # MX, CX, CY, CXY 
                    lines+= f"{'&'*int(ci != 0)}  & {config}"
                    for fi, folder in enumerate(list(store[model].keys())):
                        if (ci == 0) and (fi == 0):
                            total_counter = {key:[] for key in list(store[model][folder]['accuracy'][approach][logic].keys())}
                        metrics = (['accuracy','f1-score','financial-cost'] if (folder in financial_folder) else ['accuracy','f1-score'])
                        # metrics = list(set(metrics)-set(['classic']))
                        # print(folder, metrics, config, approach, logic)
                        for metric in metrics:
                            act = max if valll is False else (min if "financial-cost" in metric else max) 
                            # print(metric, act, valll)
                            is_sup = sorted(list(is_best(store[model][folder][metric],act)), reverse= (False if ((valll == True) and ("financial-cost" in metric )) else True))
                            # print(metric,(False if ((valll == True) and ("financial-cost" in metric )) else True),act(store[model][folder][metric][approach][logic][config]), valll, is_sup)
                            pos = is_sup.index(act(store[model][folder][metric][approach][logic][config]))

                            val = "\\textbf{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 0) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else (
                                "\\underline{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 1) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else (
                                    "\\textit{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 2) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else 
                                    str(act(store[model][folder][metric][approach][logic][config]))
                                    )
                                )

                            lines+= f"& {val}"
                            total_counter[config].append(is_sup)
                    start = (3 if ci != len(list(store[model][financial_folder[0]]['accuracy'][approach][logic].keys()))-1 else (1 if li == len(list(store[model][financial_folder[0]]['accuracy'][approach].keys()))-1 else 2))
                    # print(start)
                    lines+= f"""\\\\ """+ """ \\cline{"""+str(start)+"""-"""+str(nb_cols)+"""}

                    
                    """

        lines+= """

        \\end{tabular}
        }"""

        table = table_header + lines
        tables[model] = table
        create_domain(f"{output_path}/tableaux/{str(alpha).replace('.','_')}")
        filename1 = f"{output_path}/tableaux/{str(alpha).replace('.','_')}/tab_alpha_{str(alpha).replace('.','_')}_{model}.tex"
        _file = open(filename1, "w")
        _file.write(table)
        _file.close()
    return tables


def print_compare_v3(
    store,
    output_path,
    alpha,
    valll=False
):
    # fetch model name
    tables = {model:'' for model in list(store.keys())}
    for model in list(store.keys()):
        # add the resize box to ensure the scale of the table will be contain's inside the width space avalable.
        # start setting up the tabular dimensions setting
        table_header = """
        %\\begin{sidewaystable}
        \\resizebox{\\textwidth}{!}{

        \\begin{tabular}{|c|c|c|"""
        # identify folder with none financial details
        none_financial_folder = [folder for folder in list(store[model].keys()) if sum(['finan' in mec for mec in list(store[model][folder].keys())]) != 1]
        financial_folder = [folder for folder in list(store[model].keys()) if sum(['finan' in mec for mec in list(store[model][folder].keys())]) == 1]
        print(none_financial_folder, financial_folder)
        # add cols for financial folders
        for _ in financial_folder:
            table_header+= "r|r|r|r|r|r|r|r|r|"
        # add cols for non financial folder
        for _ in none_financial_folder:
            print(1)
            table_header+= "r|r|r|r|r|r|r|r|"
        # add col for total results
        table_header+= "} "
        # add separator clines
        nb_cols = (3+(9*len(financial_folder))+(8*len(none_financial_folder)))
        table_header+= " \\cline{1-"+str(nb_cols)+"}" # corresponding to the number of columns


        # build the first line
        lines = ''
        # add the model block
        lines += """
        \\multicolumn{3}{|c|}{"""+model+"("+str(alpha).replace('.','_')+""")}"""
        # add cols for financial folders
        for folder in financial_folder:
            lines+= "& \\multicolumn{9}{|c|}{"+folder.replace('_','\\_')+"}"
        # add cols for non financial folder
        for folder in none_financial_folder:
            print(1)
            lines+=  "& \\multicolumn{8}{|c|}{"+folder+"}"
        # add the total name
        lines+= "\\\\ "
        lines+= " \\cline{4-"+str(nb_cols)+"}"

        # build metrics' lines
        lines+= """
         \\multicolumn{3}{|c|}{}"""
        # add metrics for financial folders
        for folder in financial_folder:
            lines+= "& Acc & F1 & P1 & R1 & F11 & P0 & R0 & F10 & Cost"
        # add metrics' for non financial folder
        for folder in none_financial_folder:
            print(1)
            lines+=  "& Acc & F1 & P1 & R1 & F11 & P0 & R0 & F10 "
        # add the total name
        lines+= " \\\\ "
        lines+= " \\cline{1-"+str(nb_cols)+"}"

        # add classic metrics
        lines+="""
        \\multicolumn{3}{|c|}{Classic}"""
        for folder in financial_folder:
            for metric in ['accuracy', 'f1-score', 'precision1', 'recall1', 'f1-score1', 'precision0', 'recall0', 'f1-score0', 'financial-cost']:
                # print(model, folder, metric)
                lines+= "& "+str(round(store[model][folder]['classic'].loc[model, metric],4))+""
        # add cols for non financial folder
        for folder in none_financial_folder:
            print(1)
            for metric in ['accuracy', 'f1-score', 'precision1', 'recall1', 'f1-score1', 'precision0', 'recall0', 'f1-score0']:
                lines+= "& "+str(round(store[model][folder]['classic'].loc[model, metric],4))+""
        # add an empty cell for Total
        lines+= "\\\\ \\cline{1-"+str(nb_cols)+"""}

        """

        is_best = lambda mag,act: (set([act(mag[el][al][il]) for el in list(mag.keys()) for al in list(mag[el].keys()) for il in list(mag[el][al].keys())]))
        # fetch store to fullfil the tables with content
        for ai, approach in enumerate(list(store[model][financial_folder[0]]['accuracy'].keys())): # Mlc or MCA
            lines+= """
            \\multirow{10}{*}{"""+approach+"""}&"""
            for li, logic in enumerate(list(store[model][financial_folder[0]]['accuracy'][approach].keys())): # GLO, PER or GAP
                total_counter = {}
                lines+= f"{'&'*int(li != 0)}"+"""\\multirow{"""+str(len(list(store[model][financial_folder[0]]['accuracy'][approach][logic].keys())))+"""}{*}{"""+logic+"""}"""
                for ci,config in enumerate(list(store[model][financial_folder[0]]['accuracy'][approach][logic].keys())): # MX, CX, CY, CXY 
                    lines+= f"{'&'*int(ci != 0)}  & {config}"
                    for fi, folder in enumerate(list(store[model].keys())):
                        if (ci == 0) and (fi == 0):
                            total_counter = {key:[] for key in list(store[model][folder]['accuracy'][approach][logic].keys())}
                        metrics = (['accuracy', 'f1-score', 'precision1', 'recall1', 'f1-score1', 'precision0', 'recall0', 'f1-score0', 'financial-cost'] if (folder in financial_folder) else ['accuracy', 'f1-score', 'precision1', 'recall1', 'f1-score1', 'precision0', 'recall0', 'f1-score0'])
                        # metrics = list(set(metrics)-set(['classic']))
                        # print(folder, metrics, config, approach, logic)
                        for metric in metrics:
                            act = max if valll is False else (min if "financial-cost" in metric else max) 
                            # print(metric, act, valll)
                            is_sup = sorted(list(is_best(store[model][folder][metric],act)), reverse= (False if ((valll == True) and ("financial-cost" in metric )) else True))
                            # print(metric,(False if ((valll == True) and ("financial-cost" in metric )) else True),act(store[model][folder][metric][approach][logic][config]), valll, is_sup)
                            pos = is_sup.index(act(store[model][folder][metric][approach][logic][config]))

                            val = "\\textbf{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 0) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else (
                                "\\underline{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 1) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else (
                                    "\\textit{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 2) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else 
                                    str(act(store[model][folder][metric][approach][logic][config]))
                                    )
                                )

                            lines+= f"& {val}"
                            total_counter[config].append(is_sup)
                    start = (3 if ci != len(list(store[model][financial_folder[0]]['accuracy'][approach][logic].keys()))-1 else (1 if li == len(list(store[model][financial_folder[0]]['accuracy'][approach].keys()))-1 else 2))
                    # print(start)
                    lines+= f"""\\\\ """+ """ \\cline{"""+str(start)+"""-"""+str(nb_cols)+"""}

                    
                    """

        lines+= """

        \\end{tabular}
        }
        %\\end{sidewaystable}"""

        table = table_header + lines
        tables[model] = table
        create_domain(f"{output_path}/tableaux/{str(alpha).replace('.','_')}")
        filename1 = f"{output_path}/tableaux/{str(alpha).replace('.','_')}/tab_alpha_{str(alpha).replace('.','_')}_{model}.tex"
        _file = open(filename1, "w")
        _file.write(table)
        _file.close()
    return tables

def print_compare_v3_1(
    store,
    output_path,
    alpha,
    metrics,
    configs,
    valll=False
):
    # fetch model name
    tables = {model:'' for model in list(store.keys())}
    for model in list(store.keys()):
        # add the resize box to ensure the scale of the table will be contain's inside the width space avalable.
        # start setting up the tabular dimensions setting
        table_header = """
        %\\begin{sidewaystable}
        \\resizebox{\\textwidth}{!}{

        \\begin{tabular}{|c|c|c|"""
        # identify folder with none financial details
        none_financial_folder = [folder for folder in list(store[model].keys()) if sum(['finan' in mec for mec in list(store[model][folder].keys())]) != 1]
        financial_folder = [folder for folder in list(store[model].keys()) if sum(['finan' in mec for mec in list(store[model][folder].keys())]) == 1]
        print(none_financial_folder, financial_folder)
        # add cols for financial folders
        for _ in financial_folder:
            table_header+= "r|"*len(metrics)
        # add cols for non financial folder
        for _ in none_financial_folder:
            table_header+= "r|"*len(list(set(metrics)-set(['financial-cost'])))
        # add col for total results
        table_header+= "} "
        # add separator clines
        nb_cols = (3+(len(metrics)*len(financial_folder))+(len(list(set(metrics)-set(['financial-cost'])))*len(none_financial_folder)))
        table_header+= " \\cline{1-"+str(nb_cols)+"}" # corresponding to the number of columns


        # build the first line
        lines = ''
        # add the model block
        lines += """
        \\multicolumn{3}{|c|}{"""+model+"("+str(alpha).replace('.','_')+""")}"""
        # add cols for financial folders
        for folder in financial_folder:
            lines+= " & \\multicolumn{"+str(len(metrics))+"}{|c|}{"+folder.replace('_','\\_')+"}"
        # add cols for non financial folder
        for folder in none_financial_folder:
            lines+=  " & \\multicolumn{8}{|c|}{"+folder+"}"
        # add the total name
        lines+= "\\\\ "
        lines+= " \\cline{4-"+str(nb_cols)+"}"

        # build metrics' lines
        lines+= """
         \\multicolumn{3}{|c|}{}"""
        # add metrics for financial folders
        dicto = {'accuracy':'Acc', 'f1-score':'F1', 'precision1':'P1', 'recall1':'R1', 'f1-score1':'F11', 'precision0':'P0', 'recall0':'R0', 'f1-score0':'F10', 'financial-cost':'Cost'}
        for folder in financial_folder:
            for met in metrics:
                lines+= f" & {dicto[met]}"
        # add metrics' for non financial folder
        for folder in none_financial_folder:
            for met in list(set(metrics)-set(['financial-cost'])):
                lines+= f" & {dicto[met]}"
        # add the total name
        lines+= " \\\\ "
        lines+= " \\cline{1-"+str(nb_cols)+"}"

        # add classic metrics
        lines+="""
        \\multicolumn{3}{|c|}{Classic}"""
        for folder in financial_folder:
            for metric in metrics:
                # print(model, folder, metric)
                lines+= " & "+str(round(store[model][folder]['classic'].loc[model, metric],4))+""
        # add cols for non financial folder
        for folder in none_financial_folder:
            for metric in list(set(metrics)-set(['financial-cost'])):
                lines+= " & "+str(round(store[model][folder]['classic'].loc[model, metric],4))+""
        # add an empty cell for Total
        lines+= "\\\\ \\cline{1-"+str(nb_cols)+"""}

        """

        is_best = lambda mag,act: (set([act(mag[el][al][il]) for el in list(mag.keys()) for al in list(mag[el].keys()) for il in list(mag[el][al].keys())]))
        # fetch store to fullfil the tables with content
        for ai, approach in enumerate(list(store[model][financial_folder[0]][metrics[0]].keys())): # Mlc or MCA
            lines+= """
            \\multirow{"""+str(configs)+"""}{*}{"""+approach+"""} &"""
            for li, logic in enumerate(list(store[model][financial_folder[0]][metrics[0]][approach].keys())): # GLO, PER or GAP
                total_counter = {}
                lines+= f"{' &'*int(li != 0)}"+"""\\multirow{"""+str(len(list(store[model][financial_folder[0]][metrics[0]][approach][logic].keys())))+"""}{*}{"""+logic+"""}"""
                for ci,config in enumerate(list(store[model][financial_folder[0]][metrics[0]][approach][logic].keys())): # MX, CX, CY, CXY 
                    lines+= f"{' &'*int(ci != 0)}  & {config}"
                    for fi, folder in enumerate(list(store[model].keys())):
                        if (ci == 0) and (fi == 0):
                            total_counter = {key:[] for key in list(store[model][folder][metrics[0]][approach][logic].keys())}
                        # metrics = (['accuracy', 'f1-score', 'precision1', 'recall1', 'f1-score1', 'precision0', 'recall0', 'f1-score0', 'financial-cost'] if (folder in financial_folder) else ['accuracy', 'f1-score', 'precision1', 'recall1', 'f1-score1', 'precision0', 'recall0', 'f1-score0'])
                        # metrics = list(set(metrics)-set(['classic']))
                        # print(folder, metrics, config, approach, logic)
                        for metric in metrics:
                            act = max if valll is False else (min if "financial-cost" in metric else max) 
                            # print(metric, act, valll)
                            is_sup = sorted(list(is_best(store[model][folder][metric],act)), reverse= (False if ((valll == True) and ("financial-cost" in metric )) else True))
                            # print(metric,(False if ((valll == True) and ("financial-cost" in metric )) else True),act(store[model][folder][metric][approach][logic][config]), valll, is_sup)
                            pos = is_sup.index(act(store[model][folder][metric][approach][logic][config]))

                            val = "\\textbf{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 0) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else (
                                "\\underline{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 1) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else (
                                    "\\textit{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 2) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else 
                                    str(act(store[model][folder][metric][approach][logic][config]))
                                    )
                                )

                            lines+= f"& {val}"
                            total_counter[config].append(is_sup)
                    start = (3 if ci != len(list(store[model][financial_folder[0]]['accuracy'][approach][logic].keys()))-1 else (1 if li == len(list(store[model][financial_folder[0]]['accuracy'][approach].keys()))-1 else 2))
                    # print(start)
                    lines+= f"""\\\\ """+ """ \\cline{"""+str(start)+"""-"""+str(nb_cols)+"""}

                    
                    """

        lines+= """

        \\end{tabular}
        }
        %\\end{sidewaystable}"""

        table = table_header + lines
        tables[model] = table
        create_domain(f"{output_path}/tableaux/{str(alpha).replace('.','_')}")
        filename1 = f"{output_path}/tableaux/{str(alpha).replace('.','_')}/tab_alpha_{str(alpha).replace('.','_')}_{model}.tex"
        _file = open(filename1, "w")
        _file.write(table)
        _file.close()
    return tables

"""
    cette version permet de créer une structure de tableau de resultats où nous pouvons avoir un total permettant de decompter et de mieux analyser.
    Etant basé sur la version 3_1 qui garanti une dynamismes des informations contenu dans le tableau.  
"""
def print_compare_v3_2(
    store,
    output_path,
    alpha,
    metrics,
    configs,
    result,
    valll=False
):
    # fetch model name
    tables = {folder:'' for folder in list(store[list(store.keys())[0]].keys())}
    for folder in list(tables.keys()):
        # add the resize box to ensure the scale of the table will be contain's inside the width space avalable.
        # start setting up the tabular dimensions setting
        table_header = """
        %\\begin{sidewaystable}
        \\resizebox{\\textwidth}{!}{

        \\begin{tabular}{|c|c|"""

        # setup information columns headears
        nbMCol = 0
        for metric in metrics:
            table_header+= "r|"
            nbMCol+= 1
            for logic in store[list(store.keys())[0]][folder][metric]["MlC"].keys():
                for config in store[list(store.keys())[0]][folder][metric]["MlC"][logic].keys():
                    table_header+= "r|"
                    nbMCol+= 1
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
        dicto = {'accuracy':'Acc', 'f1-score':'F1', 'precision1':'P1', 'recall1':'R1', 'f1-score1':'F11', 'precision0':'P0', 'recall0':'R0', 'f1-score0':'F10', 'financial-cost':'Cost'}
        for metric in metrics:
            lines+= " & \\multicolumn{"+str(nbMCol//len(metrics))+"}{|c|}{"+dicto[metric]+"}"
        # add the total name
        lines+= " \\\\ "
        lines+= " \\cline{3-"+str(nb_cols)+"}"

        # build the second line: logics' line
        # add the blank block
        lines += """
        \\multicolumn{2}{|c|}{"""+folder+"""}"""
        # add cols for logics
        for metric in metrics:
            lines+= " &"
            for logic in store[list(store.keys())[0]][folder][metric]["MlC"].keys():
                # for config in store[list(store.keys())[0]][folder][metric][logic].keys():
                lines+= " & \\multicolumn{"+str(len(store[list(store.keys())[0]][folder][metric]["MlC"][logic].keys()))+"}{|c|}{"+logic+"}"
        # add the total name
        lines+= " \\\\ "
        lines+= " \\cline{3-"+str(nb_cols)+"}"

        # build the second line: logics' line
        # add the blank block
        lines += """
        \\multicolumn{2}{|c|}{}"""
        # add cols for logics
        for metric in metrics:
            lines+= " & Classic "
            for logic in store[list(store.keys())[0]][folder][metric]["MlC"].keys():
                for config in store[list(store.keys())[0]][folder][metric]["MlC"][logic].keys():
                    lines+= " & "+config
        # add the total name
        lines+= " \\\\ "
        lines+= " \\cline{1-"+str(nb_cols)+"}"

        # here we need to define a structure that will keep track of the number of times a config has the best result for a given metric
        counterStore= {}
        for metric in metrics:
            counterStore[metric] = {} if not(metric in counterStore.keys()) else counterStore[metric]
            for logic in store[list(store.keys())[0]][folder][metric]['MlC'].keys():
                counterStore[metric][logic] = {} if not(logic in counterStore[metric].keys()) else counterStore[metric][logic]
                for config in store[list(store.keys())[0]][folder][metric]["MlC"][logic].keys():
                    counterStore[metric][logic][config] = []

        counterStore_1 = deepcopy(counterStore)
        counterStore_2 = deepcopy(counterStore)
        # print(counterStore)
        # having done this, we now need a function that will return max of the metrics in a config with a given logic
        max_config = lambda store, mod, folder, metric, act: (set([act(store[mod][folder][metric][app][logic][config]) for app in store[mod][folder][metric].keys() for logic in store[mod][folder][metric][app].keys() for config in store[mod][folder][metric][app][logic].keys()]))

        # fetch on model
        for mi, model in enumerate(list(store.keys())): # LDA, LR, SVM, DT, RF, XGB
            lines+= """
            \\multirow{"""+str(len(list(store[model][folder][metrics[0]].keys())))+"""}{*}{"""+model+"""}"""
            for ai, approach in enumerate(list(store[model][folder][metrics[0]].keys())): # MlC, MCA
                total_counter = {}
                lines+= f""" & {approach}"""
                for i, metric in enumerate(metrics):
                    lines+= """ & \\multirow{2}{*}{"""+str(round(store[model][folder]['classic'].loc[model, metric],4))+"""}""" if ai == 0 else """ &"""
                    for li,logic in enumerate(list(store[model][folder][metric][approach].keys())):
                        for ci,config in enumerate(list(store[model][folder][metric][approach][logic].keys())): # MX, CX, CY, CXY 
                            # print(folder, model, approach, metric, logic, config)
                            act = max if valll is False else (min if "financial-cost" in metric else max)
                            act_b = sorted(list(max_config(store, model, folder, metric, act)), reverse= (False if ((valll == True) and ("financial-cost" in metric )) else True))
                            pos = act_b.index(act(store[model][folder][metric][approach][logic][config]))

                            val = "\\textbf{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 0) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else (
                                "\\underline{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 1) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else (
                                    "\\textit{"+str(act(store[model][folder][metric][approach][logic][config]))+"}" if (pos == 2) and (act(store[model][folder][metric][approach][logic][config]) != 0.0) else 
                                    str(act(store[model][folder][metric][approach][logic][config]))
                                    )
                                )

                            lines+= f"& {val}"
                            counterStore[metric][logic][config].append(pos == 0)
                            counterStore_1[metric][logic][config].append(pos == 1)
                            counterStore_2[metric][logic][config].append(pos == 2)
                # ---------------------------------------
                if (ai != len(store[model][folder][metrics[0]].keys())-1):
                    start = 2
                    prog = (nb_cols//len(metrics))
                    # print(start)
                    lines+= """\\\\ """
                    for _ in range(len(metrics)):
                        lines+= """ \\cline{"""+str(start)+"-"+str(start)+"""} \\cline{"""+str(start+2)+"""-"""+str(start+(nb_cols//len(metrics)))+"""}"""
                        start = start+prog
                    lines+= """
                    """
                else:
                    lines+= """\\\\ """+ """ \\cline{1-"""+str(nb_cols)+"""}
                        """ 

        # total's line
        # print(result)
        maxim = lambda dictio, metric, logic: (max([ sum(dictio[metric][logic][config]) for config in dictio[metric][logic].keys() ]))
        if 1 in result:
            lines += """
            \\multicolumn{2}{|c|}{\\textbf{Total 1st Place}}"""

            for j, metric in enumerate(metrics):
                lines += """ & """
                for logic in store[list(store.keys())[0]][folder][metric]['MlC'].keys():
                    for config in store[list(store.keys())[0]][folder][metric]['MlC'][logic].keys():
                        lines += """ & """+str(sum(counterStore[metric][logic][config])) if sum(counterStore[metric][logic][config]) != maxim(counterStore, metric,logic) else """ & \\textbf{"""+str(sum(counterStore[metric][logic][config]))+"}"

            lines+= " \\\\ "
            lines+= " \\cline{1-"+str(nb_cols)+"}"

        if 2 in result:
            lines += """
            \\multicolumn{2}{|c|}{\\textbf{Total 2nd Place}}"""

            for j, metric in enumerate(metrics):
                lines += """ & """
                for logic in store[list(store.keys())[0]][folder][metric]['MlC'].keys():
                    for config in store[list(store.keys())[0]][folder][metric]['MlC'][logic].keys():
                        lines += """ & """+str(sum(counterStore_1[metric][logic][config])) if sum(counterStore_1[metric][logic][config]) != maxim(counterStore_1, metric,logic) else """ & \\textbf{"""+str(sum(counterStore_1[metric][logic][config]))+"}"

            lines+= " \\\\ "
            lines+= " \\cline{1-"+str(nb_cols)+"}"

        if 3 in result:
            lines += """
            \\multicolumn{2}{|c|}{\\textbf{Total 3th Place}}"""

            for j, metric in enumerate(metrics):
                lines += """ & """
                for logic in store[list(store.keys())[0]][folder][metric]['MlC'].keys():
                    for config in store[list(store.keys())[0]][folder][metric]['MlC'][logic].keys():
                        lines += """ & """+str(sum(counterStore_2[metric][logic][config])) if sum(counterStore_2[metric][logic][config]) != maxim(counterStore_2, metric,logic) else """ & \\textbf{"""+str(sum(counterStore_2[metric][logic][config]))+"}"

            lines+= " \\\\ "
            lines+= " \\cline{1-"+str(nb_cols)+"}"

        lines+= """

        \\end{tabular}
        }
        %\\end{sidewaystable}"""

        table = table_header + lines
        tables[folder] = table
        create_domain(f"{output_path}/tableaux/{str(alpha).replace('.','_')}")
        filename1 = f"{output_path}/tableaux/{str(alpha).replace('.','_')}/tab_alpha_{str(alpha).replace('.','_')}_{folder}.tex"
        _file = open(filename1, "w")
        _file.write(table)
        _file.close()
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
def generate_report_tables(
    outputs_name=None, 
    cwd=None, 
    layers=[1], 
    _type='qualitative',
    alphas=[0.85]
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
    day = time.strftime("%Y_%m_%d_%H")
    ## identify the name of results folder
    data_result_folder_name = [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{outputs_name}')][0]
    print(data_result_folder_name)
    ## init a global container for containing the results of all results folder
    template_details_metrics_depth_1 = {
        'accuracy': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'f1-score': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'financial-cost': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        }
    }

    template_details_metrics_depth_2 = {
        'accuracy': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }

        },
        'f1-score': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'financial-cost': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        }
    }

    ######################################
    ig_global_details_metrics_depth_1 = {
    }
    val_global_details_metrics_depth_1 = {
    }

    ig_global_details_metrics_depth_2 = {
    }
    val_global_details_metrics_depth_2 = {
    }

    best_mlna_k_per_alpha = {
        folder: {key: {'real_best_k': [], 'predicted_best_k':[], 'value':[], 'model':[]} for key in alphas} for folder in data_result_folder_name
    }
    #####################################
    ig_global_tables = ""
    val_global_tables = ""
    ig_local_tables = {}
    val_local_tables = {}
    ## fetch on alpha values
    for index, alpha in enumerate(alphas):
        ## for each alpha, store a local container 
        ig_local_details_metrics_depth_1 = {
        }
        val_local_details_metrics_depth_1 = {
        }
        ig_local_details_metrics_depth_2 = {
        }
        val_local_details_metrics_depth_2 = {
        }

        local_best_mlna_k_per_alpha = {
            key: deepcopy(best_mlna_k_per_alpha) for key in data_result_folder_name
        }

        
        ## fetch on name of folders
        for index2, result_folder in enumerate(data_result_folder_name):
            # load predicted best mlnk
            print(result_folder)
            name = \
                [file for _, _, files in os.walk(
                    f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage')
                 for file in files if
                 '_best_features' in file][0]
            best_mlna_k_per_alpha[result_folder][alpha]['predicted_best_k'].append(read_model(f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage/{name}')["bestK"])
            ## get classic model
            classic_f = [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{os.getcwd()}/{outputs_name}/{result_folder}/data_selection_storage',
                    func=C_F,
                    verbose=False
                )
            ][-1]
            # define a list which will receive all possible value of accuracy for each layer in aims tp sorted it later and extra, the best best layer and model
            list_of_accuracy = []
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
                    mix=True
                )
                ## aggregate results belong to metrics specific to the current results
                for index4, model in enumerate(models_name):
                    # print(model)  
                    # append contains structured to our store
                    if (index2 == index3) and (index2 == 0): # if first alpha, first folder result, and first layer

                        # local ones
                        ig_local_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}
                        val_local_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}

                        ig_local_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        val_local_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                    if (index2 == index3) and (index2 == 0) and (index == 0):
                        ig_local_tables[model] = ""
                        val_local_tables[model] = ""
                        print('here')
                        # global one
                        ig_global_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}
                        val_global_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}

                        ig_global_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        val_global_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        # print(ig_local_details_metrics_depth_1)

                    # assuming that our store structore now have a financial metric section, we need to ensure that the current result folder
                    # has a financial dimension
                    hasFinancialCost = sum(['finan' in el for el in classic_f.columns.values.tolist()])
                    if hasFinancialCost == 0:
                        # if there is not financial details, remove the section in ours store
                        # for mod in list(ig_local_details_metrics_depth_1.keys()): 
                        # print(index, index2, index3, index4)
                        if 'financial-cost' in list(ig_local_details_metrics_depth_1[model][result_folder].keys()):
                            # local
                            del ig_local_details_metrics_depth_1[model][result_folder]['financial-cost'] 
                            del val_local_details_metrics_depth_1[model][result_folder]['financial-cost']
                            del ig_local_details_metrics_depth_2[model][result_folder]['financial-cost']
                            del val_local_details_metrics_depth_2[model][result_folder]['financial-cost']
                            if (index3 == 0) and (index == 0):
                                # global
                                del ig_global_details_metrics_depth_1[model][result_folder]['financial-cost']
                                del val_global_details_metrics_depth_1[model][result_folder]['financial-cost']
                                del ig_global_details_metrics_depth_2[model][result_folder]['financial-cost']
                                del val_global_details_metrics_depth_2[model][result_folder]['financial-cost']

                    val_local_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    ig_local_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    val_local_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    ig_local_details_metrics_depth_2[model][result_folder]['classic'] = classic_f

                    val_global_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    ig_global_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    val_global_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    ig_global_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    # now fetch ours results to store
                    for metric in list(set(list(ig_local_details_metrics_depth_1[model][result_folder].keys())) - set(['classic'])): # accuracy and f1-score and/or financial cost
                        for approach in list(files.keys()): # Mlc or MCA
                            # print(metric, approach)
                            for logic in list(files[approach].keys()): # GLO, PER or GAP
                                for config in list(files[approach][logic].keys()): # MX, CX, CY, CXY
                                    for result in list(range(len(files[approach][logic][config]))): # each result file's containing evaluation metrics
                                        # save exact metric values
                                        val_local_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],4))
                                        val_global_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],4))

                                        val_local_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],4))
                                        val_global_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],4))

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
                                                    round(
                                                        classic_f.loc[model, metric], 
                                                        4
                                                        )
                                                    ) * 100, 
                                                1
                                                )
                                        ig =  valu if 'finan' not in metric else  (valu if valu ==0 else -1*valu) 
                                        # save just gain information
                                        ig_local_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(ig)
                                        ig_global_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(ig)

                                        ig_local_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(ig)
                                        ig_global_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(ig)
                            
                                        if 'accu' in metric:
                                            list_of_accuracy.append((layer, model,files[approach][logic][config][result].loc[model, metric]))

            # analyse impact of layers and identify the best mlna as k
            list_of_accuracy = sorted(list_of_accuracy, key=lambda x: abs(x[2]), reverse=False) # best will be at position 0
            best_mlna_k_per_alpha[result_folder][alpha]['real_best_k'].append(list_of_accuracy[0][0])
            best_mlna_k_per_alpha[result_folder][alpha]['model'].append(list_of_accuracy[0][1])
            best_mlna_k_per_alpha[result_folder][alpha]['value'].append(list_of_accuracy[0][2])

        ## on store, call a print table function (gain, metric)
        ig_local_tables = {key: ig_local_tables[key] + "\n" + mod for key, mod in print_compare_v2(
            ig_local_details_metrics_depth_1,
            f'{cwd}/analyze/ig',
            alpha
        ).items()}
        val_local_tables = {key: val_local_tables[key] + "\n" + mod for key, mod in print_compare_v2(
            val_local_details_metrics_depth_1,
            f'{cwd}/analyze/val',
            alpha,
            True
        ).items()}

    ## print the global container (gain, metric)
    ig_global_tables = "\n".join([ mod for _, mod in print_compare_v2(
        ig_global_details_metrics_depth_1,
        f'{cwd}/analyze/ig',
        'all'
    ).items()])
    val_global_tables = "\n".join([ mod for _, mod in print_compare_v2(
        val_global_details_metrics_depth_1,
        f'{cwd}/analyze/val',
        'all',
        True
    ).items()])
    create_domain(f'{cwd}/analyze/ig/all_ig/')
    with open(f'{cwd}/analyze/ig/all_ig/ig_all.tex', "a") as fichier:
        fichier.write(header+ig_global_tables+footer)
    create_domain(f'{cwd}/analyze/val/val_all/')
    with open(f'{cwd}/analyze/val/val_all/val_all.tex', "a") as fichier:
        fichier.write(header+val_global_tables+footer)

    for key in list(ig_local_tables.keys()):
        create_domain(f'{cwd}/analyze/ig/{key}/')
        with open(f'{cwd}/analyze/ig/{key}/{key}.tex', "a") as fichier:
            fichier.write(header+ig_local_tables[key]+footer)
        create_domain(f'{cwd}/analyze/val/{key}/')
        with open(f'{cwd}/analyze/val/{key}/{key}.tex', "a") as fichier:
            fichier.write(header+val_local_tables[key]+footer)

    joblib.dump(best_mlna_k_per_alpha, f'{cwd}/analyze/best.tex')

"""
    Impact performance analytic for metrics: accuracy, f1-score, precision1, recall1, f1-score1, precision0, recall0, f1-score0 and financial-cost for all configurations
"""
def generate_report_tables_v2(
    outputs_name=None, 
    cwd=None, 
    layers=[1], 
    _type='qualitative',
    alphas=[0.85]
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
    day = time.strftime("%Y_%m_%d_%H")
    ## identify the name of results folder
    data_result_folder_name = [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{outputs_name}')][0]
    print(data_result_folder_name)
    ## init a global container for containing the results of all results folder

    template_details_metrics_depth_1 = {
        'accuracy': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'f1-score': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'precision1': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'recall1': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'f1-score1': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'precision0': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'recall0': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'f1-score0': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'financial-cost': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        }
    }
    print(template_details_metrics_depth_1.keys())
    template_details_metrics_depth_2 = {
        'accuracy': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'f1-score': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'precision1': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'recall1': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'f1-score1': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'precision0': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'recall0': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'f1-score0': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'financial-cost': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        }
    }

    ######################################
    ig_global_details_metrics_depth_1 = {
    }
    val_global_details_metrics_depth_1 = {
    }

    ig_global_details_metrics_depth_2 = {
    }
    val_global_details_metrics_depth_2 = {
    }

    best_mlna_k_per_alpha = {
        folder: {key: {'real_best_k': [], 'predicted_best_k':[], 'value':[], 'model':[]} for key in alphas} for folder in data_result_folder_name
    }
    #####################################
    ig_global_tables = ""
    val_global_tables = ""
    ig_local_tables = {}
    val_local_tables = {}
    ## fetch on alpha values
    for index, alpha in enumerate(alphas):
        ## for each alpha, store a local container 
        ig_local_details_metrics_depth_1 = {
        }
        val_local_details_metrics_depth_1 = {
        }
        ig_local_details_metrics_depth_2 = {
        }
        val_local_details_metrics_depth_2 = {
        }

        local_best_mlna_k_per_alpha = {
            key: deepcopy(best_mlna_k_per_alpha) for key in data_result_folder_name
        }

        
        ## fetch on name of folders
        for index2, result_folder in enumerate(data_result_folder_name):
            # load predicted best mlnk
            print(result_folder)
            name = \
                [file for _, _, files in os.walk(
                    f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage')
                 for file in files if
                 '_best_features' in file][0]
            best_mlna_k_per_alpha[result_folder][alpha]['predicted_best_k'].append(read_model(f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage/{name}')["bestK"])
            ## get classic model
            classic_f = [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{os.getcwd()}/{outputs_name}/{result_folder}/data_selection_storage',
                    func=C_F,
                    verbose=False
                )
            ][-1]
            # define a list which will receive all possible value of accuracy for each layer in aims tp sorted it later and extra, the best best layer and model
            list_of_accuracy = []
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
                    mix=True
                )
                ## aggregate results belong to metrics specific to the current results
                for index4, model in enumerate(models_name):
                    # print(model)  
                    # append contains structured to our store
                    if (index2 == index3) and (index2 == 0): # if first alpha, first folder result, and first layer

                        # local ones
                        ig_local_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}
                        val_local_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}

                        ig_local_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        val_local_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                    if (index2 == index3) and (index2 == 0) and (index == 0):
                        ig_local_tables[model] = ""
                        val_local_tables[model] = ""
                        print('here')
                        # global one
                        ig_global_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}
                        val_global_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}

                        ig_global_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        val_global_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        # print(ig_local_details_metrics_depth_1)

                    # assuming that our store structore now have a financial metric section, we need to ensure that the current result folder
                    # has a financial dimension
                    hasFinancialCost = sum(['finan' in el for el in classic_f.columns.values.tolist()])
                    if hasFinancialCost == 0:
                        # if there is not financial details, remove the section in ours store
                        # for mod in list(ig_local_details_metrics_depth_1.keys()): 
                        # print(index, index2, index3, index4)
                        if 'financial-cost' in list(ig_local_details_metrics_depth_1[model][result_folder].keys()):
                            # local
                            del ig_local_details_metrics_depth_1[model][result_folder]['financial-cost'] 
                            del val_local_details_metrics_depth_1[model][result_folder]['financial-cost']
                            del ig_local_details_metrics_depth_2[model][result_folder]['financial-cost']
                            del val_local_details_metrics_depth_2[model][result_folder]['financial-cost']
                            if (index3 == 0) and (index == 0):
                                # global
                                del ig_global_details_metrics_depth_1[model][result_folder]['financial-cost']
                                del val_global_details_metrics_depth_1[model][result_folder]['financial-cost']
                                del ig_global_details_metrics_depth_2[model][result_folder]['financial-cost']
                                del val_global_details_metrics_depth_2[model][result_folder]['financial-cost']

                    val_local_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    ig_local_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    val_local_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    ig_local_details_metrics_depth_2[model][result_folder]['classic'] = classic_f

                    val_global_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    ig_global_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    val_global_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    ig_global_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    # now fetch ours results to store
                    for metric in list(set(list(ig_local_details_metrics_depth_1[model][result_folder].keys())) - set(['classic'])): # accuracy and f1-score and/or financial cost
                        for approach in list(files.keys()): # Mlc or MCA
                            # print(metric, approach)
                            for logic in list(files[approach].keys()): # GLO, PER or GAP
                                for config in list(files[approach][logic].keys()): # MX, CX, CY, CXY
                                    for result in list(range(len(files[approach][logic][config]))): # each result file's containing evaluation metrics
                                        # save exact metric values
                                        val_local_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],4))
                                        val_global_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],4))

                                        val_local_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],4))
                                        val_global_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],4))

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
                                                    round(
                                                        classic_f.loc[model, metric], 
                                                        4
                                                        )
                                                    ) * 100, 
                                                1
                                                )
                                        ig =  valu if 'finan' not in metric else  (valu if valu ==0 else -1*valu) 
                                        # save just gain information
                                        ig_local_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(ig)
                                        ig_global_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(ig)

                                        ig_local_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(ig)
                                        ig_global_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(ig)
                            
                                        if 'accu' in metric:
                                            list_of_accuracy.append((layer, model,files[approach][logic][config][result].loc[model, metric]))

            # analyse impact of layers and identify the best mlna as k
            list_of_accuracy = sorted(list_of_accuracy, key=lambda x: abs(x[2]), reverse=False) # best will be at position 0
            best_mlna_k_per_alpha[result_folder][alpha]['real_best_k'].append(list_of_accuracy[0][0])
            best_mlna_k_per_alpha[result_folder][alpha]['model'].append(list_of_accuracy[0][1])
            best_mlna_k_per_alpha[result_folder][alpha]['value'].append(list_of_accuracy[0][2])

        ## on store, call a print table function (gain, metric)
        ig_local_tables = {key: ig_local_tables[key] + "\n" + mod for key, mod in print_compare_v3(
            ig_local_details_metrics_depth_1,
            f'{cwd}/analyzeV1/ig',
            alpha
        ).items()}
        val_local_tables = {key: val_local_tables[key] + "\n" + mod for key, mod in print_compare_v3(
            val_local_details_metrics_depth_1,
            f'{cwd}/analyzeV1/val',
            alpha,
            True
        ).items()}

    ## print the global container (gain, metric)
    ig_global_tables = "\n".join([ mod for _, mod in print_compare_v3(
        ig_global_details_metrics_depth_1,
        f'{cwd}/analyzeV1/ig',
        'all'
    ).items()])
    val_global_tables = "\n".join([ mod for _, mod in print_compare_v3(
        val_global_details_metrics_depth_1,
        f'{cwd}/analyzeV1/val',
        'all',
        True
    ).items()])
    create_domain(f'{cwd}/analyzeV1/ig/all_ig/')
    with open(f'{cwd}/analyzeV1/ig/all_ig/ig_all.tex', "a") as fichier:
        fichier.write(header+ig_global_tables+footer)
    create_domain(f'{cwd}/analyzeV1/val/val_all/')
    with open(f'{cwd}/analyzeV1/val/val_all/val_all.tex', "a") as fichier:
        fichier.write(header+val_global_tables+footer)

    for key in list(ig_local_tables.keys()):
        create_domain(f'{cwd}/analyzeV1/ig/{key}/')
        with open(f'{cwd}/analyzeV1/ig/{key}/{key}.tex', "a") as fichier:
            fichier.write(header+ig_local_tables[key]+footer)
        create_domain(f'{cwd}/analyzeV1/val/{key}/')
        with open(f'{cwd}/analyzeV1/val/{key}/{key}.tex', "a") as fichier:
            fichier.write(header+val_local_tables[key]+footer)

    joblib.dump(best_mlna_k_per_alpha, f'{cwd}/analyzeV1/best.tex')

"""
    Impact performance analytic for metrics: accuracy, f1-score, precision1, recall1, f1-score1, precision0, recall0, f1-score0 and financial-cost for all configurations
    !st, 2nd and 3th best values are highlight
"""
def generate_report_tables_v3(
    outputs_name=None, 
    cwd=None, 
    layers=[1], 
    _type='qualitative',
    alphas=[0.85]
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
    day = time.strftime("%Y_%m_%d_%H")
    ## identify the name of results folder
    data_result_folder_name = [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{outputs_name}')][0]
    print(data_result_folder_name)
    ## init a global container for containing the results of all results folder

    template_details_metrics_depth_1 = {
        'accuracy': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'f1-score': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'precision1': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'recall1': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'f1-score1': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'precision0': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'recall0': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'f1-score0': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        },
        'financial-cost': {
            'MlC': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            },
            'MCA': {
                'GLO': {
                    'MX': [],
                    'CX': [],
                },
                'PER': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                },
                'GAP': {
                    'MX': [],
                    'CX': [],
                    'CY': [],
                    'CXY': []
                }
            }
        }
    }
    print(template_details_metrics_depth_1.keys())
    template_details_metrics_depth_2 = {
        'accuracy': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'f1-score': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'precision1': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'recall1': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'f1-score1': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'precision0': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'recall0': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'f1-score0': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        },
        'financial-cost': {
            'MlC': {
                'GLO': [],
                'PER': [],
                'GAP': []
            },
            'MCA': {
                'GLO': [],
                'PER': [],
                'GAP': []
            }
        }
    }

    ######################################
    ig_global_details_metrics_depth_1 = {
    }
    val_global_details_metrics_depth_1 = {
    }

    ig_global_details_metrics_depth_2 = {
    }
    val_global_details_metrics_depth_2 = {
    }

    best_mlna_k_per_alpha = {
        folder: {key: {'real_best_k': [], 'predicted_best_k':[], 'value':[], 'model':[]} for key in alphas} for folder in data_result_folder_name
    }
    #####################################
    ig_global_tables = ""
    val_global_tables = ""
    ig_local_tables = {}
    val_local_tables = {}
    ## fetch on alpha values
    for index, alpha in enumerate(alphas):
        ## for each alpha, store a local container 
        ig_local_details_metrics_depth_1 = {
        }
        val_local_details_metrics_depth_1 = {
        }
        ig_local_details_metrics_depth_2 = {
        }
        val_local_details_metrics_depth_2 = {
        }

        local_best_mlna_k_per_alpha = {
            key: deepcopy(best_mlna_k_per_alpha) for key in data_result_folder_name
        }

        
        ## fetch on name of folders
        for index2, result_folder in enumerate(data_result_folder_name):
            # load predicted best mlnk
            print(result_folder)
            name = \
                [file for _, _, files in os.walk(
                    f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage')
                 for file in files if
                 '_best_features' in file][0]
            best_mlna_k_per_alpha[result_folder][alpha]['predicted_best_k'].append(read_model(f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage/{name}')["bestK"])
            ## get classic model
            classic_f = [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{os.getcwd()}/{outputs_name}/{result_folder}/data_selection_storage',
                    func=C_F,
                    verbose=False
                )
            ][-1]
            # define a list which will receive all possible value of accuracy for each layer in aims tp sorted it later and extra, the best best layer and model
            list_of_accuracy = []
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
                    mix=True
                )
                ## aggregate results belong to metrics specific to the current results
                for index4, model in enumerate(models_name):
                    # print(model)  
                    # append contains structured to our store
                    if (index2 == index3) and (index2 == 0): # if first alpha, first folder result, and first layer

                        # local ones
                        ig_local_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}
                        val_local_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}

                        ig_local_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        val_local_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                    if (index2 == index3) and (index2 == 0) and (index == 0):
                        ig_local_tables[model] = ""
                        val_local_tables[model] = ""
                        print('here')
                        # global one
                        ig_global_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}
                        val_global_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}

                        ig_global_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        val_global_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        # print(ig_local_details_metrics_depth_1)

                    # assuming that our store structore now have a financial metric section, we need to ensure that the current result folder
                    # has a financial dimension
                    hasFinancialCost = sum(['finan' in el for el in classic_f.columns.values.tolist()])
                    if hasFinancialCost == 0:
                        # if there is not financial details, remove the section in ours store
                        # for mod in list(ig_local_details_metrics_depth_1.keys()): 
                        # print(index, index2, index3, index4)
                        if 'financial-cost' in list(ig_local_details_metrics_depth_1[model][result_folder].keys()):
                            # local
                            del ig_local_details_metrics_depth_1[model][result_folder]['financial-cost'] 
                            del val_local_details_metrics_depth_1[model][result_folder]['financial-cost']
                            del ig_local_details_metrics_depth_2[model][result_folder]['financial-cost']
                            del val_local_details_metrics_depth_2[model][result_folder]['financial-cost']
                            if (index3 == 0) and (index == 0):
                                # global
                                del ig_global_details_metrics_depth_1[model][result_folder]['financial-cost']
                                del val_global_details_metrics_depth_1[model][result_folder]['financial-cost']
                                del ig_global_details_metrics_depth_2[model][result_folder]['financial-cost']
                                del val_global_details_metrics_depth_2[model][result_folder]['financial-cost']

                    val_local_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    ig_local_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    val_local_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    ig_local_details_metrics_depth_2[model][result_folder]['classic'] = classic_f

                    val_global_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    ig_global_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    val_global_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    ig_global_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    # now fetch ours results to store
                    for metric in list(set(list(ig_local_details_metrics_depth_1[model][result_folder].keys())) - set(['classic'])): # accuracy and f1-score and/or financial cost
                        for approach in list(files.keys()): # Mlc or MCA
                            # print(metric, approach)
                            for logic in list(files[approach].keys()): # GLO, PER or GAP
                                for config in list(files[approach][logic].keys()): # MX, CX, CY, CXY
                                    for result in list(range(len(files[approach][logic][config]))): # each result file's containing evaluation metrics
                                        # save exact metric values
                                        val_local_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],4))
                                        val_global_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],4))

                                        val_local_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],4))
                                        val_global_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],4))

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
                                                    round(
                                                        classic_f.loc[model, metric], 
                                                        4
                                                        )
                                                    ) * 100, 
                                                1
                                                )
                                        ig =  valu if 'finan' not in metric else  (valu if valu ==0 else -1*valu) 
                                        # save just gain information
                                        ig_local_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(ig)
                                        ig_global_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(ig)

                                        ig_local_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(ig)
                                        ig_global_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(ig)
                            
                                        if 'accu' in metric:
                                            list_of_accuracy.append((layer, model,files[approach][logic][config][result].loc[model, metric]))

            # analyse impact of layers and identify the best mlna as k
            list_of_accuracy = sorted(list_of_accuracy, key=lambda x: abs(x[2]), reverse=False) # best will be at position 0
            best_mlna_k_per_alpha[result_folder][alpha]['real_best_k'].append(list_of_accuracy[0][0])
            best_mlna_k_per_alpha[result_folder][alpha]['model'].append(list_of_accuracy[0][1])
            best_mlna_k_per_alpha[result_folder][alpha]['value'].append(list_of_accuracy[0][2])

        ## on store, call a print table function (gain, metric)
        ig_local_tables = {key: ig_local_tables[key] + "\n" + mod for key, mod in print_compare_v3(
            ig_local_details_metrics_depth_1,
            f'{cwd}/analyzeV1/ig',
            alpha
        ).items()}
        val_local_tables = {key: val_local_tables[key] + "\n" + mod for key, mod in print_compare_v3(
            val_local_details_metrics_depth_1,
            f'{cwd}/analyzeV1/val',
            alpha,
            True
        ).items()}

    ## print the global container (gain, metric)
    ig_global_tables = "\n".join([ mod for _, mod in print_compare_v3(
        ig_global_details_metrics_depth_1,
        f'{cwd}/analyzeV1/ig',
        'all'
    ).items()])
    val_global_tables = "\n".join([ mod for _, mod in print_compare_v3(
        val_global_details_metrics_depth_1,
        f'{cwd}/analyzeV1/val',
        'all',
        True
    ).items()])
    create_domain(f'{cwd}/analyzeV1/ig/all_ig/')
    with open(f'{cwd}/analyzeV1/ig/all_ig/ig_all.tex', "a") as fichier:
        fichier.write(header+ig_global_tables+footer)
    create_domain(f'{cwd}/analyzeV1/val/val_all/')
    with open(f'{cwd}/analyzeV1/val/val_all/val_all.tex', "a") as fichier:
        fichier.write(header+val_global_tables+footer)

    for key in list(ig_local_tables.keys()):
        create_domain(f'{cwd}/analyzeV1/ig/{key}/')
        with open(f'{cwd}/analyzeV1/ig/{key}/{key}.tex', "a") as fichier:
            fichier.write(header+ig_local_tables[key]+footer)
        create_domain(f'{cwd}/analyzeV1/val/{key}/')
        with open(f'{cwd}/analyzeV1/val/{key}/{key}.tex', "a") as fichier:
            fichier.write(header+val_local_tables[key]+footer)

    joblib.dump(best_mlna_k_per_alpha, f'{cwd}/analyzeV1/best.tex')

def generate_report_tables_v3_1(
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
    metrics=['accuracy','f1-score','precision1','recall1','f1-score1','precision0','recall0','f1-score0','financial-cost'],
    _type='qualitative',
    alphas=[0.85]
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
    day = time.strftime("%Y_%m_%d_%H")
    ## identify the name of results folder
    data_result_folder_name = [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{outputs_name}')][0]
    print(data_result_folder_name)
    print(metrics)
    ## init a global container for containing the results of all results folder
    template_details_metrics_depth_1 = {}
    template_details_metrics_depth_2 = {}
    _totalConfigs = 0
    for i,metric in enumerate(metrics):
        if not(metric in list(template_details_metrics_depth_1.keys())): # it's metric already exist in the container
            template_details_metrics_depth_1[metric] = {}
            template_details_metrics_depth_2[metric] = {}
        for j,approach in enumerate(approachs):
            if not(approach in list(template_details_metrics_depth_1[metric].keys())): # it's approach already exist in the container
                template_details_metrics_depth_1[metric][approach] = {}
                template_details_metrics_depth_2[metric][approach] = {}
            for logic in logics:
                if not(logic in list(template_details_metrics_depth_1[metric][approach].keys())): # it's logic already exist in the container
                    template_details_metrics_depth_1[metric][approach][logic] = {}
                    template_details_metrics_depth_2[metric][approach][logic] = []
                for config in configs:
                    if logic == 'GLO':
                        if config in ['MX','CX']:
                            template_details_metrics_depth_1[metric][approach][logic][config] = []
                            if i == 0 and j == 0:
                                _totalConfigs += 1
                    else:
                        template_details_metrics_depth_1[metric][approach][logic][config] = []
                        if i == 0 and j == 0:
                            _totalConfigs += 1

    # print(_totalConfigs)
    # return
    # print(template_details_metrics_depth_1)
    # print(template_details_metrics_depth_2)
    # return
    # template_details_metrics_depth_1 = {
    #     'accuracy': {
    #         'MlC': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         },
    #         'MCA': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         }
    #     },
    #     'f1-score': {
    #         'MlC': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         },
    #         'MCA': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         }
    #     },
    #     'precision1': {
    #         'MlC': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         },
    #         'MCA': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         }
    #     },
    #     'recall1': {
    #         'MlC': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         },
    #         'MCA': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         }
    #     },
    #     'f1-score1': {
    #         'MlC': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         },
    #         'MCA': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         }
    #     },
    #     'precision0': {
    #         'MlC': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         },
    #         'MCA': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         }
    #     },
    #     'recall0': {
    #         'MlC': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         },
    #         'MCA': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         }
    #     },
    #     'f1-score0': {
    #         'MlC': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         },
    #         'MCA': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         }
    #     },
    #     'financial-cost': {
    #         'MlC': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         },
    #         'MCA': {
    #             'GLO': {
    #                 'MX': [],
    #                 'CX': [],
    #             },
    #             'PER': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             },
    #             'GAP': {
    #                 'MX': [],
    #                 'CX': [],
    #                 'CY': [],
    #                 'CXY': []
    #             }
    #         }
    #     }
    # }
    # print(template_details_metrics_depth_1.keys())
    # template_details_metrics_depth_2 = {
    #     'accuracy': {
    #         'MlC': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         },
    #         'MCA': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         }
    #     },
    #     'f1-score': {
    #         'MlC': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         },
    #         'MCA': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         }
    #     },
    #     'precision1': {
    #         'MlC': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         },
    #         'MCA': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         }
    #     },
    #     'recall1': {
    #         'MlC': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         },
    #         'MCA': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         }
    #     },
    #     'f1-score1': {
    #         'MlC': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         },
    #         'MCA': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         }
    #     },
    #     'precision0': {
    #         'MlC': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         },
    #         'MCA': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         }
    #     },
    #     'recall0': {
    #         'MlC': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         },
    #         'MCA': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         }
    #     },
    #     'f1-score0': {
    #         'MlC': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         },
    #         'MCA': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         }
    #     },
    #     'financial-cost': {
    #         'MlC': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         },
    #         'MCA': {
    #             'GLO': [],
    #             'PER': [],
    #             'GAP': []
    #         }
    #     }
    # }

    ######################################
    ig_global_details_metrics_depth_1 = {
    }
    val_global_details_metrics_depth_1 = {
    }

    ig_global_details_metrics_depth_2 = {
    }
    val_global_details_metrics_depth_2 = {
    }

    best_mlna_k_per_alpha = {
        folder: {key: {'real_best_k': [], 'predicted_best_k':[], 'value':[], 'model':[]} for key in alphas} for folder in data_result_folder_name
    }
    #####################################
    ig_global_tables = ""
    val_global_tables = ""
    ig_local_tables = {}
    val_local_tables = {}
    ## fetch on alpha values
    for index, alpha in enumerate(alphas):
        ## for each alpha, store a local container 
        ig_local_details_metrics_depth_1 = {
        }
        val_local_details_metrics_depth_1 = {
        }
        ig_local_details_metrics_depth_2 = {
        }
        val_local_details_metrics_depth_2 = {
        }

        local_best_mlna_k_per_alpha = {
            key: deepcopy(best_mlna_k_per_alpha) for key in data_result_folder_name
        }

        
        ## fetch on name of folders
        for index2, result_folder in enumerate(data_result_folder_name):
            # load predicted best mlnk
            print(result_folder)
            name = \
                [file for _, _, files in os.walk(
                    f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage')
                 for file in files if
                 '_best_features' in file][0]
            best_mlna_k_per_alpha[result_folder][alpha]['predicted_best_k'].append(read_model(f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage/{name}')["bestK"])
            ## get classic model
            classic_f = [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{os.getcwd()}/{outputs_name}/{result_folder}/data_selection_storage',
                    func=C_F,
                    verbose=False
                )
            ][-1]
            # define a list which will receive all possible value of accuracy for each layer in aims tp sorted it later and extra, the best best layer and model
            list_of_accuracy = []
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
                    mix=True
                )
                ## aggregate results belong to metrics specific to the current results
                for index4, model in enumerate(models_name):
                    # print(model)  
                    # append contains structured to our store
                    if (index2 == index3) and (index2 == 0): # if first alpha, first folder result, and first layer

                        # local ones
                        ig_local_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}
                        val_local_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}

                        ig_local_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        val_local_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                    if (index2 == index3) and (index2 == 0) and (index == 0):
                        ig_local_tables[model] = ""
                        val_local_tables[model] = ""
                        print('here')
                        # global one
                        ig_global_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}
                        val_global_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}

                        ig_global_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        val_global_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        # print(ig_local_details_metrics_depth_1)

                    # assuming that our store structore now have a financial metric section, we need to ensure that the current result folder
                    # has a financial dimension
                    hasFinancialCost = sum(['finan' in el for el in classic_f.columns.values.tolist()])
                    if hasFinancialCost == 0:
                        # if there is not financial details, remove the section in ours store
                        # for mod in list(ig_local_details_metrics_depth_1.keys()): 
                        # print(index, index2, index3, index4)
                        if 'financial-cost' in list(ig_local_details_metrics_depth_1[model][result_folder].keys()):
                            # local
                            del ig_local_details_metrics_depth_1[model][result_folder]['financial-cost'] 
                            del val_local_details_metrics_depth_1[model][result_folder]['financial-cost']
                            del ig_local_details_metrics_depth_2[model][result_folder]['financial-cost']
                            del val_local_details_metrics_depth_2[model][result_folder]['financial-cost']
                            if (index3 == 0) and (index == 0):
                                # global
                                del ig_global_details_metrics_depth_1[model][result_folder]['financial-cost']
                                del val_global_details_metrics_depth_1[model][result_folder]['financial-cost']
                                del ig_global_details_metrics_depth_2[model][result_folder]['financial-cost']
                                del val_global_details_metrics_depth_2[model][result_folder]['financial-cost']

                    val_local_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    ig_local_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    val_local_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    ig_local_details_metrics_depth_2[model][result_folder]['classic'] = classic_f

                    val_global_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    ig_global_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    val_global_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    ig_global_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    # now fetch ours results to store
                    for metric in metrics: # accuracy and f1-score and/or financial cost
                        for approach in list(ig_local_details_metrics_depth_1[model][result_folder][metric].keys()): # Mlc or MCA
                            # print(metric, approach)
                            for logic in list(ig_local_details_metrics_depth_1[model][result_folder][metric][approach].keys()): # GLO, PER or GAP
                                for config in list(ig_local_details_metrics_depth_1[model][result_folder][metric][approach][logic].keys()): # MX, CX, CY, CXY
                                    for result in list(range(len(files[approach][logic][config]))): # each result file's containing evaluation metrics
                                        # save exact metric values
                                        val_local_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],4))
                                        val_global_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],4))

                                        val_local_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],4))
                                        val_global_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],4))

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
                                                    round(
                                                        classic_f.loc[model, metric], 
                                                        4
                                                        )
                                                    ) * 100, 
                                                1
                                                )
                                        ig =  valu if 'finan' not in metric else  (valu if valu ==0 else -1*valu) 
                                        # save just gain information
                                        ig_local_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(ig)
                                        ig_global_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(ig)

                                        ig_local_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(ig)
                                        ig_global_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(ig)
                            
                                        if 'accu' in metric and 'MX' in config:
                                            list_of_accuracy.append((layer, model,files[approach][logic][config][result].loc[model, metric]))

            # analyse impact of layers and identify the best mlna as k
            best_mlna_k_per_alpha[result_folder][alpha]['list']= list_of_accuracy
            list_of_accuracy = sorted(list_of_accuracy, key=lambda x: abs(x[2]), reverse=False) # best will be at position 0
            best_mlna_k_per_alpha[result_folder][alpha]['real_best_k'].append(list_of_accuracy[0][0])
            best_mlna_k_per_alpha[result_folder][alpha]['model'].append(list_of_accuracy[0][1])
            best_mlna_k_per_alpha[result_folder][alpha]['value'].append(list_of_accuracy[0][2])

        ## on store, call a print table function (gain, metric)
        # print(ig_local_details_metrics_depth_1)
        # return
        ig_local_tables = {key: ig_local_tables[key] + "\n" + mod for key, mod in print_compare_v3_1(
            ig_local_details_metrics_depth_1,
            f'{cwd}/{outputPath}/ig',
            alpha,
            metrics,
            _totalConfigs
        ).items()}
        val_local_tables = {key: val_local_tables[key] + "\n" + mod for key, mod in print_compare_v3_1(
            val_local_details_metrics_depth_1,
            f'{cwd}/{outputPath}/val',
            alpha,
            metrics,
            _totalConfigs,
            True
        ).items()}

    ## print the global container (gain, metric)
    ig_global_tables = "\n \\vspace{0.02cm} \\vspace{0.02cm}".join([ mod for _, mod in print_compare_v3_1(
        ig_global_details_metrics_depth_1,
        f'{cwd}/{outputPath}/ig',
        'all',
        metrics,
        _totalConfigs
    ).items()])

    val_global_tables = "\n \\vspace{0.02cm} \\vspace{0.02cm}".join([ mod for _, mod in print_compare_v3_1(
        val_global_details_metrics_depth_1,
        f'{cwd}/{outputPath}/val',
        'all',
        metrics,
        _totalConfigs,
        True
    ).items()])
    
    create_domain(f'{cwd}/{outputPath}/ig/all_ig/')
    with open(f'{cwd}/{outputPath}/ig/all_ig/ig_all.tex', "w") as fichier:
        fichier.write(header+"""
                \\begin{figure}[H]
                \\begin{center}"""+ig_global_tables+"""
                \\caption{default}
                \\label{default}
                \\end{center}
                \\end{figure}"""+footer)
    create_domain(f'{cwd}/{outputPath}/val/val_all/')
    with open(f'{cwd}/{outputPath}/val/val_all/val_all.tex', "w") as fichier:
        fichier.write(header+"""
                \\begin{figure}[H]
                \\begin{center}"""+val_global_tables+"""
                \\caption{default}
                \\label{default}
                \\end{center}
                \\end{figure}"""+footer)

    for key in list(ig_local_tables.keys()):
        create_domain(f'{cwd}/{outputPath}/ig/{key}/')
        with open(f'{cwd}/{outputPath}/ig/{key}/{key}.tex', "a") as fichier:
            fichier.write(header+ig_local_tables[key]+footer)
        create_domain(f'{cwd}/{outputPath}/val/{key}/')
        with open(f'{cwd}/{outputPath}/val/{key}/{key}.tex', "a") as fichier:
            fichier.write(header+val_local_tables[key]+footer)

    joblib.dump(best_mlna_k_per_alpha, f'{cwd}/{outputPath}/best.tex')
    print_compare_k_att_selection_v2(
        best_mlna_k_per_alpha,
        f'{cwd}/{outputPath}/descriptComp',
    ) 
    print_compare_k_att_selection(
        best_mlna_k_per_alpha,
        f'{cwd}/{outputPath}/descriptComp',
    ) 


def generate_report_tables_v3_2(
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
    metrics=['accuracy','f1-score','precision1','recall1','f1-score1','precision0','recall0','f1-score0','financial-cost'],
    _type='qualitative',
    alphas=[0.85],
    result_=[1]
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
    day = time.strftime("%Y_%m_%d_%H")
    ## identify the name of results folder
    data_result_folder_name = [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{outputs_name}')][0]
    print(data_result_folder_name)
    print(metrics)
    ## init a global container for containing the results of all results folder
    template_details_metrics_depth_1 = {}
    template_details_metrics_depth_2 = {}
    _totalConfigs = 0
    for i,metric in enumerate(metrics):
        if not(metric in list(template_details_metrics_depth_1.keys())): # it's metric already exist in the container
            template_details_metrics_depth_1[metric] = {}
            template_details_metrics_depth_2[metric] = {}
        for j,approach in enumerate(approachs):
            if not(approach in list(template_details_metrics_depth_1[metric].keys())): # it's approach already exist in the container
                template_details_metrics_depth_1[metric][approach] = {}
                template_details_metrics_depth_2[metric][approach] = {}
            for logic in logics:
                if not(logic in list(template_details_metrics_depth_1[metric][approach].keys())): # it's logic already exist in the container
                    template_details_metrics_depth_1[metric][approach][logic] = {}
                    template_details_metrics_depth_2[metric][approach][logic] = []
                for config in configs:
                    if logic == 'GLO':
                        if config in ['MX','CX']:
                            template_details_metrics_depth_1[metric][approach][logic][config] = []
                            if i == 0 and j == 0:
                                _totalConfigs += 1
                    else:
                        template_details_metrics_depth_1[metric][approach][logic][config] = []
                        if i == 0 and j == 0:
                            _totalConfigs += 1

    ######################################
    ig_global_details_metrics_depth_1 = {
    }
    val_global_details_metrics_depth_1 = {
    }

    ig_global_details_metrics_depth_2 = {
    }
    val_global_details_metrics_depth_2 = {
    }

    best_mlna_k_per_alpha = {
        folder: {key: {'real_best_k': [], 'predicted_best_k':[], 'value':[], 'model':[]} for key in alphas} for folder in data_result_folder_name
    }
    #####################################
    ig_global_tables = ""
    val_global_tables = ""
    ig_local_tables = {fol: "" for fol in data_result_folder_name}
    val_local_tables = {fol: "" for fol in data_result_folder_name}
    ## fetch on alpha values
    for index, alpha in enumerate(alphas):
        ## for each alpha, store a local container 
        ig_local_details_metrics_depth_1 = {
        }
        val_local_details_metrics_depth_1 = {
        }
        ig_local_details_metrics_depth_2 = {
        }
        val_local_details_metrics_depth_2 = {
        }

        local_best_mlna_k_per_alpha = {
            key: deepcopy(best_mlna_k_per_alpha) for key in data_result_folder_name
        }

        
        ## fetch on name of folders
        for index2, result_folder in enumerate(data_result_folder_name):
            # load predicted best mlnk
            print(result_folder)
            name = \
                [file for _, _, files in os.walk(
                    f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage')
                 for file in files if
                 '_best_features' in file][0]
            best_mlna_k_per_alpha[result_folder][alpha]['predicted_best_k'].append(read_model(f'{cwd}/{outputs_name}/{result_folder}/{alpha}/{_type}/model_storage/{name}')["bestK"])
            ## get classic model
            classic_f = [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{os.getcwd()}/{outputs_name}/{result_folder}/data_selection_storage',
                    func=C_F,
                    verbose=False
                )
            ][-1]
            # define a list which will receive all possible value of accuracy for each layer in aims tp sorted it later and extra, the best best layer and model
            list_of_accuracy = []
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
                    mix=True
                )
                ## aggregate results belong to metrics specific to the current results
                for index4, model in enumerate(models_name):
                    # print(model)  
                    # append contains structured to our store
                    if (index2 == index3) and (index2 == 0): # if first alpha, first folder result, and first layer

                        # local ones
                        ig_local_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}
                        val_local_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}

                        ig_local_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        val_local_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                    if (index2 == index3) and (index2 == 0) and (index == 0):
                        # ig_local_tables[model] = ""
                        # val_local_tables[model] = ""
                        # print('here')
                        # global one
                        ig_global_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}
                        val_global_details_metrics_depth_1[model] = {key: deepcopy(template_details_metrics_depth_1) for key in data_result_folder_name}

                        ig_global_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        val_global_details_metrics_depth_2[model] = {key: deepcopy(template_details_metrics_depth_2) for key in data_result_folder_name}
                        # print(ig_local_details_metrics_depth_1)

                    # assuming that our store structore now have a financial metric section, we need to ensure that the current result folder
                    # has a financial dimension
                    hasFinancialCost = sum(['finan' in el for el in classic_f.columns.values.tolist()])
                    if hasFinancialCost == 0:
                        # if there is not financial details, remove the section in ours store
                        # for mod in list(ig_local_details_metrics_depth_1.keys()): 
                        # print(index, index2, index3, index4)
                        if 'financial-cost' in list(ig_local_details_metrics_depth_1[model][result_folder].keys()):
                            # local
                            del ig_local_details_metrics_depth_1[model][result_folder]['financial-cost'] 
                            del val_local_details_metrics_depth_1[model][result_folder]['financial-cost']
                            del ig_local_details_metrics_depth_2[model][result_folder]['financial-cost']
                            del val_local_details_metrics_depth_2[model][result_folder]['financial-cost']
                            if (index3 == 0) and (index == 0):
                                # global
                                del ig_global_details_metrics_depth_1[model][result_folder]['financial-cost']
                                del val_global_details_metrics_depth_1[model][result_folder]['financial-cost']
                                del ig_global_details_metrics_depth_2[model][result_folder]['financial-cost']
                                del val_global_details_metrics_depth_2[model][result_folder]['financial-cost']

                    val_local_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    ig_local_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    val_local_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    ig_local_details_metrics_depth_2[model][result_folder]['classic'] = classic_f

                    val_global_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    ig_global_details_metrics_depth_1[model][result_folder]['classic'] = classic_f
                    val_global_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    ig_global_details_metrics_depth_2[model][result_folder]['classic'] = classic_f
                    # now fetch ours results to store
                    for metric in metrics: # accuracy and f1-score and/or financial cost
                        for approach in list(ig_local_details_metrics_depth_1[model][result_folder][metric].keys()): # Mlc or MCA
                            # print(metric, approach)
                            for logic in list(ig_local_details_metrics_depth_1[model][result_folder][metric][approach].keys()): # GLO, PER or GAP
                                for config in list(ig_local_details_metrics_depth_1[model][result_folder][metric][approach][logic].keys()): # MX, CX, CY, CXY
                                    for result in list(range(len(files[approach][logic][config]))): # each result file's containing evaluation metrics
                                        # save exact metric values
                                        val_local_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],4))
                                        val_global_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],4))

                                        val_local_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],4))
                                        val_global_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],4))

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
                                                    round(
                                                        classic_f.loc[model, metric], 
                                                        4
                                                        )
                                                    ) * 100, 
                                                1
                                                )
                                        ig =  valu if 'finan' not in metric else  (valu if valu ==0 else -1*valu) 
                                        # save just gain information
                                        ig_local_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(ig)
                                        ig_global_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(ig)

                                        ig_local_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(ig)
                                        ig_global_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(ig)
                            
                                        if 'accu' in metric and 'MX' in config:
                                            list_of_accuracy.append((layer, model,files[approach][logic][config][result].loc[model, metric]))

            # analyse impact of layers and identify the best mlna as k
            best_mlna_k_per_alpha[result_folder][alpha]['list']= list_of_accuracy
            list_of_accuracy = sorted(list_of_accuracy, key=lambda x: abs(x[2]), reverse=False) # best will be at position 0
            best_mlna_k_per_alpha[result_folder][alpha]['real_best_k'].append(list_of_accuracy[0][0])
            best_mlna_k_per_alpha[result_folder][alpha]['model'].append(list_of_accuracy[0][1])
            best_mlna_k_per_alpha[result_folder][alpha]['value'].append(list_of_accuracy[0][2])

        ## on store, call a print table function (gain, metric)
        # print(ig_local_details_metrics_depth_1)
        # return
        ig_local_tables = {key: ig_local_tables[key] + "\n" + mod for key, mod in print_compare_v3_2(
            ig_local_details_metrics_depth_1,
            f'{cwd}/{outputPath}/ig',
            alpha,
            metrics,
            _totalConfigs,
            result_
        ).items()}
        val_local_tables = {key: val_local_tables[key] + "\n" + mod for key, mod in print_compare_v3_2(
            val_local_details_metrics_depth_1,
            f'{cwd}/{outputPath}/val',
            alpha,
            metrics,
            _totalConfigs,
            result_,
            True
        ).items()}

    ## print the global container (gain, metric)
    ig_global_tables = "\n \\vspace{0.02cm} \\vspace{0.02cm}".join([ mod for _, mod in print_compare_v3_2(
        ig_global_details_metrics_depth_1,
        f'{cwd}/{outputPath}/ig',
        'all',
        metrics,
        _totalConfigs,
        result_
    ).items()])

    val_global_tables = "\n \\vspace{0.02cm} \\vspace{0.02cm}".join([ mod for _, mod in print_compare_v3_2(
        val_global_details_metrics_depth_1,
        f'{cwd}/{outputPath}/val',
        'all',
        metrics,
        _totalConfigs,
        result_,
        True
    ).items()])
    
    create_domain(f'{cwd}/{outputPath}/ig/all_ig/')
    with open(f'{cwd}/{outputPath}/ig/all_ig/ig_all.tex', "w") as fichier:
        fichier.write(header+"""
                \\begin{figure}[H]
                \\begin{center}"""+ig_global_tables+"""
                \\caption{default}
                \\label{default}
                \\end{center}
                \\end{figure}"""+footer)
    create_domain(f'{cwd}/{outputPath}/val/val_all/')
    with open(f'{cwd}/{outputPath}/val/val_all/val_all.tex', "w") as fichier:
        fichier.write(header+"""
                \\begin{figure}[H]
                \\begin{center}"""+val_global_tables+"""
                \\caption{default}
                \\label{default}
                \\end{center}
                \\end{figure}"""+footer)

    for key in list(ig_local_tables.keys()):
        create_domain(f'{cwd}/{outputPath}/ig/{key}/')
        with open(f'{cwd}/{outputPath}/ig/{key}/{key}.tex', "a") as fichier:
            fichier.write(header+ig_local_tables[key]+footer)
        create_domain(f'{cwd}/{outputPath}/val/{key}/')
        with open(f'{cwd}/{outputPath}/val/{key}/{key}.tex', "a") as fichier:
            fichier.write(header+val_local_tables[key]+footer)

    joblib.dump(best_mlna_k_per_alpha, f'{cwd}/{outputPath}/best.tex')
    print_compare_k_att_selection_v2(
        best_mlna_k_per_alpha,
        f'{cwd}/{outputPath}/descriptComp',
    ) 
    print_compare_k_att_selection(
        best_mlna_k_per_alpha,
        f'{cwd}/{outputPath}/descriptComp',
    ) 

"""
    Impact performance analytic for MLN1 and MLN2
    Q: How to select features used to build the graph?

    The goal here is to count the number of times where distingiush couple of performance (G,B) in MLN1 improve in MLN2
"""
def generate_g_b_impact_table(
    outputs_name=None, 
    cwd=None, 
    layers=[1], 
    _type='qualitative',
    alphas=[0.85]
):
    day = time.strftime("%Y_%m_%d_%H") # actual datetime
    data_result_folder_name = [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{outputs_name}')][0] # result folder name
    
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
                    func=C_F,
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
                        for logic in ['GLO','PER']:
                            # Loop on config
                            for config in ['MX','CX']:
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
                                                metrics_impact[metric][approach][logic][indice]["s"] += 1 if round(files[approach][logic][config][file].loc[model,metric],4) < round(classic_f.loc[model,metric],4) else 0
                                                metrics_impact[metric][approach][logic][indice]["e"] += 1 if round(files[approach][logic][config][file].loc[model,metric],4) == round(classic_f.loc[model,metric],4) else 0
                                                metrics_impact[metric][approach][logic][indice]["i"] += 1 if round(files[approach][logic][config][file].loc[model,metric],4) > round(classic_f.loc[model,metric],4) else 0
    # call the printer function
    print_g_b_impact_table(
        metrics_impact,
        f'{cwd}/analyzeV1',
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
    alpha,
    top
):
    # fetch model name
    tables = {model:'' for model in list(store.keys())}
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
            # sort the dict
            data = dict(sorted(store[model][folder].items(), key=lambda x: abs(x[1]), reverse=True)[:top])
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
                color = "green" if GLO_F(key) else ("yellow" if Y_F(key) else ("red" if PER_F(key) else "blue"))
                bchart += "\\bcbar[value=, label="+str(key).replace('_','\\_')+",color="+color+"]{"+f"{val:.{6}f}"+"""}
                    \\smallskip
                    """
            bchart += ("\\bcxlabel{"+str(folder).replace('_','\\_')+"""}
                            \\end{bchart}
                            """) if i != len(store[model].keys())-1 else ("\\bcxlabel{"+str(folder).replace('_','\\_')+"""}
                            \\bclegend{5pt}{blue/Classic, green/GLO, red/PER, yellow/Y\\_PER}
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


def print_compare_k_att_selection(
    store,
    output_path,
):
    

    folders = list(store.keys())
    gplot = "%\\begin{tabular}{"+("c"*len(store[folders[0]].keys()))+"""}
    """
    # fetch alpha
    for i,alpha in enumerate(store[folders[0]].keys()):
        # define lines plots form
        real_line = """\\addplot table[x=dataset,y=k] {
            dataset k
            """
        predicted_line = """\\addplot table[x=dataset,y=k] {
            dataset k
            """

        plot = """\\begin{tikzpicture}
            \\begin{axis}[
                xlabel=dataset,
                ylabel={k},
                symbolic x coords={"""+str(",".join(folders)).replace('_','')+"""}, % Define symbolic x labels
                xtick=data, % Get x tick marks from data
                x tick label style={% Change x tick label style
                    /pgf/number format/set thousands separator={}%
                },
                y tick label style={% Change y tick label style
                    /pgf/number format/fixed, % Use fixed-point notation
                    /pgf/number format/precision=3, % Precision to 3 decimal places
                    %/pgf/number format/fixed zerofill % Fill with zeros if needed
                },
                title={K-Features selection plot for alpha """+str(alpha)+"""},% Plot title
                tick align=outside, % Ticks on the outside
                enlargelimits = upper,
                %xmin=0, % Ensure x-axis starts at 0
                %ymin=0, % Ensure y-axis starts at 0
                legend pos = outer north east
            ]
            """
        # fetch folder
        for key, folder in store.items():
            # add line content
            real_line += str(key).replace("_","")+f""" {store[key][alpha]["real_best_k"][0]}
                """

            predicted_line += str(key).replace("_","")+f""" {store[key][alpha]["predicted_best_k"][0]}
                """

        plot+= real_line+"""};
            \\addlegendentry{real}
            """ 
        plot+= predicted_line+"""};
            \\addlegendentry{predicted}
            \\end{axis}
            \\end{tikzpicture}
            """ 
        gplot+=(plot+"""
                    
                    """) if i != len(store[folders[0]].keys())-1 else (plot+"""
            %\\end{tabular}
            }""")
        
    create_domain(f"{output_path}/selection/k")
    filename1 = f"{output_path}/selection/k/plots.tex"
    _file = open(filename1, "w")
    _file.write(plot_header+gplot+footer)
    _file.close()


def print_compare_k_att_selection_v2(
    store,
    output_path,
):
    

    folders = list(store.keys())
    gplot = "\\begin{tabular}{"+("c"*(len(store[folders[0]].keys())//2))+"""}
    """
    # fetch alpha
    for j,(key, folder) in enumerate(store.items()):
        # define lines plots form
        lines = """\\addplot table[x=k,y=Accuracy,row sep=crcr] {
            k Accuracy
        """

        plot = """
            \\begin{tikzpicture}
            \\begin{axis}[
                xlabel=k,
                ylabel={Accuracy},
                % symbolic x coords={"""+str(",".join(key)).replace('_','')+"""}, % Define symbolic x labels
                xtick=data, % Get x tick marks from data
                x tick label style={% Change x tick label style
                    /pgf/number format/set thousands separator={}%
                },
                y tick label style={% Change y tick label style
                    /pgf/number format/fixed, % Use fixed-point notation
                    /pgf/number format/precision=3, % Precision to 3 decimal places
                    %/pgf/number format/fixed zerofill % Fill with zeros if needed
                },
                title={Impact of k best features combinaison on  """+str(key).replace('_','\\_')+"""},% Plot title
                tick align=outside, % Ticks on the outside
                enlargelimits = upper,
                xmin=0, % Ensure x-axis starts at 0
                %ymin=0, % Ensure y-axis starts at 0
                legend pos = outer north east
            ]
        """
        # fetch alpha
        for i,alpha in enumerate(store[key].keys()):
            # fetch results
            plot +=lines
            # print(store[key][alpha]["list"])
            lay = {}
            for (layer, _, acc) in store[key][alpha]["list"]:
                if not(layer in list(lay.keys())):
                    lay[layer] = acc
                else:
                    lay[layer] = max(lay[layer],acc)
            for (layer, acc) in lay.items():
                # add line content
                plot += f"""{layer} {acc} \\\\
                """

            plot+= """};
                \\addlegendentry{$\\alpha = """+str(alpha)+"""$}
            """ 

        gplot+=(plot+"""
                \\end{axis}
                \\end{tikzpicture}
                &
                    """) if (j+1)%2 != 0 else (plot+"""
                \\end{axis}
                \\end{tikzpicture}
                \\\\
                    """) 
    gplot+="""\\end{tabular}}
            

                """
        
    create_domain(f"{output_path}/selection/perf")
    filename1 = f"{output_path}/selection/perf/plots_perf.tex"
    _file = open(filename1, "w")
    _file.write(plot_header+gplot+footer)
    _file.close()




# ################################################################
# ############## BarPlots Generation Logic Functions #############
# ################################################################
def generate_descriptor_ranking(
    outputs_name=None, 
    cwd=None, 
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
                    func=C_F,
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
                    mix=True
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
                                        'YN_INTER_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif YP_PER_INTER_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'YP_INTER_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif YN_PER_INTRA_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'YN_INTRA_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif YP_PER_INTRA_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'YP_INTRA_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif YN_PER_COMBINE_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'YN_COMBINE_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif YP_PER_COMBINE_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'YP_COMBINE_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif GLO_INTER_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTER_GLO'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif GLO_INTRA_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTRA_GLO'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif GLO_COMBINE_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'COMBINE_GLO'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif PER_INTER_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTER_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif PER_INTRA_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTRA_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif PER_COMBINE_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'COMBINE_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif GLO_INTER_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTER_M_GLO'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif GLO_INTRA_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTRA_M_GLO'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif GLO_COMBINE_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'COMBINE_M_GLO'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif PER_INTER_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTER_M_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif PER_INTRA_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'INTRA_M_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif PER_COMBINE_M_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'COMBINE_M_PER'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                elif DEGREE_F(att):
                                    global_details_metrics_depth_1[model][result_folder][
                                        'DEGREE'].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])
                                else:
                                    if not (att in global_details_metrics_depth_1[model][result_folder].keys()):
                                        global_details_metrics_depth_1[model][result_folder][
                                            att] = []
                                        # print(global_details_metrics_depth_1[model][result_folder])
                                    global_details_metrics_depth_1[model][result_folder][
                                        att].append(files["MlC"]["GAP"]["CXY"][result].loc[model, att])


    for model in global_details_metrics_depth_1.keys():
        for folder in global_details_metrics_depth_1[model].keys():
            for att in global_details_metrics_depth_1[model][folder].keys():
                global_details_metrics_depth_1[model][folder][att] = statistics.mean(global_details_metrics_depth_1[model][folder][att])
            # Calculer la somme de toutes les valeurs
            total_sum = sum(global_details_metrics_depth_1[model][folder].values())

            # Normaliser les valeurs
            global_details_metrics_depth_1[model][folder] = {k: v / total_sum for k, v in global_details_metrics_depth_1[model][folder].items()}
    ## print the global container (gain, metric)
    print_compare_bchart(
        global_details_metrics_depth_1,
        f'{cwd}/descriptComp/',
        "all",
        20
    )
    
        


























