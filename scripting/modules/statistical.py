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
    \\usepackage{pgfplots}
    \\usepackage[utf8]{inputenc}
    \\usepackage[T1]{fontenc}
    \\usepackage{fancyhdr}
    \\pagestyle{fancy}
    \\usepackage{multirow}
    \\usepackage{hyperref}
    \\usepackage[babel]{csquotes}

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

footer = """
\\end{document}  """

############# End declaration
########################################################

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
    var.split("__")[1] for var in [
        coll
        for coll in [
            col
            for col in x
            if not (
                    ('precision' in col)
                    or ('accuracy' in col)
                    or ('recall' in col)
                    or ('f1-score' in col)
            )
        ]
        if ("__" in coll)
    ]
]
)))

"""
    Lambda Function
    Get all quantitative features from operation results
    During the process, we specify the partern '___' like an indicator
"""
get_quantitative_from_cols = lambda x: (list(set([
    var.split("___")[1] for var in [  # ___
        coll
        for coll in [
            col
            for col in x
            if not (
                    ('precision' in col)
                    or ('accuracy' in col)
                    or ('recall' in col)
                    or ('f1-score' in col)
            )
        ]
        if ("__" in coll)
    ]
]
)))




def print_compare(
    store,
    output_path,
    alpha
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
            lines+= "& Accuracy & F1-Score & Financial-cost"
        # add metrics' for non financial folder
        for folder in financial_folder:
            lines+=  "& Accuracy & F1-Score"
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
                lines+= "& "+str(round(store[model][folder]['classic'].loc[model, metric],1))+""
        # add cols for non financial folder
        for folder in none_financial_folder:
            for metric in ['accuracy','f1-score']:
                lines+= "& "+str(round(store[model][folder]['classic'].loc[model, metric],1))+""
        # add an empty cell for Total
        lines+= "& \\\\ \\cline{1-"+str(nb_cols)+"""}

        """

        is_best = lambda mag,current: (max([max(mag[el]) for el in list(mag.keys())]) == current) and (current != 0)
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
                            # print(store[model][folder].keys(), folder, model)
                            is_sup = is_best(store[model][folder][metric][approach][logic], max(store[model][folder][metric][approach][logic][config]))
                            val = "\\textbf{"+str(max(store[model][folder][metric][approach][logic][config]))+"}" if is_sup else str(max(store[model][folder][metric][approach][logic][config]))
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



"""
    Here we start the code of new type of result table structures.
"""

def load_results(
        outputs_path,
        _type,
        k,
        alpha,
        per=True,
        glo=True,
        mix=True,

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

    Returns
    -------

    """
    files = {
        'MlC':{
            'GLO':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/global/data_selection_storage',
                        func=lambda x: (MlC_F(x)),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/global/data_selection_storage',
                        func=lambda x: (MlC_F(x)),
                        verbose=False
                    )
                ]
            },
            'PER':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/personalized/data_selection_storage',
                        func=lambda x: (MlC_F(x)),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/personalized/data_selection_storage',
                        func=lambda x: (MlC_F(x)),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/personalized/data_selection_storage',
                        func=lambda x: (MlC_F(x)),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/personalized/data_selection_storage',
                        func=lambda x: (MlC_F(x)),
                        verbose=False
                    )
                ]
            },
            'GAP':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage',
                        func=lambda x: (MlC_F(x)),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage',
                        func=lambda x: (MlC_F(x)),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage',
                        func=lambda x: (MlC_F(x)),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage',
                        func=lambda x: (MlC_F(x)),
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
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/global/data_selection_storage',
                        func=lambda x: (MCA_F(x)),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/global/data_selection_storage',
                        func=lambda x: (MCA_F(x)),
                        verbose=False
                    )
                ]
            } if glo is True else None,
            'PER':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/personalized/data_selection_storage',
                        func=lambda x: (MCA_F(x)),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/personalized/data_selection_storage',
                        func=lambda x: (MCA_F(x)),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/personalized/data_selection_storage',
                        func=lambda x: (MCA_F(x)),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/personalized/data_selection_storage',
                        func=lambda x: (MCA_F(x)),
                        verbose=False
                    )
                ]
            } if per is True else None,
            'GAP':{
                'MX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage',
                        func=lambda x: (MCA_F(x)),
                        verbose=False
                    )
                ],
                'CX': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage',
                        func=lambda x: (MCA_F(x)),
                        verbose=False
                    )
                ],
                'CY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage',
                        func=lambda x: (MCA_F(x)),
                        verbose=False
                    )
                ],
                'CXY': [
                    load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                    for file in get_filenames(
                        root_dir=f'{outputs_path}/withClass/{alpha}/{_type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage',
                        func=lambda x: (MCA_F(x)),
                        verbose=False
                    )
                ]
            } if mix is True else None,
        }
    }
    return files


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
                },
                'GLO': {
                    'MX': [],
                    'CX': [],
                }
            },
            'MCA': {
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
                },
                'GLO': {
                    'MX': [],
                    'CX': [],
                }
            }
        },
        'f1-score': {
            'MlC': {
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
                },
                'GLO': {
                    'MX': [],
                    'CX': [],
                }
            },
            'MCA': {
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
                },
                'GLO': {
                    'MX': [],
                    'CX': [],
                }
            }
        },
        'financial-cost': {
            'MlC': {
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
                },
                'GLO': {
                    'MX': [],
                    'CX': [],
                }
            },
            'MCA': {
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
                },
                'GLO': {
                    'MX': [],
                    'CX': [],
                }
            }
        }
    }

    template_details_metrics_depth_2 = {
        'accuracy': {
            'MlC': {
                'PER': [],
                'GAP': [],
                'GLO': []
            },
            'MCA': {
                'PER': [],
                'GAP': [],
                'GLO': []
            }

        },
        'f1-score': {
            'MlC': {
                'PER': [],
                'GAP': [],
                'GLO': []
            },
            'MCA': {
                'PER': [],
                'GAP': [],
                'GLO': []
            }
        },
        'financial-cost': {
            'MlC': {
                'PER': [],
                'GAP': [],
                'GLO': []
            },
            'MCA': {
                'PER': [],
                'GAP': [],
                'GLO': []
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
                                        val_local_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],1))
                                        val_global_details_metrics_depth_1[model][result_folder][metric][approach][logic][config].append(round(files[approach][logic][config][result].loc[model, metric],1))

                                        val_local_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],1))
                                        val_global_details_metrics_depth_2[model][result_folder][metric][approach][logic].append(round(files[approach][logic][config][result].loc[model, metric],1))

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
        ig_local_tables = {key: ig_local_tables[key] + "\n" + mod for key, mod in print_compare(
            ig_local_details_metrics_depth_1,
            f'{cwd}/analyze/ig',
            alpha
        ).items()}
        val_local_tables = {key: val_local_tables[key] + "\n" + mod for key, mod in print_compare(
            val_local_details_metrics_depth_1,
            f'{cwd}/analyze/val',
            alpha
        ).items()}

    ## print the global container (gain, metric)
    ig_global_tables = "\n".join([ mod for _, mod in print_compare(
        ig_global_details_metrics_depth_1,
        f'{cwd}/analyze/ig',
        'all'
    ).items()])
    val_global_tables = "\n".join([ mod for _, mod in print_compare(
        val_global_details_metrics_depth_1,
        f'{cwd}/analyze/val',
        'all'
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

        









