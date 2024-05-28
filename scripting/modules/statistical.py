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

########### End Module
################################################

################################################
######## Constant declaration

MlC_F = lambda x: (('MlC_' in x))  # find metric of model where mln were added to classic Att
MCA_F = lambda x: (('MCA_' in x))  # where mln attribut were removed first and mln were added to the rest
GAP_F = lambda x: (('GAP_' in x))  # where mln attribut were removed first and mln were added to the rest
C_F = lambda x: (('classic_' in x))  # classic metrics

GLO_MX_F = lambda x: (('GLO_MX_' in x))
PER_MX_F = lambda x: (('PER_MX_' in x))
PER_MY_F = lambda x: (('PER_MY_' in x))
PER_MXY_F = lambda x: (('PER_MXY_' in x))
GAP_MX_F = lambda x: (('GAP_MX_' in x))
GAP_MY_F = lambda x: (('GAP_MY_' in x))
GAP_MXY_F = lambda x: (('GAP_MXY_' in x))

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


def analyzer_launcher(outputs_name=None, analytical_func=None, layers=None, approach=None, aggregation_f=None,
                      logics=None, alphaConfig=[0.1]):
    """
    Analyzer Launcher
        Args:
            outputs_name (string): Name of output file
            analytical_func (function): Function that takes one argument
            layers (list): List of layers to analyse
            approach (function): Function that takes one argument
            aggregation_f (function): Function that takes one argument
            logics (list): List of logics to analyse
            alphaConfig (list): List of alpha configs to analyse
        Returns
            None
    """
    # get all directories names
    result_folders = [dirnames for _, dirnames, _ in os.walk(f'{os.getcwd()}/{outputs_name}')][0]
    # iterate on different configuration of alpha

    # fetch each datasets directories
    for dataset_name in result_folders:
        # load the classic results
        classic_f = [
            load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
            for file in get_filenames(
                root_dir=f'{os.getcwd()}/{outputs_name}/{dataset_name}/data_selection_storage',
                func=C_F,
                verbose=False
            )
        ][-1]
        # from this, extract quantitative and qualitative features
        quali_col = get_qualitative_from_cols(classic_f.columns.to_list())
        # quant_col = get_quantitative_from_cols(classic_f.columns.to_list())
        # get model list on classic results
        models_list = classic_f.index.values.tolist()
        # get model dictionary
        models = model_desc()
        # save only the ones use suring the classic learning
        models_name = {key: models[key] for key in models.keys() if key in models_list}
        # check whether a verison of training has been made using classe variable along graph modeling
        with_class = [dirnames for _, dirnames, _ in
                     os.walk(f'{os.getcwd()}/{outputs_name}/{dataset_name}')][0].__contains__("withClass")
        # if, launch a part analysis on this folder
        print(with_class)
        for i in alphaConfig:
            for typ2 in [dirnames for _, dirnames, _ in
                        os.walk(f'{os.getcwd()}/{outputs_name}/{dataset_name}/{i}')][0]:
                print(typ2)
                analytical_func(
                    outputs_path=f'{os.getcwd()}/{outputs_name}/{dataset_name}/{i}',
                    cwd=os.getcwd(),
                    data_folder=dataset_name,
                    classic_metrics=classic_f,
                    models_name=models_name,
                    layers=layers,
                    approach=approach,
                    alpha=i,
                    type=typ2
                ) if aggregation_f is None else analytical_func(
                    outputs_path=f'{os.getcwd()}/{outputs_name}/{dataset_name}/{i}',
                    cwd=os.getcwd(),
                    data_folder=dataset_name,
                    classic_metrics=classic_f,
                    models_name=models_name,
                    layers=layers if not (layers is None) else (
                        list({1, 2, len(quali_col)}) if ('qualitative' in typ2) else list(
                            {1, 2, len(quant_col)})),
                    approach=approach,
                    aggregation_f=aggregation_f,
                    alpha=i,
                    logics=logics,
                    type=typ2
                )

        if with_class:
            for i in alphaConfig:
                for typ1 in [dirnames for _, dirnames, _ in
                            os.walk(f'{os.getcwd()}/{outputs_name}/{dataset_name}/withClass/{i}')][0]:

                    print(typ1)
                    analytical_func(
                        # cols=quali_col if ('qualitative' in typ1) else quant_col,
                        outputs_path=f'{os.getcwd()}/{outputs_name}/{dataset_name}/withClass/{i}',
                        cwd=os.getcwd(),
                        data_folder=dataset_name,
                        classic_metrics=classic_f,
                        models_name=models_name,
                        layers=layers,
                        approach=approach,
                        alpha=i,
                        type=typ1
                    ) if aggregation_f is None else analytical_func(
                        # cols=quali_col if ('qualitative' in typ1) else quant_col,
                        outputs_path=f'{os.getcwd()}/{outputs_name}/{dataset_name}/withClass/{i}',
                        cwd=os.getcwd(),
                        data_folder=dataset_name,
                        classic_metrics=classic_f,
                        models_name=models_name,
                        layers=layers if not (layers is None) else (
                            list({1, 2, len(quali_col)}) if ('qualitative' in typ1) else list(
                                {1, 2, len(quant_col)})),
                        approach=approach,
                        aggregation_f=aggregation_f,
                        alpha=i,
                        logics=logics,
                        type=typ1
                    )


def load_results(
        outputs_path,
        type,
        k,
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
        'global': {
            'MlC': [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{outputs_path}/{type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/global/data_selection_storage',
                    func=lambda x: (MlC_F(x)),
                    verbose=False
                )
            ],
            'MCA': [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{outputs_path}/{type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/global/data_selection_storage',
                    func=lambda x: (MCA_F(x)),
                    verbose=False
                )
            ]
        } if glo is True else None,
        'mixed': {
            'MlC': [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{outputs_path}/{type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage',
                    func=lambda x: (MlC_F(x)),
                    verbose=False
                )
            ],
            'MCA': [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{outputs_path}/{type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage',
                    func=lambda x: (MCA_F(x)),
                    verbose=False
                )
            ]
        } if mix is True else None,
        "personalized": {
            'MlC': [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{outputs_path}/{type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/personalized/data_selection_storage',
                    func=lambda x: (MlC_F(x)),
                    verbose=False
                )
            ],
            'MCA': [
                load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                for file in get_filenames(
                    root_dir=f'{outputs_path}/{type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/personalized/data_selection_storage',
                    func=lambda x: (MCA_F(x)),
                    verbose=False
                )
            ]
        } if per is True else None
    }
    return files


def compare_MlC_MCA_for_GLO_PER_GAP(outputs_path=None, cwd=None, data_folder=None, classic_metrics=None,
                                        models_name=None, layers=[1], approach=None, alpha=0.85, type='qualitative'):
    """
    Analyzer Launcher
    Parameters
    ----------
    cols
    outputs_path
    cwd
    data_folder
    classic_metrics
    models_name
    layers
    approach
    alpha
    type

    Returns
    -------

    """
    print(f'{outputs_path}/{type}/model_storage')
    name = \
        [file for _, _, files in os.walk(
            f'{outputs_path}/{type}/model_storage')
         for file in files if
         '_best_features' in file][0]
    backup = read_model(
        f'{outputs_path}/{type}/model_storage/{name}')
    columns = backup["model"].keys()
    cols = backup['name']
    bestK = backup["bestK"]
    print(f"bestK: {bestK}, serialized objects: {backup}")
    day = time.strftime("%Y_%m_%d_%H")
    if cols is not None or classic_metrics is not None:  # check if cols and classics metrics are filled
        ## analyse of k layer

        # find out all best metric details
        """
        
                <td colspan='4'>Precision</td>
                <td colspan='4'>Recall</td>"""
        head_lambda = lambda \
                x: f"""
                <tr>
                <td colspan='2' rowspan='2'>{x}</td>
                <td colspan='4'>Accuracy</td>
                <td colspan='4'>F1-score</td>
                </tr>
                <tr>{('<td>Classic</td><td>GLO' + svg + '</td><td>PER' + svg + '</td><td>GAP' + svg + '</td>') * 2}</tr>
                """
        tab1_head = head_lambda(data_folder)
        tab1_body = ""
        metrics = {
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
            # 'precision': {
            #     'MlC': {
            #         'PER': [],
            #         'GAP': [],
            #         'GLO': []
            #     },
            #     'MCA': {
            #         'PER': [],
            #         'GAP': [],
            #         'GLO': []
            #     }
            # },
            # 'recall': {
            #     'MlC': {
            #         'PER': [],
            #         'GAP': [],
            #         'GLO': []
            #     },
            #     'MCA': {
            #         'PER': [],
            #         'GAP': [],
            #         'GLO': []
            #     }
            # },
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
            }
            # 'financial-cost': {
            #     'MlC': {
            #         'PER': [],
            #         'GAP': [],
            #         'GLO': []
            #     },
            #     'MCA': {
            #         'PER': [],
            #         'GAP': [],
            #         'GLO': []
            #     }
            # }
        }

        total_impact = {
            'accuracy': {
                'PER': [],
                'GAP': [],
                'GLO': []
            },
            # 'precision': {
            #     'PER': [],
            #     'GAP': [],
            #     'GLO': []
            # },
            # 'recall': {
            #     'PER': [],
            #     'GAP': [],
            #     'GLO': []
            # },
            'f1-score': {
                'PER': [],
                'GAP': [],
                'GLO': []
            }
            # 'financial-cost': {
            #     'PER': [],
            #     'GAP': [],
            #     'GLO': []
            # }
        }

        dictio = {
            'MlC': 'MlC',
            'MCA': 'MCA',
            'classic': 'Classic'
        }



        caption_content_lambda = lambda x: ''.join(
            [f'<span><strong>{key}</strong>: {value}</span><br>' for key, value in {
                **models_name,
                'GLO': f'Learning from classic dataset of {data_folder} where just Global MLN had been added',
                'PER': f'Learning from classic dataset of {data_folder} where just Personalised MLN had been added',
                'GAP': f'DLearning from classic dataset of {data_folder} where both Global and Personalised MLN had been added',
                'Classic': f'Learning from classic dataset',
                'MlC': f'Learning from classic dataset where MLN had been added',
                'MCA': f'Learning from classic dataset where MLN had been added and Att removed'
            }.items()])

        tab1_body_model = {key: deepcopy(metrics) for key in models_name.keys()}


        # fetch layers
        for k in layers:
            # fetch each combinantion of atributs in layers
            # if k == 1:
            #     config = get_combinations(range(len(cols)), k)
            # else:
            #     config = [columns[:k]]
            # for layer_config in config:  # create subsets of k index of OHE and fetch it
            #     col_targeted = [f'{cols[i]}' for i in layer_config]
            #     case_k = '±'.join(col_targeted) if len(layer_config) > 1 else col_targeted[0]
            #
            #     ### get files for distincts logic
            #     match = lambda x: (
            #         sum(
            #             [
            #                 re.sub(r'[^\w\s]', '', unidecode(partern)) in re.sub(r'[^\w\s]', '', unidecode(x))
            #                 for partern in case_k.split("±")
            #             ]
            #         ) == k if k > 1 else re.sub(r'[^\w\s]', '', unidecode(case_k)) in re.sub(r'[^\w\s]', '',
            #                                                                                  unidecode(x))
            #     )

            files = load_results(
                        outputs_path,
                        type,
                        k,
                        per=True,
                        glo=True,
                        mix=True
                )
            files['classic'] = classic_metrics
            # print(files)
            # outputs[logic] = files
            ### transform and normalize
            models_list = files['classic'].index.values.tolist()
            print(models_list)
            metrics = [
                "accuracy",
                # "precision",
                # "recall",
                "f1-score"
                # "financial-cost"
            ]

            # fetch models
            for p in range(len(files['global']["MCA"])):
                for model in models_list:
                    # fetch evaluation metric
                    for metric in metrics:
                        # fetch approach
                        for i, key in enumerate(approach):
                            # add metric in the vector
                            tab1_body_model[model][metric][key]['GLO'].append(round(((round(
                                files['global'][key][p].loc[model, metric], 4) - round(
                                files['classic'].loc[model, metric], 4)) / round(
                                files['classic'].loc[model, metric], 4)) * 100, 1)
                                #                                               if round(((round(
                                # files['global'][key][0].loc[model, metric], 4) - round(
                                # files['global']['classic'].loc[model, metric], 4)) / round(
                                # files['global']['classic'].loc[model, metric], 4)) * 100, 4) >= 0 else 0.0
                                                                              )

                            tab1_body_model[model][metric][key]['PER'].append(round(((round(
                                files['personalized'][key][p].loc[model, metric], 4) - round(
                                files['classic'].loc[model, metric], 4)) / round(
                                files['classic'].loc[model, metric], 4)) * 100, 1)
                                #                                               if round(((round(
                                # files['personalized'][key][0].loc[model, metric], 4) - round(
                                # files['personalized']['classic'].loc[model, metric], 4)) / round(
                                # files['personalized']['classic'].loc[model, metric], 4)) * 100, 4) >= 0 else 0.0
                                                                              )

                            tab1_body_model[model][metric][key]['GAP'].append(round(((round(
                                files['mixed'][key][p].loc[model, metric], 4) - round(
                                files['classic'].loc[model, metric], 4)) / round(
                                files['classic'].loc[model, metric], 4)) * 100, 1)
                                #                                               if round(((round(
                                # files['mixed'][key][0].loc[model, metric], 4) - round(
                                # files['mixed']['classic'].loc[model, metric], 4)) / round(
                                # files['mixed']['classic'].loc[model, metric], 4)) * 100, 4) >= 0 else 0.0
                                                                              )

                            # total_impact[metric]["PER"].append(tab1_body_model[model][metric][key]['P'] >= tab1_body_model[model][metric][key]['G'])
                            # total_impact[metric]["GLO"].append(tab1_body_model[model][metric][key]['P'] <= tab1_body_model[model][metric][key]['P'])

        # fetch each model
        for model in models_list:
            tab1_body += f'<tr> <td rowspan="{len(approach)}">{model}</td>'
            # fetch approach
            for i, key in enumerate(files['global'].keys() if approach == None else approach):
                # fetch evaluation metric
                tab1_body += f'<tr> <td>{dictio[key]}</td>' if i != 0 else f'<td>{dictio[key]}</td>'
                for y, metric in enumerate(metrics):

                    act = min if y == len(metrics) - 1 else max
                    if y != len(metrics) - 1:
                        # add metric in the vector
                        total_impact[metric]["PER"].append(
                            (act(tab1_body_model[model][metric][key]["PER"]) > 0.0)
                            and (act(tab1_body_model[model][metric][key]["GLO"]) <= act(
                            tab1_body_model[model][metric][key]["PER"]))
                            and (
                                act(tab1_body_model[model][metric][key]["GAP"]) <= act(
                            tab1_body_model[model][metric][key]["PER"]))
                        )

                        total_impact[metric]["GLO"].append(
                            (act(tab1_body_model[model][metric][key]["GLO"]) > 0.0)
                            and (act(tab1_body_model[model][metric][key]["GLO"]) >= act(
                            tab1_body_model[model][metric][key]["PER"]))
                            and (
                                act(tab1_body_model[model][metric][key]["GLO"]) >= act(
                            tab1_body_model[model][metric][key]["GAP"]))
                        )

                        total_impact[metric]["GAP"].append(
                            (act(tab1_body_model[model][metric][key]["GAP"]) > 0.0)
                            and (act(tab1_body_model[model][metric][key]["GAP"]) >= act(
                            tab1_body_model[model][metric][key]["PER"]))
                            and (
                                act(tab1_body_model[model][metric][key]["GAP"]) >= act(
                            tab1_body_model[model][metric][key]["GLO"]))
                        )
                    else:
                        # add metric in the vector
                        total_impact[metric]["PER"].append(
                            (act(tab1_body_model[model][metric][key]["PER"]) < 0.0)
                            and (act(tab1_body_model[model][metric][key]["GLO"]) >= act(
                                tab1_body_model[model][metric][key]["PER"]))
                            and (
                                    act(tab1_body_model[model][metric][key]["GAP"]) >= act(
                                tab1_body_model[model][metric][key]["PER"]))
                        )

                        total_impact[metric]["GLO"].append(
                            (act(tab1_body_model[model][metric][key]["GLO"]) < 0.0)
                            and (act(tab1_body_model[model][metric][key]["GLO"]) <= act(
                                tab1_body_model[model][metric][key]["PER"]))
                            and (
                                    act(tab1_body_model[model][metric][key]["GLO"]) <= act(
                                tab1_body_model[model][metric][key]["GAP"]))
                        )

                        total_impact[metric]["GAP"].append(
                            (act(tab1_body_model[model][metric][key]["GAP"]) < 0.0)
                            and (act(tab1_body_model[model][metric][key]["GAP"]) <= act(
                                tab1_body_model[model][metric][key]["PER"]))
                            and (
                                    act(tab1_body_model[model][metric][key]["GAP"]) <= act(
                                tab1_body_model[model][metric][key]["GLO"]))
                        )

                    tab1_body += (
                            f'<td>{round(files["classic"].loc[model, metric], 4)}</td><td>{"<strong>" * int((act(tab1_body_model[model][metric][key]["GLO"]) > 0) and (act(tab1_body_model[model][metric][key]["GLO"]) >= act(tab1_body_model[model][metric][key]["PER"])) and (act(tab1_body_model[model][metric][key]["GLO"]) >= act(tab1_body_model[model][metric][key]["GAP"])))}{act(tab1_body_model[model][metric][key]["GLO"])}{"</strong>" * int((act(tab1_body_model[model][metric][key]["GLO"]) > 0) and (act(tab1_body_model[model][metric][key]["GLO"]) >= act(tab1_body_model[model][metric][key]["PER"])) and (act(tab1_body_model[model][metric][key]["GLO"]) >= act(tab1_body_model[model][metric][key]["GAP"])))}</td>' +
                            f'<td> {"<strong>" * int((act(tab1_body_model[model][metric][key]["PER"]) > 0) and (act(tab1_body_model[model][metric][key]["GLO"]) <= act(tab1_body_model[model][metric][key]["PER"])) and (act(tab1_body_model[model][metric][key]["GAP"]) <= act(tab1_body_model[model][metric][key]["PER"])))}{act(tab1_body_model[model][metric][key]["PER"])}{"</strong>" * int((act(tab1_body_model[model][metric][key]["PER"]) > 0) and (act(tab1_body_model[model][metric][key]["GLO"]) <= act(tab1_body_model[model][metric][key]["PER"])) and (act(tab1_body_model[model][metric][key]["GAP"]) <= act(tab1_body_model[model][metric][key]["PER"])))}</td>' +
                            f'<td> {"<strong>" * int((act(tab1_body_model[model][metric][key]["GAP"]) > 0) and (act(tab1_body_model[model][metric][key]["GLO"]) <= act(tab1_body_model[model][metric][key]["GAP"])) and (act(tab1_body_model[model][metric][key]["PER"]) <= act(tab1_body_model[model][metric][key]["GAP"])))}{act(tab1_body_model[model][metric][key]["GAP"])}{"</strong>" * int((act(tab1_body_model[model][metric][key]["GAP"]) > 0) and (act(tab1_body_model[model][metric][key]["GLO"]) <= act(tab1_body_model[model][metric][key]["GAP"])) and (act(tab1_body_model[model][metric][key]["PER"]) <= act(tab1_body_model[model][metric][key]["GAP"])))}</td>'
                    ) if y != len(metrics) - 1 else (
                            f'<td>{round(files["classic"].loc[model, metric], 4)}</td><td>{"<strong>" * int((act(tab1_body_model[model][metric][key]["GLO"]) < 0) and (act(tab1_body_model[model][metric][key]["GLO"]) <= act(tab1_body_model[model][metric][key]["PER"])) and (act(tab1_body_model[model][metric][key]["GLO"]) <= act(tab1_body_model[model][metric][key]["GAP"])))}{act(tab1_body_model[model][metric][key]["GLO"])}{"</strong>" * int((act(tab1_body_model[model][metric][key]["GLO"]) < 0) and (act(tab1_body_model[model][metric][key]["GLO"]) <= act(tab1_body_model[model][metric][key]["PER"])) and (act(tab1_body_model[model][metric][key]["GLO"]) <= act(tab1_body_model[model][metric][key]["GAP"])))}</td>' +
                            f'<td> {"<strong>" * int((act(tab1_body_model[model][metric][key]["PER"]) < 0) and (act(tab1_body_model[model][metric][key]["GLO"]) >= act(tab1_body_model[model][metric][key]["PER"])) and (act(tab1_body_model[model][metric][key]["GAP"]) >= act(tab1_body_model[model][metric][key]["PER"])))}{act(tab1_body_model[model][metric][key]["PER"])}{"</strong>" * int((act(tab1_body_model[model][metric][key]["PER"]) < 0) and (act(tab1_body_model[model][metric][key]["GLO"]) >= act(tab1_body_model[model][metric][key]["PER"])) and (act(tab1_body_model[model][metric][key]["GAP"]) >= act(tab1_body_model[model][metric][key]["PER"])))}</td>' +
                            f'<td> {"<strong>" * int((act(tab1_body_model[model][metric][key]["GAP"]) < 0) and (act(tab1_body_model[model][metric][key]["GLO"]) >= act(tab1_body_model[model][metric][key]["GAP"])) and (act(tab1_body_model[model][metric][key]["PER"]) >= act(tab1_body_model[model][metric][key]["GAP"])))}{act(tab1_body_model[model][metric][key]["GAP"])}{"</strong>" * int((act(tab1_body_model[model][metric][key]["GAP"]) < 0) and (act(tab1_body_model[model][metric][key]["GLO"]) >= act(tab1_body_model[model][metric][key]["GAP"])) and (act(tab1_body_model[model][metric][key]["PER"]) >= act(tab1_body_model[model][metric][key]["GAP"])))}</td></tr>')
        tab1_body += f'<tr> <td colspan="2">Total</td>'
        for y, metric in enumerate(metrics):
            tab1_body += (
                    f'<td></td><td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["GAP"])))}{sum(total_impact[metric]["GLO"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["GAP"])))}</td>' +
                    f'<td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GAP"]) <= sum(total_impact[metric]["PER"])))}{sum(total_impact[metric]["PER"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GAP"]) <= sum(total_impact[metric]["PER"])))}</td>' +
                    f'<td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["GAP"])) and (sum(total_impact[metric]["GAP"]) >= sum(total_impact[metric]["PER"])))}{sum(total_impact[metric]["GAP"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["GAP"])) and (sum(total_impact[metric]["GAP"]) >= sum(total_impact[metric]["PER"])))}</td>'
            ) if y != len(metrics) - 1 else (
                    f'<td></td><td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["GAP"])))}{sum(total_impact[metric]["GLO"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["GAP"])))}</td>' +
                    f'<td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GAP"]) <= sum(total_impact[metric]["PER"])))}{sum(total_impact[metric]["PER"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GAP"]) <= sum(total_impact[metric]["PER"])))}</td>' +
                    f'<td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["GAP"])) and (sum(total_impact[metric]["GAP"]) >= sum(total_impact[metric]["PER"])))}{sum(total_impact[metric]["GAP"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["GAP"])) and (sum(total_impact[metric]["GAP"]) >= sum(total_impact[metric]["PER"])))}</td></tr>')

        caption = f'<caption><h2>Legend</h2>{caption_content_lambda(metric)}</caption>'
        table_html = f'<table style="border: 2px solid black; width: 100% !important; background-color: #FFFFFF; color:#000000;">{caption}{tab1_head}{tab1_body}</table>'
        htm = f'<html><head>{style}<title> GLO vs PER vs GAP </title></head><body style="background-color: white;">{table_html}</body></html>'

        create_domain(f'{cwd}/analyze_{outputs_path.split("/")[-2]}_made_on_{day}H/{data_folder}/compare_MlC_MCA_for_GLO_PER_GAP')
        timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
        filename1 = f'{cwd}/analyze_{outputs_path.split("/")[-2]}_made_on_{day}H/{data_folder}/compare_MlC_MCA_for_GLO_PER_GAP/Statistical comparaison of approachs in {data_folder} on global logic{alpha}.html'
        _file = open(filename1, "w")
        _file.write(htm)
        _file.close()


def compare_MlC_MCA_for_GLO_MX_PER_MX_PER_MY_GAP(
        outputs_path=None,
        cwd=None,
        data_folder=None,
        classic_metrics=None,
        models_name=None,
        layers=[1],
        approach=None,
        alpha=0.85,
        type='qualitative'
):
    """
    Analyzer Launcher
    Parameters
    ----------
    cols
    outputs_path
    cwd
    data_folder
    classic_metrics
    models_name
    layers
    approach
    alpha
    type

    Returns
    -------

    """
    print(f'{outputs_path}/{type}/model_storage')
    name = \
        [file for _, _, files in os.walk(
            f'{outputs_path}/{type}/model_storage')
         for file in files if
         '_best_features' in file][0]
    backup = read_model(
        f'{outputs_path}/{type}/model_storage/{name}')
    columns = backup["model"].keys()
    cols = backup['name']
    bestK = backup["bestK"]

    day = time.strftime("%Y_%m_%d_%H")
    if cols is not None or classic_metrics is not None:  # check if cols and classics metrics are filled
        ## analyse of k layer

        # find out all best metric details
        head_lambda = lambda \
                x: f"""
                <tr>
                <td colspan='2' rowspan='2'>{x}</td>
                <td colspan='4'>Accuracy</td>
                <td colspan='4'>Precision</td>
                <td colspan='4'>Recall</td>
                <td colspan='4'>F1-score</td>
                </tr>
                <tr>{('<td>Classic</td><td>GLO' + svg + '</td><td>PER' + svg + '</td><td>GAP' + svg + '</td>') * 4}</tr>
                """
        tab1_head = head_lambda(data_folder)
        tab1_body = ""
        metrics = {
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
            'precision': {
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
            'recall': {
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

        total_impact = {
            'accuracy': {
                'PER': [],
                'GAP': [],
                'GLO': []
            },
            'precision': {
                'PER': [],
                'GAP': [],
                'GLO': []
            },
            'recall': {
                'PER': [],
                'GAP': [],
                'GLO': []
            },
            'f1-score': {
                'PER': [],
                'GAP': [],
                'GLO': []
            },
            'financial-cost': {
                'PER': [],
                'GAP': [],
                'GLO': []
            }
        }

        dictio = {
            'MlC': 'MlC',
            'MCA': 'MCA',
            'classic': 'Classic'
        }

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

        caption_content_lambda = lambda x: ''.join(
            [f'<span><strong>{key}</strong>: {value}</span><br>' for key, value in {
                **models_name,
                'GLO': 'Learning from classic dataset of {data_folder} where just Global MLN had been added',
                'PER': 'Learning from classic dataset of {data_folder} where just Personalised MLN had been added',
                'GAP': 'DLearning from classic dataset of {data_folder} where both Global and Personalised MLN had been added',
                'Classic': f'Learning from classic dataset of {data_folder}',
                'MlC': f'Learning from classic dataset of {data_folder} where MLN had been added',
                'MCA': f'Learning from classic dataset of {data_folder} where MLN had been added and Att removed'
            }.items()])

        tab1_body_model = {key: deepcopy(metrics) for key in models_name.keys()}


        # fetch layers
        for k in layers:
            # fetch each combinantion of atributs in layers
            # if k == 1:
            #     config = get_combinations(range(len(cols)), k)
            # else:
            #     config = [columns[:k]]
            # for layer_config in config:  # create subsets of k index of OHE and fetch it
            #     col_targeted = [f'{cols[i]}' for i in layer_config]
            #     case_k = '±'.join(col_targeted) if len(layer_config) > 1 else col_targeted[0]
            #
            #     ### get files for distincts logic
            #     match = lambda x: (
            #         sum(
            #             [
            #                 re.sub(r'[^\w\s]', '', unidecode(partern)) in re.sub(r'[^\w\s]', '', unidecode(x))
            #                 for partern in case_k.split("±")
            #             ]
            #         ) == k if k > 1 else re.sub(r'[^\w\s]', '', unidecode(case_k)) in re.sub(r'[^\w\s]', '',
            #                                                                                  unidecode(x))
            #     )

            files = {
                'global': {
                    'classic': classic_metrics,
                    'MlC': [
                        load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                        for file in get_filenames(
                            root_dir=f'{outputs_path}/{type}/{"mlna_"+str(k) if k == 1 else "mlna_"+str(k)+"_b"}/global/data_selection_storage',
                            func=lambda x: (MlC_F(x)),
                            verbose=False
                        )
                    ],
                    'MCA': [
                        load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                        for file in get_filenames(
                            root_dir=f'{outputs_path}/{type}/{"mlna_"+str(k) if k == 1 else "mlna_"+str(k)+"_b"}/global/data_selection_storage',
                            func=lambda x: (MCA_F(x)),
                            verbose=False
                        )
                    ]
                },
                'mixed': {
                    'classic': classic_metrics,
                    'MlC': [
                        load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                        for file in get_filenames(
                            root_dir=f'{outputs_path}/{type}/{"mlna_"+str(k) if k == 1 else "mlna_"+str(k)+"_b"}/mixed/data_selection_storage',
                            func=lambda x: (MlC_F(x)),
                            verbose=False
                        )
                    ],
                    'MCA': [
                        load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                        for file in get_filenames(
                            root_dir=f'{outputs_path}/{type}/{"mlna_"+str(k) if k == 1 else "mlna_"+str(k)+"_b"}/mixed/data_selection_storage',
                            func=lambda x: (MCA_F(x)),
                            verbose=False
                        )
                    ]
                },
                "personalized": {
                    'classic': classic_metrics,
                    'MlC': [
                        load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                        for file in get_filenames(
                            root_dir=f'{outputs_path}/{type}/{"mlna_"+str(k) if k == 1 else "mlna_"+str(k)+"_b"}/personalized/data_selection_storage',
                            func=lambda x: (MlC_F(x)),
                            verbose=False
                        )
                    ],
                    'MCA': [
                        load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                        for file in get_filenames(
                            root_dir=f'{outputs_path}/{type}/{"mlna_"+str(k) if k == 1 else "mlna_"+str(k)+"_b"}/personalized/data_selection_storage',
                            func=lambda x: (MCA_F(x)),
                            verbose=False
                        )
                    ]
                }
            }
            # print(files)
            # outputs[logic] = files
            ### transform and normalize
            models_list = files['personalized']['classic'].index.values.tolist()
            print(models_list)
            metrics = ["accuracy", "precision", "recall", "f1-score"]

            # fetch models
            for i in range(len(files['global']["MCA"])):
                for model in models_list:
                    # fetch evaluation metric
                    for metric in metrics:
                        # fetch approach
                        for i, key in enumerate(approach):
                            # add metric in the vector
                            tab1_body_model[model][metric][key]['GLO'].append(round(((round(
                                files['global'][key][0].loc[model, metric], 4) - round(
                                files['global']['classic'].loc[model, metric], 4)) / round(
                                files['global']['classic'].loc[model, metric], 4)) * 100, 4) if round(((round(
                                files['global'][key][0].loc[model, metric], 4) - round(
                                files['global']['classic'].loc[model, metric], 4)) / round(
                                files['global']['classic'].loc[model, metric], 4)) * 100, 4) >= 0 else 0.0)

                            tab1_body_model[model][metric][key]['PER'].append(round(((round(
                                files['personalized'][key][0].loc[model, metric], 4) - round(
                                files['personalized']['classic'].loc[model, metric], 4)) / round(
                                files['personalized']['classic'].loc[model, metric], 4)) * 100, 4) if round(((round(
                                files['personalized'][key][0].loc[model, metric], 4) - round(
                                files['personalized']['classic'].loc[model, metric], 4)) / round(
                                files['personalized']['classic'].loc[model, metric], 4)) * 100, 4) >= 0 else 0.0)

                            tab1_body_model[model][metric][key]['GAP'].append(round(((round(
                                files['mixed'][key][0].loc[model, metric], 4) - round(
                                files['mixed']['classic'].loc[model, metric], 4)) / round(
                                files['mixed']['classic'].loc[model, metric], 4)) * 100, 4) if round(((round(
                                files['mixed'][key][0].loc[model, metric], 4) - round(
                                files['mixed']['classic'].loc[model, metric], 4)) / round(
                                files['mixed']['classic'].loc[model, metric], 4)) * 100, 4) >= 0 else 0.0)

                            # total_impact[metric]["PER"].append(tab1_body_model[model][metric][key]['P'] >= tab1_body_model[model][metric][key]['G'])
                            # total_impact[metric]["GLO"].append(tab1_body_model[model][metric][key]['P'] <= tab1_body_model[model][metric][key]['P'])

        # fetch each model
        for model in models_list:
            tab1_body += f'<tr> <td rowspan="{len(approach)}">{model}</td>'
            # fetch approach
            for i, key in enumerate(files['global'].keys() if approach == None else approach):
                # fetch evaluation metric
                tab1_body += f'<tr> <td>{dictio[key]}</td>' if i != 0 else f'<td>{dictio[key]}</td>'
                for y, metric in enumerate(metrics):
                    # add metric in the vector
                    total_impact[metric]["PER"].append((max(tab1_body_model[model][metric][key]["GLO"]) <= max(
                        tab1_body_model[model][metric][key]["PER"])) and (
                                                            max(tab1_body_model[model][metric][key]["GAP"]) <= max(
                                                        tab1_body_model[model][metric][key]["PER"])))

                    total_impact[metric]["GLO"].append((max(tab1_body_model[model][metric][key]["GLO"]) >= max(
                        tab1_body_model[model][metric][key]["PER"])) and (
                                                            max(tab1_body_model[model][metric][key]["GLO"]) >= max(
                                                        tab1_body_model[model][metric][key]["GAP"])))

                    total_impact[metric]["GAP"].append((max(tab1_body_model[model][metric][key]["GAP"]) >= max(
                        tab1_body_model[model][metric][key]["PER"])) and (
                                                            max(tab1_body_model[model][metric][key]["GAP"]) >= max(
                                                        tab1_body_model[model][metric][key]["PER"])))

                    tab1_body += (
                            f'<td>{round(files["global"]["classic"].loc[model, metric], 4)}</td><td>{"<strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) >= max(tab1_body_model[model][metric][key]["PER"])) and (max(tab1_body_model[model][metric][key]["GLO"]) >= max(tab1_body_model[model][metric][key]["GAP"])))}{max(tab1_body_model[model][metric][key]["GLO"])}{"</strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) >= max(tab1_body_model[model][metric][key]["PER"])) and (max(tab1_body_model[model][metric][key]["GLO"]) >= max(tab1_body_model[model][metric][key]["GAP"])))}</td>' +
                            f'<td> {"<strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) <= max(tab1_body_model[model][metric][key]["PER"])) and (max(tab1_body_model[model][metric][key]["GAP"]) <= max(tab1_body_model[model][metric][key]["PER"])))}{max(tab1_body_model[model][metric][key]["PER"])}{"</strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) <= max(tab1_body_model[model][metric][key]["PER"])) and (max(tab1_body_model[model][metric][key]["GAP"]) <= max(tab1_body_model[model][metric][key]["PER"])))}</td>' +
                            f'<td> {"<strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) <= max(tab1_body_model[model][metric][key]["GAP"])) and (max(tab1_body_model[model][metric][key]["PER"]) <= max(tab1_body_model[model][metric][key]["GAP"])))}{max(tab1_body_model[model][metric][key]["GAP"])}{"</strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) <= max(tab1_body_model[model][metric][key]["GAP"])) and (max(tab1_body_model[model][metric][key]["PER"]) <= max(tab1_body_model[model][metric][key]["GAP"])))}</td>'
                    ) if y != len(metrics) - 1 else (
                            f'<td>{round(files["global"]["classic"].loc[model, metric], 4)}</td><td>{"<strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) >= max(tab1_body_model[model][metric][key]["PER"])) and (max(tab1_body_model[model][metric][key]["GLO"]) >= max(tab1_body_model[model][metric][key]["GAP"])))}{max(tab1_body_model[model][metric][key]["GLO"])}{"</strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) >= max(tab1_body_model[model][metric][key]["PER"])) and (max(tab1_body_model[model][metric][key]["GLO"]) >= max(tab1_body_model[model][metric][key]["GAP"])))}</td>' +
                            f'<td> {"<strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) <= max(tab1_body_model[model][metric][key]["PER"])) and (max(tab1_body_model[model][metric][key]["GAP"]) <= max(tab1_body_model[model][metric][key]["PER"])))}{max(tab1_body_model[model][metric][key]["PER"])}{"</strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) <= max(tab1_body_model[model][metric][key]["PER"])) and (max(tab1_body_model[model][metric][key]["GAP"]) <= max(tab1_body_model[model][metric][key]["PER"])))}</td>' +
                            f'<td> {"<strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) <= max(tab1_body_model[model][metric][key]["GAP"])) and (max(tab1_body_model[model][metric][key]["PER"]) <= max(tab1_body_model[model][metric][key]["GAP"])))}{max(tab1_body_model[model][metric][key]["GAP"])}{"</strong>" * int((max(tab1_body_model[model][metric][key]["GLO"]) <= max(tab1_body_model[model][metric][key]["GAP"])) and (max(tab1_body_model[model][metric][key]["PER"]) <= max(tab1_body_model[model][metric][key]["GAP"])))}</td></tr>')
        tab1_body += f'<tr> <td colspan="2">Total</td>'
        for y, metric in enumerate(metrics):
            tab1_body += (
                    f'<td></td><td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["GAP"])))}{sum(total_impact[metric]["GLO"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["GAP"])))}</td>' +
                    f'<td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GAP"]) <= sum(total_impact[metric]["PER"])))}{sum(total_impact[metric]["PER"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GAP"]) <= sum(total_impact[metric]["PER"])))}</td>' +
                    f'<td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["GAP"])) and (sum(total_impact[metric]["GAP"]) >= sum(total_impact[metric]["PER"])))}{sum(total_impact[metric]["GAP"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["GAP"])) and (sum(total_impact[metric]["GAP"]) >= sum(total_impact[metric]["PER"])))}</td>'
            ) if y != len(metrics) - 1 else (
                    f'<td></td><td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["GAP"])))}{sum(total_impact[metric]["GLO"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GLO"]) >= sum(total_impact[metric]["GAP"])))}</td>' +
                    f'<td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GAP"]) <= sum(total_impact[metric]["PER"])))}{sum(total_impact[metric]["PER"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["PER"])) and (sum(total_impact[metric]["GAP"]) <= sum(total_impact[metric]["PER"])))}</td>' +
                    f'<td>{"<strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["GAP"])) and (sum(total_impact[metric]["GAP"]) >= sum(total_impact[metric]["PER"])))}{sum(total_impact[metric]["GAP"])}{"</strong>" * int((sum(total_impact[metric]["GLO"]) <= sum(total_impact[metric]["GAP"])) and (sum(total_impact[metric]["GAP"]) >= sum(total_impact[metric]["PER"])))}</td></tr>')

        caption = f'<caption><h2>Legend</h2>{caption_content_lambda(metric)}</caption>'
        table_html = f'<table style="border: 2px solid black; width: 100% !important; background-color: #FFFFFF; color:#000000;">{caption}{tab1_head}{tab1_body}</table>'
        htm = f'<html><head>{style}<title> GLO vs PER vs GAP </title></head><body style="background-color: white;">{table_html}</body></html>'

        create_domain(f'{cwd}/analyze_{outputs_path.split("/")[-2]}_made_on_{day}H/{data_folder}/plots/fnTab/tab3')
        timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
        filename1 = f'{cwd}/analyze_{outputs_path.split("/")[-2]}_made_on_{day}H/{data_folder}/plots/fnTab/tab3/Statistical comparaison of approachs in {data_folder} on global logic{alpha}.html'
        _file = open(filename1, "w")
        _file.write(htm)
        _file.close()


def Descriptors_comparison(
        outputs_path=None,
        models_name=None,
        layers=[1],
        approach=None,
        aggregation_f=None,
        alphas=[0.85],
        type='qualitative'
):
    """

    Parameters
    ----------
    outputs_path
    models_name
    layers
    approach
    aggregation_f

    Returns
    -------
    tab1_body_model_f: dictionary of descriptors arrays values
    """



    # find out all best metric details
    descripteurs = {
        'GLO_INTER': [],
        'GLO_INTRA': [],
        'GLO_COMBINE': [],
        'PER_INTER': [],
        'PER_INTRA': [],
        'PER_COMBINE': [],
        'GLO_INTER_M': [],
        'GLO_INTRA_M': [],
        'GLO_COMBINE_M': [],
        'PER_INTER_M': [],
        'PER_INTRA_M': [],
        'PER_COMBINE_M': [],
        'DEGREE': []

    }
    mlnL = {f'MLN {key}': descripteurs for i, key in enumerate(layers)}

    tab1_body_model_f = {key: deepcopy(descripteurs) for key in models_name.keys()}
    tab1_body_model = {key: deepcopy(mlnL) for key in models_name.keys()}

    for alp in alphas:
        print(f'{outputs_path}/{alp}/{type}/model_storage')
        name = \
            [file for _, _, files in os.walk(
                f'{outputs_path}/{alp}/{type}/model_storage')
             for file in files if
             '_best_features' in file][0]
        backup = read_model(
            f'{outputs_path}/{alp}/{type}/model_storage/{name}')
        cols = backup['name']
        if cols is not None:  # check if cols and classics metrics are filled
            ## analyse of k layer
            # fetch layers
            for d, k in enumerate(layers):
                ### get files for distincts logic
                # print(f'{outputs_path}/{type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage')
                # files = {
                #     'mixed':  [
                #         load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
                #         for file in get_filenames(
                #             root_dir=f'{outputs_path}/{type}/{"mlna_" + str(k) if k == 1 else "mlna_" + str(k) + "_b"}/mixed/data_selection_storage',
                #             func=lambda x: (MCA_F(x) or MlC_F(x)),
                #             verbose=False
                #         )
                #     ]
                # }
                files = load_results(
                        f'{outputs_path}/{alp}',
                        type,
                        k,
                        per=False,
                        glo=False,
                        mix=True,
                    )
                # print(files)
                # outputs[logic] = files
                ### transform and normalize
                models_list = files['mixed']["MlC"][0].index.values.tolist()
                # print(models_list)
                logics = ["MlC","MCA"]

                # fetch models
                for model in models_list:
                    # fetch evaluation metric
                    for i, key in enumerate(files['mixed'].keys() if approach is None else approach):
                        # fetch on column or attributs
                        for logic in logics:
                            for p in range(len(files[key][logic])):
                                colo = files[key][logic][p].columns
                                for att in colo:
                                    if not (att in ["accuracy", "precision", "recall", "f1-score","financial-cost"]):
                                        if GLO_INTER_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'GLO_INTER'].append(files[key][logic][p].loc[model, att])
                                        elif GLO_INTRA_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'GLO_INTRA'].append(files[key][logic][p].loc[model, att])
                                        elif GLO_COMBINE_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'GLO_COMBINE'].append(files[key][logic][p].loc[model, att])
                                        elif PER_INTER_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'PER_INTER'].append(files[key][logic][p].loc[model, att])
                                        elif PER_INTRA_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'PER_INTRA'].append(files[key][logic][p].loc[model, att])
                                        elif PER_COMBINE_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'PER_COMBINE'].append(files[key][logic][p].loc[model, att])
                                        elif GLO_INTER_M_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'GLO_INTER_M'].append(files[key][logic][p].loc[model, att])
                                        elif GLO_INTRA_M_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'GLO_INTRA_M'].append(files[key][logic][p].loc[model, att])
                                        elif GLO_COMBINE_M_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'GLO_COMBINE_M'].append(files[key][logic][p].loc[model, att])
                                        elif PER_INTER_M_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'PER_INTER_M'].append(files[key][logic][p].loc[model, att])
                                        elif PER_INTRA_M_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'PER_INTRA_M'].append(files[key][logic][p].loc[model, att])
                                        elif PER_COMBINE_M_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'PER_COMBINE_M'].append(files[key][logic][p].loc[model, att])
                                        elif DEGREE_F(att):
                                            tab1_body_model[model][f'MLN {k}'][
                                                'DEGREE'].append(files[key][logic][p].loc[model, att])
                                        else:
                                            if not (att in tab1_body_model[model][
                                                f'MLN {k}'].keys()):
                                                tab1_body_model[model][f'MLN {k}'][
                                                    att] = []
                                                tab1_body_model_f[model][att] = []
                                                descripteurs[att] = []
                                                # print(tab1_body_model[model][f'MLN {k}'])
                                            tab1_body_model[model][f'MLN {k}'][
                                                att].append(files[key][logic][p].loc[model, att])

        # fetch each model
    for model in models_list:
        # fetch layers
        for z, layer in enumerate(mlnL.keys()):
            # fetch attributs
            for att in tab1_body_model[model][layer].keys():
                # print(att)
                # print(tab1_body_model[model][layer][att])
                tab1_body_model_f[model][att] = [*tab1_body_model_f[model][att],
                                                 *tab1_body_model[model][layer][att]]
                descripteurs[att] = [*descripteurs[att], *tab1_body_model[model][layer][att]]
        for att in tab1_body_model_f[model].keys():
            tab1_body_model_f[model][att] = aggregation_f(tab1_body_model_f[model][att])
    # data = dict(sorted(tab1_body_model_f[model].items(), key=lambda x: abs(x[1]), reverse=False)[-25:])
    return tab1_body_model_f, tab1_body_model


def analyzer_launcher_for_descriptor_rank(
        outputs_name=None,
        analytical_func=None,
        approach=None,
        layers=None,
        aggregation_f=None,
        top=None,
        alphaConfig=None,
        with_class=False
):
    result_folders = [dirnames for _, dirnames, _ in os.walk(f'{os.getcwd()}/{outputs_name}')][0]
    # for i in alphaConfig:
    head_lambda = None
    day = time.strftime("%Y_%m_%d_%H")
    model_lines = {dataset: None for dataset in result_folders}

    result_content = {
        "nb": "<tr>",
        "pos": "<tr>"
    }
    style = '<style> table, th, td {border: 1px solid black;border-collapse: collapse;} .wrap-text {word-wrap: break-word;} .wrap-text {overflow-wrap: break-word;} .limited-column {width: 100px;} .dashed-border {border: 1px dashed black;}.dotted-border {border: 1px dotted black;} td {text-align: center;} .data {text-align: left;} caption {margin:0; margin-bottom: 2px; text-align: start; border: 1px dashed black;} caption > h2 {text-align: center;}</style>'

    for dataset_name in result_folders:
        print(dataset_name)
        classic_f = [
            load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
            for file in get_filenames(
                root_dir=f'{os.getcwd()}/{outputs_name}/{dataset_name}/data_selection_storage',
                func=C_F,
                verbose=False
            )
        ][-1]
        # quali_col = get_qualitative_from_cols(classic_f.columns.to_list())
        models_list = classic_f.index.values.tolist()
        models = model_desc()
        models_name = {key: models[key] for key in models.keys() if key in models_list}


        if model_lines[dataset_name] is None:
            model_lines[dataset_name] = {
                model: {
                'GLO': {
                    'nb': {},
                    'pos': {}
                },
                'PER': {
                    'nb': {},
                    'pos': {}
                },
                'var': 0
            } for model in models_name}
        if head_lambda == None:
            head_lambda = lambda \
                x: f'<tr><td colspan="2">{x}</td>{"".join(["<td>" + modeli + "</td>" for modeli in models_name.keys()])}</tr>'

        rank,_ = analytical_func(
            outputs_path=f'{os.getcwd()}/{outputs_name}/{dataset_name}{"/withClass"*int(with_class)}',
            models_name=models_name,
            approach=approach,
            aggregation_f=aggregation_f,
            layers=layers,
            alphas=alphaConfig,
            type='qualitative',
        #
        #     cols=quali_col,
        #     cwd=os.getcwd(),
        #     data_folder=dataset_name,
        #
        # outputs_path = None, models_name = None, layers = [1], approach = None,
        # aggregation_f = None, alpha = 0.85, type = 'qualitative'
        )
        # fetch result on dataset base on model
        for model in models_name.keys():
            # compute the number time that MLN attribut appear in the top list
            data = dict(sorted(rank[model].items(), key=lambda x: abs(x[1]), reverse=False)[-top:])
            data1 = dict(sorted(rank[model].items(), key=lambda x: abs(x[1]), reverse=False))
            nb = sum([('GLO' in key) for key in data.keys()])
            vec_id = [i for i, key in enumerate(data1.keys()) if ('GLO' in key)]
            nb1 = sum([('PER' in key) for key in data.keys()])
            vec_id1 = [i for i, key in enumerate(data1.keys()) if ('PER' in key)]
            # print(f"{dataset_name} {model} {data}")
            pos = (len(data1.keys()) - vec_id[len(vec_id) - 1]) if len(vec_id) != 0 else 'NA'
            pos1 = (len(data1.keys()) - vec_id[len(vec_id1) - 1]) if len(vec_id1) != 0 else 'NA'
            model_lines[dataset_name][model]['GLO']["nb"] = f"<td>{nb}</td>"
            model_lines[dataset_name][model]['GLO']["pos"] = f"<td>{pos}</td>"
            model_lines[dataset_name][model]['PER']["nb"] = f"<td>{nb1}</td>"
            model_lines[dataset_name][model]['PER']["pos"] = f"<td>{pos1}</td>"
            model_lines[dataset_name][model]["var"] = len(data1.keys())
    # print(model_lines)
    for dataset_name in result_folders:
        for p in result_content.keys():
            result_content[
                p] += f'<td class="data" rowspan="2">{dataset_name} ({model_lines[dataset_name][model]["var"] - 14} + 14)</td>'
            for j,logic in enumerate(['GLO', 'PER']):
                result_content[p] += f'{"" if j == 0 else "<tr>"}<td>{logic}</td>'
                for _,model in enumerate(models_name.keys()):
                    result_content[p] += (f'{model_lines[dataset_name][model][logic][p]}')
                result_content[p] += "</tr>"

    # for dataset_name in result_folders:
    #     for p in result_content.keys():
    #         for e in result_content[p].keys():
    #             result_content[
    #                 p][e] += f'<td class="data">{dataset_name} ({model_lines[dataset_name][model]["var"] - 9} + 9)</td>'
    #             for i, model in enumerate(models_name.keys()):
    #                 result_content[p][e] += (f'{model_lines[dataset_name][model][p][e]}')
    #             result_content[p][e] += "</tr>"
    for p in result_content.keys():
        tab3_head_g = head_lambda(f'TOP {top} {p.upper()}')
        # comparaison
        tabs = f'<table style="border: 2px solid black; width: 100% !important; background-color: #FFFFFF; color:#000000;">{tab3_head_g}{result_content[p]}</table>'

        # caption = f'<caption><h2>Legend</h2>{caption_content_lambda(metric)}</caption>'
        # table_html = f'<table style="border: 2px solid black; width: 100% !important; background-color: #FFFFFF; color:#000000;">{tab3_head_g[metric]}{tab3_body_g1[metric]}</table>'
        htm = f'<html><head>{style}<title> Top {top} {p.upper()} MLN attributs ranking </title></head><body style="background-color: white;">{tabs}</body></html>'

        create_domain(f'{os.getcwd()}/analyze_{outputs_name}_made_on_{day}H/descriptor/{p}/{("-".join([str(el) for el in alphaConfig])).replace(".","_")}')
        timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
        filename1 = f'{os.getcwd()}/analyze_{outputs_name}_made_on_{day}H/descriptor/{p}/{("-".join([str(el) for el in alphaConfig])).replace(".","_")}/Top {top} {p.upper()} MLN attributs ranking.html'
        _file = open(filename1, "w")
        _file.write(htm)
        _file.close()


def analyzer_launcher_for_descriptor_rank_plot(
        outputs_name=None,
        analytical_func=None,
        approach=None,
        layers=None,
        aggregation_f=None,
        top=None,
        alphaConfig=None,
        with_class=False
):
    result_folders = [dirnames for _, dirnames, _ in os.walk(f'{os.getcwd()}/{outputs_name}')][0]
    day = time.strftime("%Y_%m_%d_%H")
    for dataset_name in result_folders:
        print(dataset_name)
        classic_f = [
            load_data_set_from_url(path=file, sep='\t', encoding='utf-8', index_col=0, na_values=None)
            for file in get_filenames(
                root_dir=f'{os.getcwd()}/{outputs_name}/{dataset_name}/data_selection_storage',
                func=C_F,
                verbose=False
            )
        ][-1]
        # quali_col = get_qualitative_from_cols(classic_f.columns.to_list())
        models_list = classic_f.index.values.tolist()
        models = model_desc()
        models_name = {key: models[key] for key in models.keys() if key in models_list}

        rank,_ = analytical_func(
            outputs_path=f'{os.getcwd()}/{outputs_name}/{dataset_name}{"/withClass"*int(with_class)}',
            models_name=models_name,
            approach=approach,
            aggregation_f=aggregation_f,
            layers=layers,
            alphas=alphaConfig,
            type='qualitative',
        #
        #     cols=quali_col,
        #     cwd=os.getcwd(),
        #     data_folder=dataset_name,
        #
        # outputs_path = None, models_name = None, layers = [1], approach = None,
        # aggregation_f = None, alpha = 0.85, type = 'qualitative'
        )
        # fetch result on dataset base on model
        for model in models_name.keys():
            create_domain(f'{os.getcwd()}/analyze_{outputs_name}_made_on_{day}H/{"withClass/"*int(with_class)}descriptor/shap/{top}')
            # timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
            # filename1 = f'{cwd}/analyze_{outputs_path.split("/")[-2]}_made_on_{day}H/{data_folder}/shap/{"_".join(approach)}_{data_folder}_{model}_shapley'
            filename1 = f'{os.getcwd()}/analyze_{outputs_name}_made_on_{day}H/{"withClass/"*int(with_class)}descriptor/shap/{top}/{dataset_name}_{model}_{("-".join([str(el) for el in alphaConfig])).replace(".","_")}_shapley.png'

            # pl.show()
            # pl.savefig(filename1+".png", dpi=300)

            # fig setup
            width = 15
            height = int(len(np.unique(rank[model].keys())) * 1.5)
            # Set a larger figure size
            pl.figure(figsize=(width, height))
            data = dict(sorted(rank[model].items(), key=lambda x: abs(x[1]), reverse=False)[-top:])
            df = pd.DataFrame({'features': data.keys(), 'importances': data.values()})
            # print(custom_color(df.features, [])[0],df.features)
            bars = df.plot.barh(x='features', y='importances', color=custom_color(df.features, [])[0])

            pl.title(f"{model}")
            # Custom colors for the legend
            mln_color = 'green'
            classic_color = 'dodgerblue'

            # Add custom colors to the legend
            legend_elements = [
                Patch(facecolor=mln_color, edgecolor='black', label='MLN'),
                Patch(facecolor=classic_color, edgecolor='black', label='Classic')
            ]

            # Reduce the font size of x-axis label
            pl.xlabel('Importances', fontsize=8)

            # Reduce the font size of y-axis label
            pl.ylabel('Features', fontsize=8)

            # Reduce the font size of tick labels on x-axis
            pl.xticks(fontsize=6)

            # Reduce the font size of tick labels on y-axis
            pl.yticks(fontsize=5)

            # Reduce the font size of the plot title
            pl.title(f'{models_name[model]} - {dataset_name}', fontsize=10)

            # Reduce the font size of the legend
            pl.legend(fontsize=8)
            # Add values on top of the bars
            for i, bar in enumerate(bars.containers[0]):
                width = bar.get_width()
                pl.annotate(f'{width:.2f}', xy=(width, bar.get_y() + bar.get_height() / 2), xytext=(3, 0),
                            textcoords='offset points', ha='left', va='center', fontsize=8)
            pl.legend(handles=legend_elements, facecolor='white', framealpha=1, bbox_to_anchor=(1, 1), loc='upper left',
                      title='Legend Title')

            # pl.axvline(x=0, color=".5")
            # pl.subplots_adjust(left=0.3)
            # pl.tight_layout()

            # Mise en gras des xticks
            font = FontProperties(weight='bold')
            pl.xticks(fontproperties=font, fontsize=8)
            pl.yticks(fontproperties=font, fontsize=6)

            # Suppression de la légende
            pl.legend().remove()
            pl.savefig(filename1, dpi=300)  # .png,.pdf will also support here
            pl.close()  # close the plot windows
