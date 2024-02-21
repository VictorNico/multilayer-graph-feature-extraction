    
"""
  Author: VICTOR DJIEMBOU
  addedAt: 15/11/2023
  changes:
    - 15/11/2023:
      - add pipeline methods
"""
#################################################
##          Libraries importation
#################################################

###### Begin

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#import missingno as msno
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
#import shap
import sys
import numpy as np
import time
import os
import joblib
import networkx as nx
from IPython.core.display import HTML
import imgkit
from modules.file import create_domain
import pandas as pd
# from memory_profiler import profile
# from markdown2pdf import convert
# styling
# %matplotlib inline
sns.set_style('darkgrid')
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.facecolor'] = '#00000000'
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.facecolor'] = '#00000000'

###### End


#################################################
##          Methods definition
#################################################

#Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell
# @profile
def multi_table(table_list):
        return HTML('<table><tr style="background-color:#2020d1; color: #FFFFFF;">' +  ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +'</tr></table>')

# @profile
def plot_graph():
        # show
        colors = nx.get_edge_attributes(CRP_G_1,'color').values()
        colorsN = nx.get_node_attributes(CRP_G_1,'color').values()
        nx.draw(
            CRP_G_1
            ,edge_color=colors
            ,node_color=colorsN
            #,with_labels=True
        )
        timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
        filename1 = './plots/graph_mln_1_person_home_ownership_loan_intent'+'_'+timestr+'.png'
        plt.savefig(filename1,dpi=700) #.png,.pdf will also support here
        plt.show()

# @profile
def custom_color(dataframe, graph_a=[]):
    cols= dataframe.columns.tolist()
    colors= []
    for col in cols:
        if col in graph_a:
            colors.append('yellow')
        elif 'MLN_' in col:
            colors.append('green')
        # elif 'STAT_' in col:
        #     colors.append('blue')
        else:
            colors.append('dodgerblue')
    return [colors, cols]

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def get_color():
    return color

def model_desc():
    modelD = {
    #'sv' :'SVM',
    'xgb':'XGBOOST',
    'dtc':'DECISION TREE',
    'lrc':'LOGISTIC REGRESSION',
    'rfc':'RANDOM FOREST',
    #'knn':knc
    }
    return modelD

# @profile
def plot_features_importance_as_barh(data, getColor, modelDictName,plotTitle, cwd, graph_a=[], save=True, prefix=None):
    for index in data.index.values.tolist():
        ok = data.drop([
                'f1-score',
                'recall',
                'precision',
                'accuracy'
                ], axis=1
            )
        
        ok = ok.sort_values(
                by = index,
                axis = 1, 
                ascending = True
            )
        # fig setup
        width = 10
        height = int(len(np.unique(ok.columns.tolist()))/4)
        # plt.figure(figsize=(width,height))
        
        ok = ok.loc[[index],:]
        
        df = pd.DataFrame({'Attr':ok.columns.tolist(),'Val':ok.values[0]})
        df.plot.barh(x='Attr', y='Val', figsize=(width, height), color=getColor(ok, graph_a)[0])
        
        plt.title(f"{plotTitle}")
        plt.axvline(x=0, color=".5")
        
        label = f"""
        {modelDictName[index]}
        ACCURACY: {data.loc[index,'accuracy']}, PRECISION: {data.loc[index,'precision']},
        RECALL: {data.loc[index,'recall']}, F1-SCORE: {data.loc[index,'f1-score']}
        """
        
        plt.xlabel(label)
        plt.subplots_adjust(left=0.3)
        
        if save == True:
            create_domain(cwd+'/plots/')

            timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
            filename1 = cwd+'/plots'+'/'+('_'.join(plotTitle.split(' ')))+'_'+index+'_'+timestr+'.png'
            # Adjust the layout to cover all content
            plt.tight_layout()

            plt.savefig(filename1,dpi=700) #.png,.pdf will also support here
            plt.close() # close the plot windows
            
        #plt.show()

# @profile
def print_summary(table_list, modelDict):
    baseline = table_list[0][1].index.values.tolist()
    head = '<tr><td></td><td></td><td>Accuracy</td>'+'<td>Precision</td><td>Recall</td><td>F1-score</td></tr>'
    body = ''
    for model in baseline:
        modelLines = f'<tr><td rowspan="{len(table_list)}" >{modelDict[model]}</td>'
        for i, (step,data) in enumerate(table_list):
            modelLines = (
                modelLines + 
                f'<td>{step}</td><td style={"color:blue; font-size: 40px; font-weight: bold;" * int(sum([ round(data.loc[model,"accuracy"],4) == max(round(dat.loc[model,"accuracy"],4),round(data.loc[model,"accuracy"],4)) for step, dat in table_list])/len(table_list))} >{round(data.loc[model,"accuracy"],4)}</td>'+
                f'<td style={"color:blue; font-size: 40px; font-weight: bold;" * int(sum([ round(data.loc[model,"precision"],4) == max(round(dat.loc[model,"precision"],4),round(data.loc[model,"precision"],4)) for step, dat in table_list])/len(table_list))}>{round(data.loc[model,"precision"],4)}</td>'+
                f'<td style={"color:blue; font-size: 40px; font-weight: bold;" * int(sum([ round(data.loc[model,"recall"],4) == max(round(dat.loc[model,"recall"],4),round(data.loc[model,"recall"],4)) for step, dat in table_list])/len(table_list))}>{round(data.loc[model,"recall"],4)}</td>'+
                f'<td style={"color:blue; font-size: 40px; font-weight: bold;" * int(sum([ round(data.loc[model,"f1-score"],4) == max(round(dat.loc[model,"f1-score"],4),round(data.loc[model,"f1-score"],4)) for step, dat in table_list])/len(table_list))}>{round(data.loc[model,"f1-score"],4)}</td>'+
                '</tr> <tr >') if i < len(table_list) - 1 else (
                modelLines + 
                f'<td>{step}</td><td style={"color:blue; font-size: 40px; font-weight: bold;" * int(sum([ round(data.loc[model,"accuracy"],4) == max(round(dat.loc[model,"accuracy"],4),round(data.loc[model,"accuracy"],4)) for step, dat in table_list])/len(table_list))} >{round(data.loc[model,"accuracy"],4)}</td>'+
                f'<td style={"color:blue; font-size: 40px; font-weight: bold;" * int(sum([ round(data.loc[model,"precision"],4) == max(round(dat.loc[model,"precision"],4),round(data.loc[model,"precision"],4)) for step, dat in table_list])/len(table_list))}>{round(data.loc[model,"precision"],4)}</td>'+
                f'<td style={"color:blue; font-size: 40px; font-weight: bold;" * int(sum([ round(data.loc[model,"recall"],4) == max(round(dat.loc[model,"recall"],4),round(data.loc[model,"recall"],4)) for step, dat in table_list])/len(table_list))}>{round(data.loc[model,"recall"],4)}</td>'+
                f'<td style={"color:blue; font-size: 40px; font-weight: bold;" * int(sum([ round(data.loc[model,"f1-score"],4) == max(round(dat.loc[model,"f1-score"],4),round(data.loc[model,"f1-score"],4)) for step, dat in table_list])/len(table_list))}>{round(data.loc[model,"f1-score"],4)}</td>'+
                '</tr>' )
        body = body + modelLines
    style = '<style>table, th, td {border: 1px solid black;border-collapse: collapse;}</style>'
    table_html = f'<table style="border: 2px solid black; width: 100% !important; background-color: #FFFFFF; color:#000000;">{head}{body}</table>'
    htm = f'<html><head>{style}<title> Summary </title></head><body style="background-color: white;">{table_html}</body></html>'
    return (HTML(htm),htm)

# @profile
def create_file(content, cwd, filename, extension=".html", prefix=None):
    """
    """

    create_domain(cwd+'/reports/')
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename1 = cwd+'/reports'+'/'+filename+timestr+extension
    _file= open(filename1,"w")
    _file.write(content)
    _file.close()

    return filename1

def metrics_analyzer(cols=None, outputs_path=None, cwd=None, data_folder=None, classic_metrics=None, models_name=None):
    """ build relevance results about the datasset
    
    Args:
        - cols: list of qualitative variable in the dataset
        - outputs_path: the path where the experimental results are located
        - 
    
    Returns:
        A dedicated folder with those relevante reports and charts
    
    """
    outputs = {}
    
    if cols != None or classic_metrics != None: # check if cols and classics metrics are filled
        ## analyse of k layer
        head_lambda = lambda x: f'<tr><td rowspan="2" colspan="3">{f}<td><td colspan="4">Classic</td><td colspan="4">Classic - Att</td><td colspan="4">Classic + MLN</td><td colspan="4">Classic + MLN - Att</td></tr><tr class="metrics">'+
                        '<td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-score</td>' * 4 +
                        '</tr>'
        head = {
            'xgb': head_lambda('xgb'),
            'dtc': head_lambda('dtc'),
            'lrc': head_lambda('lrc'),
            'rfc': head_lambda('rfc')
        }
        body = {
            'xgb': '',
            'dtc': '',
            'lrc': '',
            'rfc': ''
        }
        style = '<style>table, th, td {border: 1px solid black;border-collapse: collapse;} .wrap-text {word-wrap: break-word;}'+
            ' .wrap-text {overflow-wrap: break-word;} .limited-column {width: 100px;} .dashed-border {border: 1px dashed black;}.dotted-border {border: 1px dotted black;}'+
            '</style>'
        
        caption_content_lambda = lambda x: ''.join(['<span><strong>{key}</strong>: {value}</span><br>' for key, value in {
            '{x}': models_name[x],
            'MLN': 'MultiLayer Network',
            'MLN k Layer(s)': 'MLN with k layer(s)',
            'Att': 'Attributs or modalities of variable(s) used to build MLN',
            'Desc': 'Descriptors extracted from MLN',
            'Classic': f'Learning from classic dataset of {data_folder}',
            'Classic - Att': f'Learning from classic dataset of {data_folder} where Att had been removed',
            'Classic + Desc': f'Learning from classic dataset of {data_folder} where Desc had been added',
            'Classic + Desc - Att': f'Learning from classic dataset of {data_folder} where Desc had been added and Att removed'
            }.items()])

        for k in list(set([1, 2, len(cols)])):
            #for logic in ['global', 'personalized']:
            #outputs[logic] = {}
            ### get all combination of col
            body_l = {
                'xgb': '',
                'dtc': '',
                'lrc': '',
                'lrc': ''
            }
            LayerLines = f'<tr><td rowspan="{len(get_combinations(range(len(cols)),k)) * 2}" >MLN {k} layer(s)</td>'
            
            for layer_config in get_combinations(range(len(cols)),k): # create subsets of k index of OHE and fetch it
                col_targeted= [f'{cols[i]}' for i in layer_config]
                case_k= '±'.join(col_targeted) if len(layer_config)>1 else col_targeted[0]
                #if sum(
                #        [
                #            re.sub(r'[^\w\s]', '', unidecode(partern)) in re.sub(r'[^\w\s]', '', unidecode('cb_person_default_on_file±loan_intent'))
                #            for partern in case_k.split("±")
                #            ]
                #        ) == k: # check mission context
                #    continue
                print(case_k)
                
                VarLines = f'<tr><td rowspan="2" >{case_k}</td>'
                
                ### get files for distincts logic
                match= lambda x: (
                    sum(
                        [
                            re.sub(r'[^\w\s]', '', unidecode(partern)) in re.sub(r'[^\w\s]', '', unidecode(x))
                            for partern in case_k.split("±")
                            ]
                        ) == k if k > 1 else re.sub(r'[^\w\s]', '', unidecode(case_k)) in re.sub(r'[^\w\s]', '', unidecode(x))
                    )
                files = {
                    'global':{
                        'classic': classic_metrics,
                        'classic_-_mlna':[
                            load_data_set_from_url(path=file,sep='\t', encoding='utf-8',index_col=0, na_values=None) 
                            for file in get_filenames(
                                root_dir=f'{outputs_path}/qualitative/mlna_{k}/data_selection_storage', 
                                func=lambda x: ((MLN_C_F(x)) and (match(x))), 
                                verbose=False
                                )
                            ][-1],
                        'classic_mln':[
                            load_data_set_from_url(path=file,sep='\t', encoding='utf-8',index_col=0, na_values=None) 
                            for file in get_filenames(
                                root_dir=f'{outputs_path}/qualitative/mlna_{k}/global/data_selection_storage', 
                                func=lambda x: ((MLN_F(x)) and (match(x))), 
                                verbose=False
                                )
                            ][-1],
                        'classic_mln_-_mlna':[
                            load_data_set_from_url(path=file,sep='\t', encoding='utf-8',index_col=0, na_values=None) 
                            for file in get_filenames(
                                root_dir=f'{outputs_path}/qualitative/mlna_{k}/global/data_selection_storage', 
                                func=lambda x: ((MLN__F(x)) and (match(x))), 
                                verbose=False
                                )
                            ][-1]
                    },
                    "personalized":{
                        'classic': classic_metrics,
                        'classic_-_mlna':[
                            load_data_set_from_url(path=file,sep='\t', encoding='utf-8',index_col=0, na_values=None) 
                            for file in get_filenames(
                                root_dir=f'{outputs_path}/qualitative/mlna_{k}/data_selection_storage', 
                                func=lambda x: ((MLN_C_F(x)) and (match(x))), 
                                verbose=False
                                )
                            ][-1],
                        'classic_mln':[
                            load_data_set_from_url(path=file,sep='\t', encoding='utf-8',index_col=0, na_values=None) 
                            for file in get_filenames(
                                root_dir=f'{outputs_path}/qualitative/mlna_{k}/personalized/data_selection_storage', 
                                func=lambda x: ((MLN_F(x)) and (match(x))), 
                                verbose=False
                                )
                            ][-1],
                        'classic_mln_-_mlna':[
                            load_data_set_from_url(path=file,sep='\t', encoding='utf-8',index_col=0, na_values=None) 
                            for file in get_filenames(
                                root_dir=f'{outputs_path}/qualitative/mlna_{k}/personalized/data_selection_storage', 
                                func=lambda x: ((MLN__F(x)) and (match(x))), 
                                verbose=False
                                )
                            ][-1]
                    }
                }
                # print(files)
                #outputs[logic] = files
                ### transform and normalize
                models_list = files['personalized']['classic'].index.values.tolist()
                print(models_list)
                metrics = ["accuracy","precision","recall","f1-score"]
                
                for model in models_list:
                    max_g = {metric:
                        max([round(files['global'][key].loc[model,metric],4) for key in files['global'].keys()]) for metric in metrics
                        }
                    max_p = {metric:
                        max([round(files['personalized'][key].loc[model,metric],4) for key in files['personalized'].keys()]) for metric in metrics
                        }
                    data = {
                        'global':'<td>Global</td>',
                        'personalized':'<tr><td>Global</td>'
                    }
                    for key in files['global'].keys():
                        data['global']+= ''.join([ 
                            f"<td style={'color:blue; font-size: 40px; font-weight: bold;' * round(files['global'][key].loc[model,metric],4) == max_g[metric]}>{round(files['global'][key].loc[model,metric],4)}</td>" 
                            for metric in metrics
                            ])
                        data['personalized']+= ''.join([ 
                            f"<td style={'color:blue; font-size: 40px; font-weight: bold;' * round(files['personalized'][key].loc[model,metric],4) == max_p[metric]}>{round(files['personalized'][key].loc[model,metric],4)}</td>" 
                            for metric in metrics
                            ])
                    data['global']+= '</tr>'
                    data['personalized']+= '</tr>'
                    body_l[model] += (LayerLines + VarLines + data['global'] + data['personalized'])  if len(body_f[model]) == 0 else (VarLines + data['global'] + data['personalized'])
            
            for model in models_name.keys():
                caption = f'<caption><h2>Legend</h2>{caption_content_lambda(model)}</caption>'
                table_html = f'<table style="border: 2px solid black; width: 100% !important; background-color: #FFFFFF; color:#000000;">{caption}{head}{body}</table>'
                htm = f'<html><head>{style}<title> Summary </title></head><body style="background-color: white;">{table_html}</body></html>'

                create_domain(cwd+'/reports/')
                timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
                filename1 = cwd+'/reports'+'/'+filename+timestr+extension
                _file= open(filename1,"w")
                _file.write(content)
                _file.close()        
                


                ### generate figures
                nrow = 2
                ncol = 4
                
                fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, 15))
                count=0
                for r in range(ncol): 
                    # Barplot i
                    models[models_list[count]].plot(kind='bar', x='Metrics', y=files['global'].keys(), ax=axs[0,r])
                    axs[0,r].set_title(f'{models_name[models_list[count]]}')
                    axs[0,r].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    models_1[models_list[count]].plot(kind='bar', x='Metrics', y=files['personalized'].keys(), ax=axs[1,r])
                    axs[1,r].set_title(f'{models_name[models_list[count]]}')
                    axs[1,r].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    axs[0,r].legend().set_visible(False)
                    axs[1,r].legend().set_visible(False)
                    if r == ncol-1:
                        axs[1,r].legend().set_visible(True)
                    count+=1
                axs[0,0].set_ylabel('global')
                axs[1,0].set_ylabel('personalized')
                if True:
                    create_domain(f'{cwd}/analyser/{data_folder}/plots/mlna_{k}/mixed')

                    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
                    filename1 = f'{cwd}/analyser/{data_folder}/plots/mlna_{k}/mixed/_metrics_comparaison_for_{case_k}'+'_'+timestr+'.png'
                    # Adjust the layout to cover all content
                    plt.tight_layout()

                    plt.savefig(filename1,dpi=150) #.png,.pdf will also support here
                    plt.close() # close the plot windows
                ### generate reports
    return outputs


