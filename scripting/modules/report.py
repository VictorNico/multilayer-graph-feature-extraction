    
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

