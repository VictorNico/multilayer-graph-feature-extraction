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

import sys
import pandas as pd
import time
import os
import joblib
from networkx import write_gml, write_graphml_lxml

###### End


#################################################
##          Methods definition
#################################################

def load_data_set_from_url(path, sep='\t', encoding='utf-8',index_col=None):
    """Read dataset from multi format
    Args:
      path: path to dataset
      sep: the delimiter in the file
      encoding: encoding used to save file
      index_col: column with index values

    Returns:
      The dataset instance loaded
    """
    
    extension = os.path.splitext(path)[1] if isinstance(path, str) else None

    if extension == None:
        extension = '.csv'

    print(f"file ext know as {extension}")

    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".json": pd.read_json,


    }

    dataset = (readers[extension](path, sep=sep, encoding='utf-8',index_col=index_col) if '.csv' in extension else readers[extension](path, index_col=index_col)) if readers[extension] else f"no reader define for the extension {extension}"

    return dataset

def save_model(cwd, clf, prefix, clf_name):
    """Save model
    Args:
      clf: An instance of model
      clf_name: their name

    Returns:
      None
    """
    create_domain(cwd+'/model_storage/'+prefix) # create the domaine folder

    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename = cwd+'/model_storage/'+prefix+'/'+clf_name+'_'+timestr+'.sav'
    joblib.dump(clf, filename)

    return filename

def save_graph(cwd, graph, name, rows_len, prefix, cols_len):
    """Save graph
    Args:
      graph: An instance of graph
      name: their name
      rows_len: number of examples used to build
      cols_len: number of dimension used

    Returns:
      None
    """
    
    create_domain(cwd+'/graph_storage/'+prefix)

    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename = cwd+'/graph_storage/'+prefix+'/'+name+'_'+str(rows_len)+'_'+str(cols_len)+'_'+timestr+'.gml.gz'
    write_gml(graph, filename)

    return filename

def save_digraph(cwd, graph, name, rows_len, cols_len, prefix=None):
    """Save directed graph
    Args:
      graph: An instance of digraph
      name: their name
      rows_len: number of examples used to build
      cols_len: number of dimension used

    Returns:
      None
    """

    create_domain(cwd+'/graph_storage/')
    
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename = cwd+'/graph_storage/'+name+'_'+str(rows_len)+'_'+str(cols_len)+'_'+timestr+'.gml.gz'
    write_graphml_lxml(graph, filename)

    return filename
    
def save_dataset(cwd, dataframe, name, prefix=None, sep='\t'):
    """Save a dataframe
    Args:
      dataframe: An instance of dataframe
      name: their name
      sep: the separator

    Returns:
      None
    """

    create_domain(cwd+'/data_selection_storage')
    
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename = cwd+'/data_selection_storage'+'/'+name+'_'+timestr+'.csv'
    dataframe.to_csv(filename, sep=sep, encoding='utf-8')

    return filename
    
def read_model(path):
    """Read model
    Args:
      path: path to model

    Returns:
      The model instance saved previously
    """
    
    return joblib.load(path)

def read_graph(path):
    """Read graph
    Args:
      path: path to graph

    Returns:
      The graph instance saved previously
    """
    
    return nx.read_gml(path)

def read_digraph(path):
    """Read directed graph
    Args:
      path: path to digraph

    Returns:
      The digraph instance saved previously
    """
    
    return nx.read_graphml_lxml(path)
    
def read_dataset(path,sep='\t'):
    """Read graph
    Args:
      path: path to dataset
      sep: the separator

    Returns:
      The dataset instance saved previously as dataframe
    """
    
    return pd.read_csv(path, sep=sep, encoding='utf-8',index_col=0)

def create_domain(directory, verbose=True):
    """Create a domain anaysis directory
    Args:
      directory: path to director
      verbose: whether a printing console is need

    Returns:
      The state of creation
    """

    state=False
    try:
        os.makedirs(directory)
        state=True
        print(f"Directory '{directory}' created successfully.") if verbose else None
    except FileExistsError:
        print(f"Directory '{directory}' already exists.") if verbose else None
    except OSError as e:
        print(f"An error occurred while creating directory '{directory}': {e}") if verbose else None
    return state

