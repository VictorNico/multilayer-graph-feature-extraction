"""
    Author: VICTOR DJIEMBOU
    addedAt: 28/11/2023
    changes:
        - 25/11/2023:
            - add matrix_2d_test, generate_entity_nodes_list, build_mlg, find_indices, build_modalities_graph, get_number_of_borrowers_with_same_second_layer_value,
            get_intra_node_label, get_inter_node_label, compute_personlization, get_number_of_borrowers_with_same_first_and_same_second_layer_value,
            get_max_borrower_pr, get_max_modality_pagerank_score, get_persons, standard_extraction, inject_features_extracted methods, get_number_of_borrowers_with_same_first_layer_value
        - 28/11/2023:
            - add get_number_of_borrowers_with_same_n_layer_value, get_number_of_borrowers_with_same_custom_layer_value
            - remove get_number_of_borrowers_with_same_first_layer_value, get_number_of_borrowers_with_same_first_and_same_second_layer_value, get_number_of_borrowers_with_same_second_layer_value
"""
#################################################
##          Libraries importation
#################################################

###### Begin

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import collections
from itertools import combinations

###### End


#################################################
##          Methods definition
#################################################

def matrix_2d_test(matrix):
    """Check if Matrix is 2D
    Args:
      matrix: 2D Matrix to check out

    Returns:
      True if it's a 2D Matrix anf False else
    """
    
    flag = False
    
    if isinstance(matrix, list):
        if isinstance(matrix[0], list):
            intern_flag =True
            for row in matrix:
                if len(row) != len(matrix):
                    intern_flag = False
            flag = intern_flag
            
    return flag

def generate_entity_nodes_list(data):
    """Generate entity nodes list
    Args:
      data: a source of data which be a dataframe, series, list or 2D matrix

    Returns:
      NODE_LIST
    """
    
    NB_ENTITY = 0
    NODE_LIST = []
    if isinstance(data, pd.DataFrame):
        NB_ENTITY = data.shape[0]
        
    if isinstance(data, pd.Series):
        NB_ENTITY = data.size
        
    """if isinstance(data, list):
        NB_ENTITY = len(data)
    if isinstance(data, list):
        if isinstance(data[0], list):
            if matrix_2d_test(data):"""
    
    NODE_LIST = [i for i in range(NB_ENTITY)]
    
    return NODE_LIST

def build_mlg(data, features):
    """Build a multi-layer graph
    Args:
      data: a dataframe
      features: a list of dimension

    Returns:
      A directed graph
    """
    
    CRP_G = nx.DiGraph() # create an empty directed graph

    # build edges
    list_of_edges = []
    list_of_nodes = []
    
    LIST_OF_CUSTOMERS = data.index.values.tolist()
    LEN_OF_FEATURES = len(features)
    
    colors = [
        '#e6194b',
        '#ffe119',
        '#4363d8',
        '#f58231',
        '#911eb4',
        '#46f0f0',
        '#f032e6',
        '#bcf60c',
        '#fabebe',
        '#008080',
        '#e6beff',
        '#9a6324',
        '#fffac8',
        '#800000',
        '#aafdc9',
        '#808000',
        '#ffd8b1',
        '#000075',
        '#9cb44b',
        '#808080'

    ]
    
    for el in LIST_OF_CUSTOMERS: #fetch on custumers list
        # layer building
        for i in range(LEN_OF_FEATURES):
            # add nodes
            list_of_nodes.append(('C'+str(i)+'-U-'+str(el),{'color': 'g'}))
            for attr in features[i].tolist(): # fetch on home ownership encode values
                code = f"#{format(255-10*i, '02x')}{format(150+9*i, '02x')}{format(55+10*i, '02x')}"
                if int(data.loc[el,attr]) == 1: # check if exists relation between both
                    # bidirectional relation between home ownership and user
                    list_of_edges.append(('C'+str(i)+'-U-'+str(el),'C'+str(i)+'-M-'+attr, {'color': 'b'})) # add edge to list
                    list_of_edges.append(('C'+str(i)+'-M-'+attr, 'C'+str(i)+'-U-'+str(el), {'color': 'b'})) # add edge to list
                    # add nodes
                    list_of_nodes.append(('C'+str(i)+'-M-'+attr,{'color': colors[i]}))
            # add directed relation between user node from C1 and C2
            list_of_edges.append(('C'+str(i)+'-U-'+str(el),'C'+str(i+1 if i+1 < LEN_OF_FEATURES else i-1)+'-U-'+str(el), {'color': 'r'})) # add edge to list
            list_of_edges.append(('C'+str(i+1 if i+1 < LEN_OF_FEATURES else i-1 )+'-U-'+str(el), 'C'+str(i)+'-U-'+str(el), {'color': 'r'})) # add edge to list

    # add edges to the oriented graph
    # print(list_of_nodes)
    CRP_G.add_nodes_from(list_of_nodes)
    CRP_G.add_edges_from(list_of_edges)

    # return the graph
    return CRP_G

def find_indices(list_to_check, item_to_find):
    """Find indices
    Args:
      list_to_check: a list of binary value
      item_to_find: a value

    Returns:
      A list of indices or index
    """
    
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]

def build_modalities_graph(X,Y,n):
    """Build a modalities graph
    Args:
      X: a dataframe without class
      Y: a vector of class values
      n: the number of rows

    Returns:
      A modality graph
    """
    
    CRP_G = nx.DiGraph() # create an empty directed graph

    # join X and Y
    data = X.copy()
    data.astype('int')
    temp = data.copy()
    data["CLASS"] = Y.values
    data = data.head(n)

    # cast columns type to int
    data.astype('int')



    # build edges
    LIST_OF_CUSTOMERS = data.index.values.tolist()
    
    for el in LIST_OF_CUSTOMERS: #fetch on custumers list
        # find all columns belong to customer
        LINE = temp.loc[el,].values
        COLUMNS_BELONG = find_indices(LINE,1)

        # create edges
        COLUMNS = X.columns
        for i, col in enumerate(COLUMNS_BELONG): # fetch belong columns
            #print([i for i in range(i+1,len(COLUMNS_BELONG))], COLUMNS_BELONG)
            for j in range(i+1,len(COLUMNS_BELONG)): # fetch successor
                #if j < len(COLUMNS_BELONG) - 1: # check if it's the last column
                if CRP_G.has_edge(COLUMNS[col], COLUMNS[COLUMNS_BELONG[j]]):
                    # we added this one before, just increase the weight by one
                    CRP_G[COLUMNS[col]][COLUMNS[COLUMNS_BELONG[j]]]['weight'] += 1
                else:
                    # new edge. add with weight=1
                    CRP_G.add_edge(COLUMNS[col], COLUMNS[COLUMNS_BELONG[j]], weight=1)

                if CRP_G.has_edge(COLUMNS[COLUMNS_BELONG[j]], COLUMNS[col]):
                    # we added this one before, just increase the weight by one
                    CRP_G[COLUMNS[COLUMNS_BELONG[j]]][COLUMNS[col]]['weight'] += 1
                else:
                    # new edge. add with weight=1
                    CRP_G.add_edge(COLUMNS[COLUMNS_BELONG[j]], COLUMNS[col], weight=1)

            """if i == len(COLUMNS_BELONG) - 2: # check if it's the last column
                if CRP_G.has_edge(COLUMNS[col], COLUMNS[COLUMNS_BELONG[i+1]]):
                    # we added this one before, just increase the weight by one
                    CRP_G[COLUMNS[col]][COLUMNS[COLUMNS_BELONG[i+1]]]['weight'] += 1
                else:
                    # new edge. add with weight=1
                    CRP_G.add_edge(COLUMNS[col], COLUMNS[COLUMNS_BELONG[i+1]], weight=1)

                if CRP_G.has_edge(COLUMNS[COLUMNS_BELONG[i+1]], COLUMNS[col]):
                    # we added this one before, just increase the weight by one
                    CRP_G[COLUMNS[COLUMNS_BELONG[i+1]]][COLUMNS[col]]['weight'] += 1
                else:
                    # new edge. add with weight=1
                    CRP_G.add_edge(COLUMNS[COLUMNS_BELONG[i+1]], COLUMNS[col], weight=1)"""



            # add to class
            CLASS = data.loc[el,'CLASS']
            if CRP_G.has_edge(COLUMNS[col], CLASS):
                # we added this one before, just increase the weight by one
                CRP_G[COLUMNS[col]][CLASS]['weight'] += 1
            else:
                # new edge. add with weight=1
                CRP_G.add_edge(COLUMNS[col], CLASS, weight=1)

            if CRP_G.has_edge(CLASS, COLUMNS[col]):
                # we added this one before, just increase the weight by one
                CRP_G[CLASS][COLUMNS[col]]['weight'] += 1
            else:
                # new edge. add with weight=1
                CRP_G.add_edge(CLASS, COLUMNS[col], weight=1)




    # return the graph
    return CRP_G, data

def get_intra_node_label(graph):
    """Get intra nodel labels
    Args:
      graph: a multilayer graph

    Returns:
      A list of all intra nodes inside the graph
    """
    
    nodes = graph.nodes()
    intras = [k for k in nodes if '-M-' in k]
    return intras

def get_inter_node_label(graph):
    """Get inter nodel labels
    Args:
      graph: a multilayer graph

    Returns:
      A list of all inter nodes inside the graph
    """
    
    nodes = graph.nodes()
    inters = [k for k in nodes if '-U-' in k]
    return inters

def get_user_nodes_label(graph,borrowers, layers):
    """Get borrower liked nodel labels
    Args:
      graph: a multilayer graph
      layers: the number of layers

    Returns:
      A list of all inter nodes inside the graph
    """
    linked_table= []
    
    for borrower in borrowers:
        edges = [(A,B) for i in range(layers) for (A,B) in graph.edges(['C'+str(i)+'-U-'+str(borrower)])]
        linked = set()
        for A, B in edges: # for each edge of the borower in the layer 
            linked.add(A)
            linked.add(B)

        linked_table.append(list(linked)) # convert to list of node label
    return (linked_table, set(tuple(borr) for borr in linked_table))

def compute_personlization(node_list):
    """Compute personalization
    Args:
      node_list: nodes list

    Returns:
      A list of initial influence of nodes
    """
    
    personlized = dict()
    a = [ personlized.update({ k : 1/len(node_list) }) for k in node_list ]
    return personlized


def get_number_of_borrowers_with_same_n_layer_value(borrower, graph, layer_nber=0):
    """Get number of borrower with same value in first layer
    Args:
      borrower: index of borrower
      graph: a graph
      layer_nber: the layer number. Default layer 0

    Returns:
      A list of list of borrower and their length
    """
    
    edge = [
          B # return the modaliy
          for (A,B) in graph.edges(['C'+str(layer_nber)+'-U-' + str(borrower)]) # for each nodes link to my borower's
          if ('C'+str(layer_nber)+'-M-' in B) and ('C'+str(layer_nber)+'-U-' + str(borrower) == A) # if clause is respect
      ]
    #print(edge[0][1])
    edges = [
      (A,B) 
      for (A,B) in graph.edges(edge)
      ]
    #print(edges)
    return [edges,len(edges) - 1]

def get_max_borrower_pr(pr):
    """Get max modality of borrower in the pagerank output
    Args:
      pr: a specify pagerank

    Returns:
      A list of max modality pagerank for each borrower
    """
    
    borrower = {}
    for key, val in pr.items():
        if '-U-' in key:
            borrower[int(key.split("-U-")[1])] = max(val, borrower[int(key.split("-U-")[1])]) if int(key.split("-U-")[1]) in borrower else val
    return [val for key, val in collections.OrderedDict(sorted(borrower.items())).items()]

def get_number_of_borrowers_with_same_custom_layer_value(borrower, graph, custom_layer=[0,1]):
    """Get number of borrower with same value in first and second layer
    Args:
      borrower: index of borrower
      graph: a graph
      custom_layer: a list f focus analystic layer

    Returns:
      A list of list of example and their length
    """
    
    edges = {'edges_l'+str(l): get_number_of_borrowers_with_same_n_layer_value(borrower, graph,l)[0] for l in custom_layer} # get the number of borrower with same modality inside each layer
    just_borrowers = {}
    for (key, value) in edges.items(): # get index of borrower only inside each layer
        just_borrowers[key] = [B.split("-U-")[1] for (A,B) in value]

    for (key, value) in just_borrowers.items(): # convert to set of index borrower
        just_borrowers[key] = set(value)

    intersections = set() 
    for (key, value) in just_borrowers.items(): # fetch sets of borrowers in each layer
        intersections = intersections.intersection(value) # find the shared borrowers

    return [
        list(intersections), # convert intersection set to list 
        len(intersections) - 1 # compute the len 
    ]

def get_max_modality_pagerank_score(borrower, graph, k, pr):
    """Get max modality of borrower in the pagerank output belong to specify layer
    Args:
        borrower: indice of the borrower
        graph: the graph mln
        k: the number of layer      
        pr: a specify pagerank

    Returns:
        The max score
    """
    
    edges = [(A,B) for i in range(k) for (A,B) in graph.edges(['C'+str(i)+'-U-'+str(borrower)])] # get all edges of the borrower
    maxi = 0 # default maxi to 0
    for A, B in edges: # for each edge of the borower in the layer
        if sum(['-U-'+str(borrower) in A and 'C'+str(i)+'-M-' in B for i in range(k)]) > 0: # verify the form
            maxi = maxi if max(maxi,pr[B]) == maxi else pr[B] # update the max
            #break
    return maxi #return the max

def get_persons(dataframe):
    """Get index dimension of dataframe
    Args:
        dataframe: a dataframe

    Returns:
        The index list
    """
    
    index_users = dataframe.index.values.tolist() # get all index present in the dataset as list
    return index_users # return it

def standard_extraction(extracts, feats):
    """Standardize the features extracted
    Args:
        extracts: features dict
        feats: list of keys feature to standardize 

    Returns:
        None
    """
    
    for key in feats:
        extracts[key] = [el/max(extracts[key]) for el in extracts[key]]

def inject_features_extracted(data,features):
    """inject features extracted in dataframe
    Args:
        data: dataframe of loan
        features: dict of features extracted 

    Returns:
        A new loan dataset
    """
    
    dataframe = data.copy()
    for key, val in features.items():
        dataframe[key] = val
    return dataframe

