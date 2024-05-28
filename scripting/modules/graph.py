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
import matplotlib as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import collections
# import dask
# import pyopencl as cl
# import dask.dataframe as dd
# from dask.distributed import Client
# import dask.array as da
from itertools import combinations
import copy
from .file import *


# from memory_profiler import profile

###### End


#################################################
##          Methods definition
#################################################
# @profile
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
            intern_flag = True
            for row in matrix:
                if len(row) != len(matrix):
                    intern_flag = False
            flag = intern_flag

    return flag


# @profile
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


# @profile
def build_mlg(data, features):
    """Build a multi-layer graph
    Args:
      data: a dataframe
      features: a list of dimension

    Returns:
      A directed graph
    """

    CRP_G = nx.DiGraph()  # create an empty directed graph

    # build edges
    list_of_edges = []
    list_of_nodes = []

    LIST_OF_CUSTOMERS = get_persons(data)
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

    for el in LIST_OF_CUSTOMERS:  #fetch on custumers list
        # layer building
        for i in range(LEN_OF_FEATURES):
            # add nodes
            list_of_nodes.append(('C' + str(i) + '-U-' + str(el), {'color': 'g'}))
            for attr in features[i].tolist():  # fetch on home ownership encode values
                code = f"#{format(255 - 10 * i, '02x')}{format(150 + 9 * i, '02x')}{format(55 + 10 * i, '02x')}"
                # print(f"{el} <-{data.loc[el,attr]}-> {attr}")
                if int(data.loc[el, attr]) == 1:  # check if exists relation between both
                    # print(f"{el} <--> {attr}")
                    # bidirectional relation between home ownership and user
                    list_of_edges.append(('C' + str(i) + '-U-' + str(el), 'C' + str(i) + '-M-' + attr,
                                          {'color': 'b'}))  # add edge to list
                    list_of_edges.append(('C' + str(i) + '-M-' + attr, 'C' + str(i) + '-U-' + str(el),
                                          {'color': 'b'}))  # add edge to list
                    # add nodes
                    list_of_nodes.append(('C' + str(i) + '-M-' + attr, {'color': colors[i]}))
            # add directed relation between user node from C1 and C2
            if i + 1 < LEN_OF_FEATURES:
                list_of_edges.append(('C' + str(i) + '-U-' + str(el),
                                      'C' + str(i + 1) + '-U-' + str(el),
                                      {'color': 'r'}))  # add edge to list
                list_of_edges.append(('C' + str(i + 1) + '-U-' + str(el),
                                      'C' + str(i) + '-U-' + str(el), {'color': 'r'}))  # add edge to list

    # add edges to the oriented graph
    # print(list_of_nodes)
    CRP_G.add_nodes_from(list_of_nodes)
    CRP_G.add_edges_from(list_of_edges)

    # return the graph
    return CRP_G


def add_specific_loan_in_mlg(graph, data, features):
    """
    Add specific loan in graph
    Parameters
    ----------
    graph
    data
    features

    Returns
    -------
    New graph with specific loan in graph
    """

    CRP_G = copy.deepcopy(graph)  # create an empty directed graph

    # build edges
    list_of_edges = []
    list_of_nodes = []

    LIST_OF_CUSTOMERS = get_persons(data)
    LEN_OF_FEATURES = len(features)
    # print(f"{LIST_OF_CUSTOMERS} - {features}")

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

    for el in LIST_OF_CUSTOMERS:  #fetch on custumers list
        # layer building
        for i in range(LEN_OF_FEATURES):
            # add nodes
            list_of_nodes.append(('C' + str(i) + '-U-' + str(el), {'color': 'g'}))
            for attr in features[i].tolist():  # fetch on home ownership encode values
                code = f"#{format(255 - 10 * i, '02x')}{format(150 + 9 * i, '02x')}{format(55 + 10 * i, '02x')}"
                # print(f"{el} <- {data.loc[el,attr]} -> {attr}")
                if int(data.loc[el, attr]) == 1:  # check if exists relation between both
                    # print(f"{el} <--> {attr}")
                    # bidirectional relation between attribut and user
                    list_of_edges.append(('C' + str(i) + '-U-' + str(el), 'C' + str(i) + '-M-' + attr,
                                          {'color': 'b'}))  # add edge to list
                    list_of_edges.append(('C' + str(i) + '-M-' + attr, 'C' + str(i) + '-U-' + str(el),
                                          {'color': 'b'}))  # add edge to list
                    # add nodes
                    list_of_nodes.append(('C' + str(i) + '-M-' + attr, {'color': colors[i]}))
            # add directed relation between user node from C1 and C2
            if i + 1 < LEN_OF_FEATURES:
                list_of_edges.append(('C' + str(i) + '-U-' + str(el),
                                      'C' + str(i + 1) + '-U-' + str(el),
                                      {'color': 'r'}))  # add edge to list
                list_of_edges.append(('C' + str(i + 1) + '-U-' + str(el),
                                      'C' + str(i) + '-U-' + str(el), {'color': 'r'}))  # add edge to list

    # add edges to the oriented graph
    # print(list_of_nodes)
    CRP_G.add_nodes_from(list_of_nodes)
    CRP_G.add_edges_from(list_of_edges)
    # print(f"{list_of_nodes} - {list_of_edges}")
    return CRP_G

# @profile
def find_indices(list_to_check, item_to_find):
    """Find indices
    Args:
      list_to_check: a list of binary value
      item_to_find: a value

    Returns:
      A list of indices or index
    """

    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]


# @profile
def build_modalities_graph(X, Y, n):
    """Build a modalities graph
    Args:
      X: a dataframe without class
      Y: a vector of class values
      n: the number of rows

    Returns:
      A modality graph
    """

    CRP_G = nx.DiGraph()  # create an empty directed graph

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

    for el in LIST_OF_CUSTOMERS:  #fetch on custumers list
        # find all columns belong to customer
        LINE = temp.loc[el,].values
        COLUMNS_BELONG = find_indices(LINE, 1)

        # create edges
        COLUMNS = X.columns
        for i, col in enumerate(COLUMNS_BELONG):  # fetch belong columns
            #print([i for i in range(i+1,len(COLUMNS_BELONG))], COLUMNS_BELONG)
            for j in range(i + 1, len(COLUMNS_BELONG)):  # fetch successor
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
            CLASS = data.loc[el, 'CLASS']
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


# @profile
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


# @profile
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


# @profile
def get_combine_perso_nodes_label(graph, borrowers, layers):
    """Get borrower liked nodel labels
    Args:
      graph: a multilayer graph
      layers: the number of layers

    Returns:
      A list of all inter nodes inside the graph
    """
    linked_table = []
    # print(nx.number_of_isolates(graph))
    for borrower in borrowers:
        edges = [(A, B) for i in range(layers) for (A, B) in graph.edges(['C' + str(i) + '-U-' + str(borrower)])]
        linked = set()
        for A, B in edges:  # for each edge of the borower in the layer
            linked.add(A)
            linked.add(B)

        linked_table.append(list(linked))  # convert to list of node label
    return (linked_table, set(tuple(borr) for borr in linked_table))


# @profile
def get_intra_perso_nodes_label(graph, borrowers, layers):
    """Get borrower liked nodel labels
    Args:
      graph: a multilayer graph
      layers: the number of layers

    Returns:
      A list of all inter nodes inside the graph
    """
    linked_table = []
    # print(f"---------{borrowers}")
    for borrower in borrowers:
        edges = [(A, B) for i in range(layers) for (A, B) in graph.edges(['C' + str(i) + '-U-' + str(borrower)])]
        linked = set()
        for A, B in edges:  # for each edge of the borower in the layer
            # print(f"{A} -> {B}")
            linked.add(A) if '-M-' in A else None
            linked.add(B) if '-M-' in B else None

        linked_table.append(list(linked))  # convert to list of node label
        #print(linked_table)
    return (linked_table, set(tuple(borr) for borr in linked_table))


# @profile
def get_inter_perso_nodes_label(graph, borrowers, layers):
    """Get borrower liked nodel labels
    Args:
      graph: a multilayer graph
      layers: the number of layers

    Returns:
      A list of all inter nodes inside the graph
    """
    linked_table = []

    for borrower in borrowers:
        edges = [(A, B) for i in range(layers) for (A, B) in graph.edges(['C' + str(i) + '-U-' + str(borrower)])]
        linked = set()
        for A, B in edges:  # for each edge of the borower in the layer
            linked.add(A) if '-U-' in A else None
            linked.add(B) if '-U-' in B else None

        linked_table.append(list(linked))  # convert to list of node label
    return (linked_table, set(tuple(borr) for borr in linked_table))


# @profile
def compute_personlization(node_list, graph):
    """Compute personalization
    Args:
      node_list: nodes list

    Returns:
      A list of initial influence of nodes
    """

    personlized = dict()
    # print(len(node_list))
    #a = [ personlized.update({ k : 1/len(node_list) }) for k in node_list ]
    a = [personlized.update({k: (1 / len(node_list)) if k in node_list else 0}) for k in graph.nodes()]
    # print(personlized)
    return personlized


# @profile
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
        B  # return the modaliy
        for (A, B) in graph.edges(['C' + str(layer_nber) + '-U-' + str(borrower)])
        # for each nodes link to my borower's
        if ('C' + str(layer_nber) + '-M-' in B) and ('C' + str(layer_nber) + '-U-' + str(borrower) == A)
        # if clause is respect
    ]
    #print(edge[0][1])
    edges = [
        (A, B)
        for (A, B) in graph.edges(edge)
    ]
    #print(edges)
    return [edges, len(edges) - 1]


# @profile
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
            borrower[int(key.split("-U-")[1])] = max(val, borrower[int(key.split("-U-")[1])]) if int(
                key.split("-U-")[1]) in borrower else val
    return [[val for key, val in collections.OrderedDict(sorted(borrower.items())).items()], borrower]


# @profile
def get_class_pr(pr):
    """Get classes score in the pagerank output
    Args:
      pr: a specific pagerank

    Returns:
      A list of classes pagerank
    """

    borrower = {}
    for key, val in pr.items():
        if '-M-C-' in key:
            borrower[int(key.split("-M-C-")[1])] = max(val, borrower[int(key.split("-M-C-")[1])]) if int(
                key.split("-M-C-")[1]) in borrower else val
    return [val for key, val in collections.OrderedDict(sorted(borrower.items())).items()]


# @profile
def get_number_of_borrowers_with_same_custom_layer_value(borrower, graph, custom_layer=[0, 1]):
    """Get number of borrower with same value in first and second layer
    Args:
      borrower: index of borrower
      graph: a graph
      custom_layer: a list f focus analystic layer

    Returns:
      A list of list of example and their length
    """

    edges = {'edges_l' + str(l): get_number_of_borrowers_with_same_n_layer_value(borrower, graph, l)[0] for l in
             custom_layer}  # get the number of borrower with same modality inside each layer

    just_borrowers = {}
    for (key, value) in edges.items():  # get index of borrower only inside each layer
        just_borrowers[key] = [B.split("-U-")[1] for (A, B) in value]

    for (key, value) in just_borrowers.items():  # convert to set of index borrower
        just_borrowers[key] = set(value)

    intersections = set(just_borrowers[list(just_borrowers.keys())[0]])
    for key, value in just_borrowers.items():  # fetch sets of borrowers in each layer
        # print(type(value))
        intersections = intersections.intersection(value)  # find the shared borrowers
    # print(F"""
    #         ---------
    #         {len(edges.keys())} layers: {edges.keys()}
    #         borrower: {borrower}
    #         {just_borrowers.values()}
    #         ----------
    #         {intersections}
    #         ---------
    #         """)
    return [
        list(intersections),  # convert intersection set to list
        len(intersections) - 1  # compute the len
    ]


# @profile
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

    edges = [(A, B) for i in range(k) for (A, B) in
             graph.edges(['C' + str(i) + '-U-' + str(borrower)])]  # get all edges of the borrower
    maxi = 0  # default maxi to 0
    for A, B in edges:  # for each edge of the borower in the layer
        if sum(['-U-' + str(borrower) in A and 'C' + str(i) + '-M-' in B for i in range(k)]) > 0:  # verify the form
            maxi = maxi if max(maxi, pr[B]) == maxi else pr[B]  # update the max
            #break
    return maxi  #return the max


# @profile
def get_persons(dataframe):
    """Get index dimension of dataframe
    Args:
        dataframe: a dataframe

    Returns:
        The index list
    """

    index_users = dataframe.index.values.tolist()  # get all index present in the dataset as list
    return index_users  # return it


def get_maximun_std_descriptor(extracts_1, extracts_2, feats):
    internalMaxConfig = dict()
    for key in feats:
        internalMaxConfig[key] = max(max(extracts_1[key]),max(extracts_2[key]))

    return internalMaxConfig


# @profile
def standard_extraction(extracts, feats, maxConfig=None):
    """

    Parameters
    ----------
    extracts
    feats
    maxConfig

    Returns
    -------

    """
    internalMaxConfig = maxConfig if maxConfig is not None else dict()
    if len(list(internalMaxConfig.keys())) == 0:
        for key in feats:
                internalMaxConfig[key] = max(extracts[key])
    for key in feats:
        norm = []
        for el in extracts[key]:
            if el >= internalMaxConfig[key]: # if the current element is greater or equal to the  key internalMaxConfig
                norm.append(1)
            else:
                norm.append(el/internalMaxConfig[key])
        extracts[key] = norm
    return internalMaxConfig




# @profile
def inject_features_extracted(data, features):
    """inject features extracted in dataframe
    Args:
        data: dataframe of loan
        features: dict of features extracted 

    Returns:
        A new loan dataset
    """

    dataframe = data.copy()

    if isinstance(features, pd.DataFrame):
        for key in features.columns.values.tolist():
            # print(f"""
            #     {key}: {features[key]}""")
            dataframe[key] = features[key].tolist()
    else:
        for key, val in features.items():
            # print(f"""
            #     {key}: {val}""")
            dataframe[key] = val
    # print(dataframe)
    return dataframe


# show
def plot_digraph(CRP_G, path):
    colors = nx.get_edge_attributes(CRP_G, 'color').values()
    colorsN = nx.get_node_attributes(CRP_G, 'color').values()
    nx.draw(
        CRP_G,
        edge_color=colors,
        node_color=colorsN,
        with_labels=True)
    plt.show()
    plt.savefig(path, dpi=700)
    plt.close()


def build_mlg_with_class(data, features, className):
    """Build a multi-layer graph
    Args:
      data: a dataframe
      features: a list of dimension
      className: the target name

    Returns:
      A directed graph
    """

    CRP_G = nx.DiGraph()  # create an empty directed graph

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

    for el in LIST_OF_CUSTOMERS:  #fetch on custumers list
        # layer building
        for i in range(LEN_OF_FEATURES):
            # add nodes
            list_of_nodes.append(('C' + str(i) + '-U-' + str(el), {'color': 'g'}))
            for attr in features[i].tolist():  # fetch on home ownership encode values
                #code = f"#{format(255-10*i, '02x')}{format(150+9*i, '02x')}{format(55+10*i, '02x')}"
                if int(data.loc[el, attr]) == 1:  # check if exists relation between both
                    # bidirectional relation between home ownership and user
                    list_of_edges.append(('C' + str(i) + '-U-' + str(el), 'C' + str(i) + '-M-' + attr,
                                          {'color': 'b'}))  # add edge to list
                    list_of_edges.append(('C' + str(i) + '-M-' + attr, 'C' + str(i) + '-U-' + str(el),
                                          {'color': 'b'}))  # add edge to list
                    # add nodes
                    list_of_nodes.append(('C' + str(i) + '-M-' + attr, {'color': colors[i]}))
            # add directed relation between user node from C1 and C2
            list_of_edges.append(('C' + str(i) + '-U-' + str(el), 'C' + str(i + 1) + '-U-' + str(el),
                                  {'color': 'r'}))  # add edge to list
            list_of_edges.append(('C' + str(i + 1) + '-U-' + str(el), 'C' + str(i) + '-U-' + str(el),
                                  {'color': 'r'}))  # add edge to list
        #code = f"#{format(255-10*LEN_OF_FEATURES, '02x')}{format(150+9*LEN_OF_FEATURES, '02x')}{format(55+10*LEN_OF_FEATURES, '02x')}"
        # add class nodes
        list_of_nodes.append(('C' + str(LEN_OF_FEATURES) + '-U-' + str(el), {'color': 'g'}))
        list_of_nodes.append(
            ('C' + str(LEN_OF_FEATURES) + '-M-C-' + str(data.loc[el, className]), {'color': colors[LEN_OF_FEATURES]}))
        # bidirectional relation between home ownership and user
        list_of_edges.append(('C' + str(LEN_OF_FEATURES) + '-U-' + str(el),
                              'C' + str(LEN_OF_FEATURES) + '-M-C-' + str(data.loc[el, className]),
                              {'color': 'b'}))  # add edge to list
        list_of_edges.append(('C' + str(LEN_OF_FEATURES) + '-M-C-' + str(data.loc[el, className]),
                              'C' + str(LEN_OF_FEATURES) + '-U-' + str(el), {'color': 'b'}))  # add edge to list
        # add directed relation between user node from C1 and C2
        list_of_edges.append(('C' + str(LEN_OF_FEATURES) + '-U-' + str(el),
                              'C' + str(LEN_OF_FEATURES - 1) + '-U-' + str(el), {'color': 'r'}))  # add edge to list
        list_of_edges.append(('C' + str(LEN_OF_FEATURES - 1) + '-U-' + str(el),
                              'C' + str(LEN_OF_FEATURES) + '-U-' + str(el), {'color': 'r'}))  # add edge to list
    # add edges to the oriented graph
    # print(list_of_nodes)
    CRP_G.add_nodes_from(list_of_nodes)
    CRP_G.add_edges_from(list_of_edges)

    # return the graph
    return CRP_G


# def add_specific_loan_in_mlg_with_class(graph, data, features):
#     """
#     Add a specific loan into the graph
#     Parameters
#     ----------
#     graph
#     data
#     features
#
#     Returns
#     -------
#
#     """
#
#     CRP_G = copy.deepcopy(graph)  # create an empty directed graph
#
#     # build edges
#     list_of_edges = []
#     list_of_nodes = []
#
#     LIST_OF_CUSTOMERS = data.index.values.tolist()
#     LEN_OF_FEATURES = len(features)
#
#     colors = [
#         '#e6194b',
#         '#ffe119',
#         '#4363d8',
#         '#f58231',
#         '#911eb4',
#         '#46f0f0',
#         '#f032e6',
#         '#bcf60c',
#         '#fabebe',
#         '#008080',
#         '#e6beff',
#         '#9a6324',
#         '#fffac8',
#         '#800000',
#         '#aafdc9',
#         '#808000',
#         '#ffd8b1',
#         '#000075',
#         '#9cb44b',
#         '#808080'
#
#     ]
#
#     for el in LIST_OF_CUSTOMERS:  #fetch on custumers list
#         # layer building
#         for i in range(LEN_OF_FEATURES):
#             # add nodes
#             list_of_nodes.append(('C' + str(i) + '-U-' + str(el), {'color': 'g'}))
#             for attr in features[i].tolist():  # fetch on home ownership encode values
#                 #code = f"#{format(255-10*i, '02x')}{format(150+9*i, '02x')}{format(55+10*i, '02x')}"
#                 if int(data.loc[el, attr]) == 1:  # check if exists relation between both
#                     # bidirectional relation between home ownership and user
#                     list_of_edges.append(('C' + str(i) + '-U-' + str(el), 'C' + str(i) + '-M-' + attr,
#                                           {'color': 'b'}))  # add edge to list
#                     list_of_edges.append(('C' + str(i) + '-M-' + attr, 'C' + str(i) + '-U-' + str(el),
#                                           {'color': 'b'}))  # add edge to list
#                     # add nodes
#                     list_of_nodes.append(('C' + str(i) + '-M-' + attr, {'color': colors[i]}))
#             # add directed relation between user node from C1 and C2
#             list_of_edges.append(('C' + str(i) + '-U-' + str(el), 'C' + str(i + 1) + '-U-' + str(el),
#                                   {'color': 'r'}))  # add edge to list
#             list_of_edges.append(('C' + str(i + 1) + '-U-' + str(el), 'C' + str(i) + '-U-' + str(el),
#                                   {'color': 'r'}))  # add edge to list
#         #code = f"#{format(255-10*LEN_OF_FEATURES, '02x')}{format(150+9*LEN_OF_FEATURES, '02x')}{format(55+10*LEN_OF_FEATURES, '02x')}"
#         # add class nodes
#         list_of_nodes.append(('C' + str(LEN_OF_FEATURES) + '-U-' + str(el), {'color': 'g'}))
#         # list_of_nodes.append(
#         #     ('C' + str(LEN_OF_FEATURES) + '-M-C-' + str(data.loc[el, className]), {'color': colors[LEN_OF_FEATURES]}))
#         # bidirectional relation between home ownership and user
#         # list_of_edges.append(('C' + str(LEN_OF_FEATURES) + '-U-' + str(el),
#         #                       'C' + str(LEN_OF_FEATURES) + '-M-C-' + str(data.loc[el, className]),
#         #                       {'color': 'b'}))  # add edge to list
#         # list_of_edges.append(('C' + str(LEN_OF_FEATURES) + '-M-C-' + str(data.loc[el, className]),
#         #                       'C' + str(LEN_OF_FEATURES) + '-U-' + str(el), {'color': 'b'}))  # add edge to list
#         # add directed relation between user node from C1 and C2
#         list_of_edges.append(('C' + str(LEN_OF_FEATURES) + '-U-' + str(el),
#                               'C' + str(LEN_OF_FEATURES - 1) + '-U-' + str(el), {'color': 'r'}))  # add edge to list
#         list_of_edges.append(('C' + str(LEN_OF_FEATURES - 1) + '-U-' + str(el),
#                               'C' + str(LEN_OF_FEATURES) + '-U-' + str(el), {'color': 'r'}))  # add edge to list
#     # add edges to the oriented graph
#     # print(list_of_nodes)
#     CRP_G.add_nodes_from(list_of_nodes)
#     CRP_G.add_edges_from(list_of_edges)
#
#     # return the graph
#     return CRP_G

# def pagerank_with_openCL(graph, personalization=None, alpha=0.85):
#     # Create a pandas adjacency matrix from the NetworkX graph
#     adjacency_matrix = nx.to_pandas_adjacency(graph)
#
#     # OpenCL setup
#     platforms = cl.get_platforms()
#     devices = platforms[0].get_devices()
#     context = cl.Context(devices=[devices[0]])
#     queue = cl.CommandQueue(context)
#
#     # Load OpenCL kernel
#     kernel_code = open("pagerank_kernel.cl").read()
#     program = cl.Program(context, kernel_code).build()
#
#     # Create OpenCL buffers
#     adjacency_matrix_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=adjacency_matrix.values)
#     personalization_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=personalization)
#     alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(alpha))
#     pagerank_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, adjacency_matrix.values.nbytes)
#
#     # Set kernel arguments
#     kernel = program.pagerank_kernel
#     kernel.set_arg(0, adjacency_matrix_buf)
#     kernel.set_arg(1, personalization_buf)
#     kernel.set_arg(2, alpha_buf)
#     kernel.set_arg(3, pagerank_buf)
#
#     # Execute the kernel
#     global_size = (adjacency_matrix.values.size,)
#     local_size = None
#     cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
#
#     # Read PageRank results
#     pagerank = np.empty_like(adjacency_matrix.values)
#     cl.enqueue_copy(queue, pagerank, pagerank_buf)
#
#     # Release resources
#     queue.finish()
#     pagerank_buf.release()
#     adjacency_matrix_buf.release()
#     personalization_buf.release()
#     alpha_buf.release()
#     program.release()
#     context.release()
#
#     # Convert PageRank results to dictionary
#     result = dict(zip(graph.nodes(), pagerank))
#
#     return result


# def pagerank(graph, personalization=None, alpha=0.85):
#     # Créez un client Dask pour gérer le cluster
#     client = Client()
#
#     # Créez une matrice d'adjacence pandas à partir du graphe NetworkX
#     adjacency_matrix = nx.to_pandas_adjacency(graph)
#
#     def calculate_pagerank(subgraph):
#         # Utilisez Dask.delayed pour paralléliser les calculs de PageRank
#         page_ranks = [
#             dask.delayed(nx.pagerank)(
#                 nx.from_pandas_adjacency(subgraph[i]),
#                 personalization=personalization,
#                 alpha=alpha,
#             )
#             for i in range(len(subgraph))
#         ]
#
#         # Calculez les PageRanks en parallèle
#         page_ranks = dask.compute(page_ranks)
#
#         # Créez un DataFrame Dask avec les noms de nœuds et les PageRanks
#         df = dd.from_pandas(
#             pd.DataFrame(
#                 {
#                     "node_name": list(subgraph.index),
#                     "pagerank": [pr for pr in page_ranks],
#                 }
#             ),
#             npartitions=1,
#         )
#         # print(df)
#
#         return df
#
#     # Convertissez la matrice d'adjacence pandas en un DataFrame Dask
#     print(client.ncores)
#     dask_dataframe = dd.from_pandas(adjacency_matrix, npartitions=1
#     # sum([core for core in client.ncores().values()])
#     )
#
#     # Calculez PageRank en parallèle avec personnalisation et alpha
#     ranks = dask_dataframe.map_partitions(calculate_pagerank, personalization, alpha
#                                           , meta=pd.DataFrame(columns=['node_name', 'pagerank']
#                                                               ,index=dask_dataframe.index
#                                                               )
#                                           )
#
#     # Convertissez les résultats en un DataFrame Dask
#     print(ranks.compute())
#     # df = ranks.set_index('node_name').compute()
#
#
#     # Convertissez le DataFrame Dask en un dictionnaire
#     result = ranks.to_dict()['pagerank']
#
#     # Fermez le client Dask
#     client.close()
#
#     return result
#

# def pagerank(graph, personalization=None, alpha=0.85):
#     # Create a pandas adjacency matrix from the NetworkX graph
#     adjacency_matrix = nx.to_pandas_adjacency(graph)
#
#     # Convert the pandas adjacency matrix to a Dask DataFrame
#     dask_dataframe = dd.from_pandas(adjacency_matrix, npartitions=1)
#
#     # Calculate PageRank in parallel with customization and alpha
#     def calculate_pagerank(subgraph):
#         # Create a DataFrame with one row for each node
#         graph = nx.from_pandas_adjacency(subgraph)
#         # df = pd.DataFrame(index=graph.nodes())
#         # Calculate PageRank for each node using nx.pagerank
#         pg = nx.pagerank(graph, personalization=personalization, alpha=alpha)
#         df = pd.DataFrame(data={
#             "node_name": list(pg.keys()),
#             "pagerank": list(pg.values())
#         })
#         # print(df)
#         return df
#
#     # Convert the results to a Dask DataFrame
#     ranks = dask_dataframe.map_partitions(calculate_pagerank, meta=pd.DataFrame(columns=['node_name', 'pagerank']))
#
#     # Get the node names and PageRank values
#     node_names = ranks['node_name'].compute().values
#     pagerank_values = ranks['pagerank'].compute().values
#     # print(node_names,pagerank_values)
#
#     # Convert the node names and PageRank values to NumPy arrays
#     node_indices = da.from_array(node_names, chunks=len(node_names))
#     pagerank_indices = da.from_array(pagerank_values, chunks=len(pagerank_values))
#
#     # Create a Dask DataFrame with node names and PageRank values
#     df = dd.from_dask_array(node_indices, columns=['node_name'])
#     df['pagerank'] = pagerank_indices
#
#     # Convert the Dask DataFrame to a dictionary
#     result = df.set_index('node_name').compute().to_dict()['pagerank']
#
#     return result


def compute_distance_descriptor(graph, k, label, loan_id):
    # Format loan node and decision label node
    loan_node = f"C-0-U-{loan_id}"
    decision_label_node = f"C-{k}-M-C-{label}"
    # Remove edge between start_node and end_node
    modified_graph = graph.copy()
    modified_graph.remove_edge(loan_node, decision_label_node)

    # Compute paths between start_node and end_node in the modified graph
    paths = nx.all_simple_paths(modified_graph, loan_node, decision_label_node)

    # Analyze the paths to calculate distance descriptor
    num_paths = 0
    total_length = 0

    for path in paths:
        num_paths += 1
        total_length += len(path) - 1

    average_length = total_length / num_paths

    return num_paths, average_length


def removeEdge(graph, k, label, loan_id):
    """

    Parameters
    ----------
    graph
    k
    label
    loan_id

    Returns
    -------
    modified_graph
    """
    # Format loan node and decision label node
    # print(f"""
    # {k},
    # {label}
    # {loan_id}
    # """)
    loan_node = f"C{k}-U-{loan_id}"
    decision_label_node = f"C{k}-M-C-{label}"
    # Remove edge between start_node and end_node
    modified_graph = copy.deepcopy(graph)
    if modified_graph.has_edge(loan_node, decision_label_node) is True:
        modified_graph.remove_edge(loan_node, decision_label_node)
        modified_graph.remove_node(loan_node)
    # print(f"{loan_node} - {decision_label_node} still exist {modified_graph.has_edge(loan_node, decision_label_node)}")

    return modified_graph
