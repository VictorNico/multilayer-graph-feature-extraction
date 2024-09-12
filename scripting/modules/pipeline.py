"""
	Author: VICTOR DJIEMBOU
	addedAt: 28/11/2023
	changes:
		- 28/11/2023:
			- add pipeline methods def
		- 29/11/2023
			- add utils modules importation
			- add preprocessing instructions
		- 30/11/2023:
			- start MLNA on categorial variables
		- 01/12/2023:
			- run categorial MLNA 1 VAR with success but somes issues when plotting barh diagram
			- start fixing and in parallel build MLNA k VAR
			- applied the same logique on numerical variables
			- modularisation of code in some pipeline stage
		- 02/12/2023:
			- test modularisation of pipeline stages
		- 14/12/2023:
			- apply all computation on discretize OHE dataset
		- 07/01/2024:
			- ajout d'un processus d'eda après toute étapes de prétraitement
			- mise à jour du bloc lig503-509, 509:mlna.add(nominal_factor_colums[i])->mlna.add(column.split("__")[-1])
			- separer distinctement les resultats des variables catégorielles, numeriques, combinée dans la méthode mlnaPipeline
			
"""
import copy

#################################################
##          Libraries importation
#################################################

###### Begin
from tqdm import tqdm
import statistics
import importlib

# List des modules to load
modules = [
    'modules.file',
    'modules.eda',
    'modules.graph',
    'modules.modeling',
    'modules.report',
    'modules.preprocessing',
    # 'memory_profiler'
]

# Charger les bibliothèques en utilisant importlib
with tqdm(total=len(modules), desc="MODULES LOADING", ncols=80) as pbar:
    for module_name in modules:
        try:
            # module = importlib.import_module(library_name)
            exec(f'from {module_name} import *')

        except ImportError:
            # En cas d'erreur lors du chargement de la bibliothèque
            pbar.set_description(f"Unable to load {module_name}")
        else:
            # Succès lors du chargement de la bibliothèque
            pbar.set_description(f"{module_name} successfull loaded")
        finally:
            pbar.update(1)


###### End


#################################################
##          Methods definition
#################################################

# @profile
def make_eda(dataframe, verbose):
    df = dataframe.copy(deep=True)
    if isinstance(dataframe, pd.DataFrame):
        isNAColumns = get_na_columns(dataframe)
        print(f"{isNAColumns}")
        if len(isNAColumns) > 0:
            df = impute_nan_values(dataframe, isNAColumns)
            print(f"{get_na_columns(df)}")

    return df

# @profile
def make_preprocessing(dataset, to_remove, domain, cwd, target_variable, verbose, levels):
    """Apply some framework preprocessing like, OHE, LableEncoding, normalisation, discretization

    Args:
        dataset:
        to_remove:
        domain:
        cwd:
        target_variable:
        verbose
        levels

    Returns:
        A tuple of parameters which are need to nexted framework execution
    """

    dataset.drop(to_remove, axis=1, inplace=True)
    ## get list of numeric, ordinal and nominal dimension in dataset

    col_list = dataset.columns.tolist()

    numeric_col = get_numerical_columns(dataset)

    categorial_col = get_categorial_columns(dataset)

    ordinal_factor_colums = get_ordinal_columns(dataset)

    nominal_factor_colums = list(set(categorial_col) - set(ordinal_factor_colums))

    numeric_with_outliers_columns = get_numeric_vector_with_outlier(dataset)

    numeric_uniform_colums = list(set(numeric_col) - set(numeric_with_outliers_columns))

    print(f"""
	col_list:{col_list}
	numeric_col:{numeric_col}
	categorial_col:{categorial_col}
	ordinal_factor_colums:{ordinal_factor_colums}
	nominal_factor_colums:{nominal_factor_colums}
	numeric_with_outliers_columns:{numeric_with_outliers_columns}
	numeric_uniform_colums:{numeric_uniform_colums}
		""") if verbose else None

    ## apply preprocessing treatments
    _ALL_PRETRAETED_DATA = {}
    dataset_original = dataset.copy(deep=True)  # savea backup of our dataset

    # binarise nominals factors
    DATA_OHE, OHE = nominal_factor_encoding(
        dataset,
        categorial_col if not (target_variable in categorial_col) else list(
            set(categorial_col) - set([target_variable]))
    ) if len(categorial_col) > 0 else (dataset, [])
    DATA_OHE = make_eda(dataframe=DATA_OHE, verbose=verbose)
    print(f"OHE <----> {OHE}")
    # # label encoding of ordinal data
    # DATA_OHE_LB = ordinal_factor_encoding(
    #     DATA_OHE,
    #     [target_variable]
    # ) if len([target_variable]) > 0 else DATA_OHE
    DATA_OHE_LB = DATA_OHE
    # standard normalisation of label encoded data to deeve it into interval 0,1
    # DATA_OHE_LB_LBU = numeric_uniform_standardization(
    # 	DATA_OHE,
    # 	ordinal_factor_colums if not(target_variable in ordinal_factor_colums) else list(set(ordinal_factor_colums)-set([target_variable]))
    # 	) if len(ordinal_factor_colums) > 0 else DATA_OHE_LB
    # print(f"{get_na_columns(DATA_OHE_LB_LBU)}")
    # DATA_OHE_LB_LBU = make_eda(dataframe=DATA_OHE_LB_LBU, verbose=verbose)
    # standard normalisation of numeric data to deeve it into interval 0,1
    DATA_OHE_LB_LBU_STDU = numeric_uniform_standardization(
        DATA_OHE_LB,
        numeric_uniform_colums if not (target_variable in numeric_uniform_colums) else list(
            set(numeric_uniform_colums) - set([target_variable]))
    ) if len(numeric_uniform_colums) > 0 else DATA_OHE_LB
    print(f"{get_na_columns(DATA_OHE_LB_LBU_STDU)}")
    DATA_OHE_LB_LBU_STDU = make_eda(dataframe=DATA_OHE_LB_LBU_STDU, verbose=verbose)
    # normalisation of numeric data with outliers to deeve it into interval 0,1
    DATA_OHE_LB_LBU_STDU_STDWO = numeric_standardization_with_outliers(
        DATA_OHE_LB_LBU_STDU,
        numeric_with_outliers_columns if not (target_variable in numeric_with_outliers_columns) else list(
            set(numeric_with_outliers_columns) - set([target_variable]))
    ) if len(numeric_with_outliers_columns) > 0 else DATA_OHE_LB_LBU_STDU
    print(f"{get_na_columns(DATA_OHE_LB_LBU_STDU_STDWO)}")
    DATA_OHE_LB_LBU_STDU_STDWO = make_eda(dataframe=DATA_OHE_LB_LBU_STDU_STDWO, verbose=verbose)
    # save the first one where just all factor are binarized
    # _ALL_PRETRAETED_DATA['withAllFactorsVBinarized']= DATA_OHE_LB_LBU_STDU_STDWO.copy(deep=True)
    print(f"{get_na_columns(DATA_OHE_LB_LBU_STDU_STDWO)}")
    # save preprocessed 1st one data
    link_to_preprocessed_factor_data = save_dataset(
        cwd=cwd + '/mlna_preprocessing',
        dataframe=DATA_OHE_LB_LBU_STDU_STDWO,
        name=f'{domain}_preprocessed',
        # prefix= domain,
        sep=','
    )

    if (4 in levels) or (5 in levels):
        # discretize nuneric value
        DATA_DISCRETIZE = discretise_numeric_dimension_by_entropy(
            bins=3,
            threshold=.00000001,
            target=target_variable,
            columns=list(set([*numeric_with_outliers_columns, *numeric_uniform_colums]) - set([target_variable])),
            dataframe=DATA_OHE_LB_LBU_STDU_STDWO,
            inplace=False,
            verbose=verbose
        ) if len(list({*numeric_with_outliers_columns, *numeric_uniform_colums} - {target_variable})) > 0 else DATA_OHE_LB_LBU_STDU_STDWO
        link_to_preprocessed_disc_data = save_dataset(
            cwd=cwd + '/mlna_preprocessing',
            dataframe=DATA_DISCRETIZE,
            name=f'{domain}_preprocessed_discretize',
            # prefix= domain,
            sep=','
        )
        print(f"{get_na_columns(DATA_DISCRETIZE)}")
        DATA_DISCRETIZE = make_eda(dataframe=DATA_DISCRETIZE, verbose=verbose)
        # binarise nominals factors
        DATA_DISCRETIZE_OHE_2, OHE_2 = nominal_factor_encoding(
            DATA_DISCRETIZE,
            list(set([*numeric_with_outliers_columns, *numeric_uniform_colums]) - set([target_variable]))
        ) if len(
            list(set([*numeric_with_outliers_columns, *numeric_uniform_colums]) - set([target_variable]))) > 0 else (
            DATA_DISCRETIZE, [])
        print(f"{get_na_columns(DATA_DISCRETIZE_OHE_2)}")
        DATA_DISCRETIZE_OHE_2 = make_eda(dataframe=DATA_DISCRETIZE_OHE_2, verbose=verbose)

        link_to_preprocessed_all_data = save_dataset(
            cwd=cwd + '/mlna_preprocessing',
            dataframe=DATA_DISCRETIZE_OHE_2,
            name=f'{domain}_preprocessed_all',
            # prefix= domain,
            sep=','
        )
        print(f"discretize data shape: {DATA_DISCRETIZE_OHE_2.shape}") if verbose else None
        print(f"{get_na_columns(DATA_DISCRETIZE_OHE_2)}")
        # save the second one where all variables are binarized
        # _ALL_PRETRAETED_DATA['withAllVariablesBinarized']= DATA_DISCRETIZE_OHE_2.copy(deep=True)
        return [
            col_list,
            numeric_col,
            categorial_col if not (target_variable in categorial_col) else list(
                set(categorial_col) - set([target_variable])),
            ordinal_factor_colums,
            nominal_factor_colums,
            numeric_with_outliers_columns,
            numeric_uniform_colums,
            DATA_OHE_LB_LBU_STDU_STDWO,
            OHE,
            DATA_DISCRETIZE_OHE_2,
            OHE_2
        ]

    return [
        col_list,
        numeric_col,
        categorial_col if not (target_variable in categorial_col) else list(
            set(categorial_col) - set([target_variable])),
        ordinal_factor_colums,
        nominal_factor_colums,
        numeric_with_outliers_columns,
        numeric_uniform_colums,
        DATA_OHE_LB_LBU_STDU_STDWO,
        OHE
    ]

def extract_descriptors_from_graph_model(
    graph=None,
    y_graph=None,
    graphWithClass=None,
    alpha=.85,
    borrower=None,
    layers=1
):
    descriptors = {}

    ################################
    ####### Global Descriptor ######
    ################################
    descriptors['Att_DEGREE_GLO'] = (
        get_number_of_borrowers_with_same_n_layer_value(
                borrower=borrower,
                graph=graph,
                layer_nber=0
           )[1]
        if layers == 1 else
        get_number_of_borrowers_with_same_custom_layer_value(
            borrower=borrower,
            graph=graph,
            custom_layer=list(range(layers))
        )[1]
    )
    bipart_combine = nx.pagerank(graph, alpha=alpha)

    # get intra page rank
    bipart_intra_pagerank = nx.pagerank(
        graph,
        personalization=compute_personlization(get_intra_node_label(graph), graph),
        alpha=alpha
    )

    # get inter page rank
    bipart_inter_pagerank = nx.pagerank(
        graph,
        personalization=compute_personlization(get_inter_node_label(graph), graph),
        alpha=alpha
    )

    descriptors['Att_INTRA_GLO'] = get_max_modality_pagerank_score(borrower, graph, layers, bipart_intra_pagerank)
    descriptors['Att_INTER_GLO'] = get_max_modality_pagerank_score(borrower, graph, layers, bipart_inter_pagerank)
    descriptors['Att_COMBINE_GLO'] = get_max_modality_pagerank_score(borrower, graph, layers, bipart_combine)
    descriptors['Att_M_INTRA_GLO'] = get_max_borrower_pr(bipart_intra_pagerank)[1][borrower]
    descriptors['Att_M_INTER_GLO'] = get_max_borrower_pr(bipart_inter_pagerank)[1][borrower]
    descriptors['Att_M_COMBINE_GLO'] = get_max_borrower_pr(bipart_combine)[1][borrower]

    ######################################
    ####### Personalized Descriptor ######
    ######################################
    descriptors[f'Att_DEGREE_PER'] = descriptors[f'Att_DEGREE_GLO']
    descriptors[f'Att_COMBINE_PER'] = get_max_borrower_pr(
        nx.pagerank(
            graph,
            personalization=compute_personlization(
                get_combine_perso_nodes_label(
                    graph,
                    [borrower],
                    1
                )[0][0],
                graph
            ),
            alpha=alpha
        )
    )[1][borrower]

    if graphWithClass is True:
        descriptors[f'YN_COMBINE_PER'] = get_class_pr(
            nx.pagerank(
                y_graph,
                personalization=compute_personlization(
                    get_combine_perso_nodes_label(
                        y_graph,
                        [borrower],
                        1
                    )[0][0],
                    y_graph
                ),
                alpha=alpha
            )
        )[0]

        descriptors[f'YP_COMBINE_PER'] = get_class_pr(
            nx.pagerank(
                y_graph,
                personalization=compute_personlization(
                    get_combine_perso_nodes_label(
                        y_graph,
                        [borrower],
                        1
                    )[0][0],
                    y_graph
                ),
                alpha=alpha
                )
        )[1]

    descriptors[f'Att_INTER_PER'] = get_max_borrower_pr(
        nx.pagerank(
            graph,
            personalization=compute_personlization(
                get_inter_perso_nodes_label(
                    graph,
                    [borrower],
                    1
                )[0][0],
                graph
            ),
            alpha=alpha
        )
        )[1][borrower]

    if graphWithClass is True:
        descriptors[f'YN_INTER_PER'] = get_class_pr(
            nx.pagerank(
                y_graph,
                personalization=compute_personlization(
                    get_inter_perso_nodes_label(
                        y_graph,
                        [borrower],
                        1
                    )[0][0],
                    y_graph
                ),
                alpha=alpha
            )
        )[0]

        descriptors[f'YP_INTER_PER'] = get_class_pr(
            nx.pagerank(
                y_graph,
                personalization=compute_personlization(
                    get_inter_perso_nodes_label(
                        y_graph,
                        [borrower],
                        1
                    )[0][0],
                    y_graph
                ),
                alpha=alpha
            )
        )[1]

    descriptors[f'Att_INTRA_PER'] = get_max_borrower_pr(
        nx.pagerank(
            graph,
            personalization=compute_personlization(
                get_intra_perso_nodes_label(
                    graph,
                    [borrower],
                    1
                )[0][0],
                graph
            ),
            alpha=alpha
        )
    )[1][borrower]

    if graphWithClass is True:
        descriptors[f'YN_INTRA_PER'] = get_class_pr(
            nx.pagerank(
                y_graph,
                personalization=compute_personlization(
                    get_intra_perso_nodes_label(
                        y_graph,
                        [borrower],
                        1
                    )[0][0],
                    y_graph
                ),
                alpha=alpha
            )
        )[0]

        descriptors[f'YP_INTRA_PER'] = get_class_pr(
            nx.pagerank(
                y_graph,
                personalization=compute_personlization(
                    get_intra_perso_nodes_label(
                        y_graph,
                        [borrower],
                        1
                    )[0][0],
                    y_graph
                ),
                alpha=alpha
            )
        )[1]

    descriptors[f'Att_M_COMBINE_PER'] = get_max_modality_pagerank_score(
        borrower,
        graph,
        1,
        nx.pagerank(
            graph,
            personalization=compute_personlization(
                get_combine_perso_nodes_label(
                    graph,
                    [borrower],
                    1
                )[0][0],
                graph
            ),
            alpha=alpha
        )
    )

    descriptors[f'Att_M_INTER_PER'] = get_max_modality_pagerank_score(
        borrower,
        graph,
        1,
        nx.pagerank(
            graph,
            personalization=compute_personlization(
                get_inter_perso_nodes_label(
                    graph,
                    [borrower],
                    1
                )[0][0],
                graph
            ),
            alpha=alpha
        )
    )

    descriptors[f'Att_M_INTRA_PER'] = get_max_modality_pagerank_score(
        borrower,
        graph,
        1,
        nx.pagerank(
            graph,
            personalization=compute_personlization(
                get_intra_perso_nodes_label(
                    graph,
                    [borrower],
                    1
                )[0][0],
                graph
            ),
            alpha=alpha
        )
    )

    return descriptors

def get_test_examples_descriptors_from_graph_model(
        MLN=None,
        x_test=None,
        graphWithClass=False,
        OHE=None,
        i=None,
        alpha=None,
        layers=1,
        layer_config=None
):
    extracts_g_t = {}
    extracts_p_t = {}

    PERSONS_T = get_persons(x_test)

    """
    computed descriptors for test set
    """

    # Binome page rank
    extracts_g_t[f'Att_DEGREE_GLO'] = (
        [
        get_number_of_borrowers_with_same_n_layer_value(
            borrower=index,
            graph=(
                add_specific_loan_in_mlg(MLN, x_test.loc[[index]], [OHE[i]])
                if (graphWithClass is True) else
                add_specific_loan_in_mlg(MLN, x_test.loc[[index]], [OHE[i]])
            ),
            layer_nber=0
        )[1] for index in PERSONS_T
        ] if layers == 1 else
        [
        get_number_of_borrowers_with_same_custom_layer_value(
            borrower=index,
            graph=(
                add_specific_loan_in_mlg(MLN, x_test.loc[[index]], [OHE[j] for j in layer_config])
                if (graphWithClass is True) else
                add_specific_loan_in_mlg(MLN, x_test.loc[[index]], [OHE[j] for j in layer_config])
            ),
            custom_layer=list(range(i))
        )[1] for index in PERSONS_T
    ])
    extracts_g_t[f'Att_INTRA_GLO'] = list(range(len(PERSONS_T)))
    extracts_g_t[f'Att_INTER_GLO'] = list(range(len(PERSONS_T)))
    extracts_g_t[f'Att_COMBINE_GLO'] = list(range(len(PERSONS_T)))
    extracts_g_t[f'Att_M_INTRA_GLO'] = list(range(len(PERSONS_T)))
    extracts_g_t[f'Att_M_INTER_GLO'] = list(range(len(PERSONS_T)))
    extracts_g_t[f'Att_M_COMBINE_GLO'] = list(range(len(PERSONS_T)))
    for l, index in enumerate(PERSONS_T):
        # print(f"""
        # index = {index}
        # row = {x_test.loc[[index]]}
        # type = {type(x_test.loc[[index]])}
        # ohe = {OHE[i]}
        # """)
        grf = (
            add_specific_loan_in_mlg(MLN, x_test.loc[[index]], [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
            if (graphWithClass is True) else
            add_specific_loan_in_mlg(MLN, x_test.loc[[index]], [OHE[i]]  if layers == 1 else [OHE[j] for j in layer_config] )
        )
        bipart_combine = nx.pagerank(grf, alpha=alpha)

        # get intra page rank
        bipart_intra_pagerank = nx.pagerank(
            grf,
            personalization=compute_personlization(get_intra_node_label(grf), grf),
            alpha=alpha
        )

        # get inter page rank
        bipart_inter_pagerank = nx.pagerank(
            grf,
            personalization=compute_personlization(get_inter_node_label(grf), grf),
            alpha=alpha
        )

        extracts_g_t[f'Att_M_INTRA_GLO'][l] = get_max_modality_pagerank_score(index, grf, layers, bipart_intra_pagerank)
        extracts_g_t[f'Att_M_INTER_GLO'][l] = get_max_modality_pagerank_score(index, grf, layers, bipart_inter_pagerank)
        extracts_g_t[f'Att_M_COMBINE_GLO'][l] = get_max_modality_pagerank_score(index, grf, layers, bipart_combine)

        extracts_g_t[f'Att_INTRA_GLO'][l] = get_max_borrower_pr(bipart_intra_pagerank)[1][index]
        extracts_g_t[f'Att_INTER_GLO'][l] = get_max_borrower_pr(bipart_inter_pagerank)[1][index]
        extracts_g_t[f'Att_COMBINE_GLO'][l] = get_max_borrower_pr(bipart_combine)[1][index]
        grf = None

    extracts_p_t[f'Att_DEGREE_PER'] = extracts_g_t[f'Att_DEGREE_GLO']
    extracts_p_t[f'Att_COMBINE_PER'] = [
        get_max_borrower_pr(
            nx.pagerank(
                (
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    if graphWithClass is True else
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                ),
                personalization=compute_personlization(
                    get_combine_perso_nodes_label(
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        ),
                        [val],
                        1
                    )[0][0],
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    )
                ),
                alpha=alpha
            )
        )[1][val] for index, val in enumerate(PERSONS_T)]
    if graphWithClass is True:
        extracts_p_t[f'YN_COMBINE_PER'] = [
            get_class_pr(
                nx.pagerank(
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    ),
                    personalization=compute_personlization(
                        get_combine_perso_nodes_label(
                            (
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                                if graphWithClass is True else
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            ),
                            [val],
                            1
                        )[0][0],
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        )
                    ),
                    alpha=alpha
                )
            )[0] for index, val in enumerate(PERSONS_T)]
        extracts_p_t[f'YP_COMBINE_PER'] = [
            get_class_pr(
                nx.pagerank(
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    ),
                    personalization=compute_personlization(
                        get_combine_perso_nodes_label(
                            (
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                                if graphWithClass is True else
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            ),
                            [val],
                            1
                        )[0][0],
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        )
                    ),
                    alpha=alpha
                )
            )[1] for index, val in enumerate(PERSONS_T)]
    extracts_p_t[f'Att_INTER_PER'] = [
        get_max_borrower_pr(
            nx.pagerank(
                (
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    if graphWithClass is True else
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                ),
                personalization=compute_personlization(
                    get_inter_perso_nodes_label(
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        ),
                        [val],
                        1
                    )[0][0],
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    )
                ),
                alpha=alpha
            )
        )[1][val] for index, val in enumerate(PERSONS_T)]
    if graphWithClass is True:
        extracts_p_t[f'YN_INTER_PER'] = [
            get_class_pr(
                nx.pagerank(
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    ),
                    personalization=compute_personlization(
                        get_inter_perso_nodes_label(
                            (
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                                if graphWithClass is True else
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            ),
                            [val],
                            1
                        )[0][0],
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        )
                    ),
                    alpha=alpha
                )
            )[0] for index, val in enumerate(PERSONS_T)]
        extracts_p_t[f'YP_INTER_PER'] = [
            get_class_pr(
                nx.pagerank(
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    ),
                    personalization=compute_personlization(
                        get_inter_perso_nodes_label(
                            (
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                                if graphWithClass is True else
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            ),
                            [
                                val],
                            1)[
                            0][
                            0],
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        )),
                    alpha=alpha))[
                1] for index, val in enumerate(PERSONS_T)]
    extracts_p_t[f'Att_INTRA_PER'] = [
        get_max_borrower_pr(
            nx.pagerank(
                (
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    if graphWithClass is True else
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                ),
                personalization=compute_personlization(
                    get_intra_perso_nodes_label(
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        ), [val], 1
                    )[0][0],
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    )
                ),
                alpha=alpha
            )
        )[1][val] for index, val in enumerate(PERSONS_T)]
    if graphWithClass is True:
        extracts_p_t[f'YN_INTRA_PER'] = [
            get_class_pr(
                nx.pagerank(
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    ),
                    personalization=compute_personlization(
                        get_intra_perso_nodes_label(
                            (
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                                if graphWithClass is True else
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            ),
                            [val],
                            1
                        )[0][0],
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        )
                    ),
                    alpha=alpha
                )
            )[0] for index, val in enumerate(PERSONS_T)]
        extracts_p_t[f'YP_INTRA_PER'] = [
            get_class_pr(
                nx.pagerank(
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    ),
                    personalization=compute_personlization(
                        get_intra_perso_nodes_label(
                            (
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                                if graphWithClass is True else
                                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            ),
                            [val],
                            1
                        )[0][0],
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        )
                    ),
                    alpha=alpha
                )
            )[1] for index, val in enumerate(PERSONS_T)]
    extracts_p_t[f'Att_M_COMBINE_PER'] = [
        get_max_modality_pagerank_score(
            val,
            (
                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                if graphWithClass is True else
                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
            ),
            1,
            nx.pagerank(
                (
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    if graphWithClass is True else
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                ),
                personalization=compute_personlization(
                    get_combine_perso_nodes_label(
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        ),
                        [val],
                        1
                    )[0][0],
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    )
                ),
                alpha=alpha
            )
        ) for index, val in enumerate(PERSONS_T)]
    extracts_p_t[f'Att_M_INTER_PER'] = [
        get_max_modality_pagerank_score(
            val,
            (
                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                if graphWithClass is True else
                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
            ),
            1,
            nx.pagerank(
                (
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    if graphWithClass is True else
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                ),
                personalization=compute_personlization(
                    get_inter_perso_nodes_label(
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        ),
                        [val],
                        1
                    )[0][0],
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    )
                ),
                alpha=alpha)) for index, val in enumerate(PERSONS_T)]
    extracts_p_t[f'Att_M_INTRA_PER'] = [
        get_max_modality_pagerank_score(
            val,
            (
                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                if graphWithClass is True else
                add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                         [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
            ),
            1,
            nx.pagerank(
                (
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    if graphWithClass is True else
                    add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                             [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                ),
                personalization=compute_personlization(
                    get_intra_perso_nodes_label(
                        (
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                            if graphWithClass is True else
                            add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                     [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        ),
                        [val],
                        1
                    )[0][0],
                    (
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                        if graphWithClass is True else
                        add_specific_loan_in_mlg(MLN, x_test.loc[[val]],
                                                 [OHE[i]] if layers == 1 else [OHE[j] for j in layer_config])
                    )
                ),
                alpha=alpha
            )
        ) for index, val in enumerate(PERSONS_T)
    ]
    if layers > 1:
        print(f"""
        {len(list(extracts_p_t.keys()))}
        {x_test.shape[0]}
        """)
    standard_extraction(extracts_g_t, extracts_g_t.keys())
    standard_extraction(extracts_p_t, extracts_p_t.keys())

    return extracts_g_t, extracts_p_t


def generate_config_df(
    graphWithClass=False,
    mlnL='/mlna_1',
    cwd=None,
    domain=None,
    extracts_g=None,
    extracts_p=None,
    extracts_g_t=None,
    extracts_p_t=None,
    name=None,
):
    config_df = dict()
    """
    train
    """
    config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df'] = pd.DataFrame(extracts_g)
    config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df'] = pd.DataFrame({key: extracts_p[key] for key in extracts_p.keys() if not ("Y" in key)})
    if graphWithClass is True:
        config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df'] = pd.DataFrame({key: extracts_p[key] for key in extracts_p.keys() if ("Y" in key)})
        config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df'] = pd.DataFrame(extracts_p)

    """
    test
    """
    config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df_t'] = pd.DataFrame(extracts_g_t)
    config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df_t'] = pd.DataFrame({key: extracts_p_t[key] for key in extracts_p_t.keys() if not ("Y" in key)})
    if graphWithClass is True:
        config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df_t'] = pd.DataFrame({key: extracts_p_t[key] for key in extracts_p_t.keys() if ("Y" in key)})
        config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df_t'] = pd.DataFrame(extracts_p_t)

    save_dataset(
        cwd=cwd + f'{mlnL}/global',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_GLO_df_for_{name}',
        prefix=domain,
        sep=','
    )
    save_dataset(
        cwd=cwd + f'{mlnL}/personalized',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_PER_df_for_{name}',
        prefix=domain,
        sep=','
    )
    save_dataset(
        cwd=cwd + f'{mlnL}/global',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df_t'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_GLO_df_t_for_{name}',
        prefix=domain,
        sep=','
    )
    save_dataset(
        cwd=cwd + f'{mlnL}/personalized',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df_t'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_PER_df_t_for_{name}',
        prefix=domain,
        sep=','
    )
    if graphWithClass is True:
        save_dataset(
            cwd=cwd + f'{mlnL}/personalized',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}Y_PER_df_for_{name}',
            prefix=domain,
            sep=','
        )
        save_dataset(
            cwd=cwd + f'{mlnL}/personalized',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}XY_PER_df_for_{name}',
            prefix=domain,
            sep=','
        )
        save_dataset(
            cwd=cwd + f'{mlnL}/personalized',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df_t'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}Y_PER_df_t_for_{name}',
            prefix=domain,
            sep=','
        )
        save_dataset(
            cwd=cwd + f'{mlnL}/personalized',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df_t'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}XY_PER_df_t_for_{name}',
            prefix=domain,
            sep=','
        )

    return config_df


def build_MlC(
    x_train,
    x_test,
    y_train,
    y_test,
    graphWithClass,
    config_df,
    fix_imbalance,
    target_variable,
    clfs,
    domain,
    verbose,
    cwd,
    duration_divider,
    rate_divider,
    withCost,
    financialOption,
    logic_i_g,
    logic_i_p,
    logic_i_pg,
    original,
    name,
    mlnL='/mlna_1'
):
    mlc_cf = dict()
    """
    train
    """
    mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X'] = inject_features_extracted(x_train, config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df'])

    mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}X'] = inject_features_extracted(x_train, config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df'])
    if graphWithClass is True:
        mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}Y'] = inject_features_extracted(x_train, config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df'])
        mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}XY'] = inject_features_extracted(x_train, config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df'])

    mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}X'] = inject_features_extracted(mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X'], config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df'])
    if graphWithClass is True:
        mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}Y'] = inject_features_extracted(mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X'], config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df'])
        mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}XY'] = inject_features_extracted(mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X'], config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df'])

    """
    test
    """
    mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X_T'] = inject_features_extracted(x_test, config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df_t'])

    mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}X_T'] = inject_features_extracted(x_test, config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df_t'])
    if graphWithClass is True:
        mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}Y_T'] = inject_features_extracted(x_test, config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df_t'])
        mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}XY_T'] = inject_features_extracted(x_test, config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df_t'])

    mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}X_T'] = inject_features_extracted(mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X_T'], config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df_t'])
    if graphWithClass is True:
        mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}Y_T'] = inject_features_extracted(mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X_T'], config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df_t'])
        mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}XY_T'] = inject_features_extracted(mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X_T'], config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df_t'])

    logic_i_p.append(
        make_builder(
            fix_imbalance=fix_imbalance,
            DATA_OVER=None,
            x_traini=mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}X'],
            x_testi=mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}X_T'],
            y_traini=y_train,
            y_testi=y_test,
            target_variable=target_variable,
            clfs=clfs,
            domain=f'MlC_PER_{"C" if graphWithClass is True else "M"}X_for_{name}',
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            prefix=domain,
            verbose=verbose,
            cwd=cwd + f'{mlnL}/personalized',
            withCost=withCost,
            financialOption=financialOption,
            original = original
        )
    )
    if graphWithClass is True:
        logic_i_p.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}Y'],
                x_testi=mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}Y_T'],
                y_traini=y_train,
                y_testi=y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MlC_PER_{"C" if graphWithClass is True else "M"}Y_for_{name}',
                prefix=domain,
                verbose=verbose,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
                cwd=cwd + f'{mlnL}/personalized',
                withCost=withCost,
                financialOption=financialOption,
                original = original
            )
        )
        logic_i_p.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}XY'],
                x_testi=mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}XY_T'],
                y_traini=y_train,
                y_testi=y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MlC_PER_{"C" if graphWithClass is True else "M"}XY_for_{name}',
                prefix=domain,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
                verbose=verbose,
                cwd=cwd + f'{mlnL}/personalized',
                withCost=withCost,
                financialOption=financialOption,
                original = original
            )
        )
    logic_i_g.append(
        make_builder(
            fix_imbalance=fix_imbalance,
            DATA_OVER=None,
            x_traini=mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X'],
            x_testi=mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X_T'],
            y_traini=y_train,
            y_testi=y_test,
            target_variable=target_variable,
            clfs=clfs,
            domain=f'MlC_GLO_{"C" if graphWithClass is True else "M"}X_for_{name}',
            prefix=domain,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            verbose=verbose,
            cwd=cwd + f'{mlnL}/global',
            withCost=withCost,
            financialOption=financialOption,
             original = original
        )
    )
    logic_i_pg.append(
        make_builder(
            fix_imbalance=fix_imbalance,
            DATA_OVER=None,
            x_traini=mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}X'],
            x_testi=mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}X_T'],
            y_traini=y_train,
            y_testi=y_test,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            target_variable=target_variable,
            clfs=clfs,
            domain=f'MlC_GAP_{"C" if graphWithClass is True else "M"}X_for_{name}',
            prefix=domain,
            verbose=verbose,
            cwd=cwd + f'{mlnL}/mixed',
            withCost=withCost,
            financialOption=financialOption,
            original = original
        )
    )
    if graphWithClass is True:
        logic_i_pg.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}Y'],
                x_testi=mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}Y_T'],
                y_traini=y_train,
                y_testi=y_test,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MlC_GAP_{"C" if graphWithClass is True else "M"}Y_for_{name}',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'{mlnL}/mixed',
                withCost=withCost,
                financialOption=financialOption,
                original = original
            )
        )
        logic_i_pg.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}XY'],
                x_testi=mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}XY_T'],
                y_traini=y_train,
                y_testi=y_test,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MlC_GAP_{"C" if graphWithClass is True else "M"}XY_for_{name}',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'{mlnL}/mixed',
                withCost=withCost,
                financialOption=financialOption,
                original = original
            )
        )

    return mlc_cf

def build_MCA(
    x_train,
    x_test,
    y_train,
    y_test,
    graphWithClass,
    config_df,
    fix_imbalance,
    target_variable,
    clfs,
    domain,
    verbose,
    cwd,
    withCost,
    duration_divider,
    rate_divider,
    financialOption,
    logic_i_g,
    logic_i_p,
    logic_i_pg,
    mlna,
    mlc_cf,
    original,
    name,
    mlnL='/mlna_1'
):
    """
    train
    """
    MCA_GLO_MX = mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X'].drop(list(mlna), axis=1)

    MCA_PER_MX = mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}X'].drop(list(mlna), axis=1)
    if graphWithClass is True:
        MCA_PER_MY = mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}Y'].drop(list(mlna), axis=1)
        MCA_PER_MXY = mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}XY'].drop(list(mlna), axis=1)

    MCA_GAP_MX = mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}X'].drop(list(mlna), axis=1)
    if graphWithClass is True:
        MCA_GAP_MY = mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}Y'].drop(list(mlna), axis=1)
        MCA_GAP_MXY = mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}XY'].drop(list(mlna), axis=1)

    """
    test
    """
    MCA_GLO_MX_T = mlc_cf[f'MlC_GLO_{"C" if graphWithClass is True else "M"}X_T'].drop(list(mlna), axis=1)

    MCA_PER_MX_T = mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}X_T'].drop(list(mlna), axis=1)
    if graphWithClass is True:
        MCA_PER_MY_T = mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}Y_T'].drop(list(mlna), axis=1)
        MCA_PER_MXY_T = mlc_cf[f'MlC_PER_{"C" if graphWithClass is True else "M"}XY_T'].drop(list(mlna), axis=1)

    MCA_GAP_MX_T = mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}X_T'].drop(list(mlna), axis=1)
    if graphWithClass is True:
        MCA_GAP_MY_T = mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}Y_T'].drop(list(mlna), axis=1)
        MCA_GAP_MXY_T = mlc_cf[f'MlC_GAP_{"C" if graphWithClass is True else "M"}XY_T'].drop(list(mlna), axis=1)

    logic_i_p.append(
        make_builder(
            fix_imbalance=fix_imbalance,
            DATA_OVER=None,
            x_traini=MCA_PER_MX,
            x_testi=MCA_PER_MX_T,
            y_traini=y_train,
            y_testi=y_test,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            target_variable=target_variable,
            clfs=clfs,
            domain=f'MCA_PER_{"C" if graphWithClass is True else "M"}X_for_{name}',
            prefix=domain,
            verbose=verbose,
            cwd=cwd + f'{mlnL}/personalized',
            withCost=withCost,
            financialOption=financialOption,
            original = original
        )
    )
    if graphWithClass is True:
        logic_i_p.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=MCA_PER_MY,
                x_testi=MCA_PER_MY_T,
                y_traini=y_train,
                y_testi=y_test,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MCA_PER_{"C" if graphWithClass is True else "M"}Y_for_{name}',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'{mlnL}/personalized',
                withCost=withCost,
                financialOption=financialOption,
                original = original
            )
        )
        logic_i_p.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=MCA_PER_MXY,
                x_testi=MCA_PER_MXY_T,
                y_traini=y_train,
                y_testi=y_test,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MCA_PER_{"C" if graphWithClass is True else "M"}XY_for_{name}',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'{mlnL}/personalized',
                withCost=withCost,
                financialOption=financialOption,
                original = original
            )
        )
    logic_i_g.append(
        make_builder(
            fix_imbalance=fix_imbalance,
            DATA_OVER=None,
            x_traini=MCA_GLO_MX,
            x_testi=MCA_GLO_MX_T,
            y_traini=y_train,
            y_testi=y_test,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            target_variable=target_variable,
            clfs=clfs,
            domain=f'MCA_GLO_{"C" if graphWithClass is True else "M"}X_for_{name}',
            prefix=domain,
            verbose=verbose,
            cwd=cwd + f'{mlnL}/global',
            withCost=withCost,
            financialOption=financialOption,
            original = original
        )
    )
    logic_i_pg.append(
        make_builder(
            fix_imbalance=fix_imbalance,
            DATA_OVER=None,
            x_traini=MCA_GAP_MX,
            x_testi=MCA_GAP_MX_T,
            y_traini=y_train,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            y_testi=y_test,
            target_variable=target_variable,
            clfs=clfs,
            domain=f'MCA_GAP_{"C" if graphWithClass is True else "M"}X_for_{name}',
            prefix=domain,
            verbose=verbose,
            cwd=cwd + f'{mlnL}/mixed',
            withCost=withCost,
            financialOption=financialOption,
            original = original
        )
    )
    if graphWithClass is True:
        logic_i_pg.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=MCA_GAP_MY,
                x_testi=MCA_GAP_MY_T,
                y_traini=y_train,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
                y_testi=y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MCA_GAP_{"C" if graphWithClass is True else "M"}Y_for_{name}',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'{mlnL}/mixed',
                withCost=withCost,
                financialOption=financialOption,
                original = original
            )
        )
        logic_i_pg.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=MCA_GAP_MXY,
                x_testi=MCA_GAP_MXY_T,
                y_traini=y_train,
                y_testi=y_test,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MCA_GAP_{"C" if graphWithClass is True else "M"}XY_for_{name}',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'{mlnL}/mixed',
                withCost=withCost,
                financialOption=financialOption,
                original = original
            )
        )

def make_mlna_1_variable_v2(
        default,
        OHE,
        nominal_factor_colums,
        cwd,
        domain,
        fix_imbalance,
        target_variable,
        custom_color,
        modelD,
        verbose,
        clfs,
        alpha,
        withCost,
        x_traini,
        x_testi,
        y_traini,
        y_testi,
        duration_divider,
        rate_divider,
        financialOption,
        original,
        graphWithClass=False
):
    ## visualization of result
    modelD = model_desc()
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)
    PERSONS_T = get_persons(x_test)
    # print(PERSONS)
    # print(PERSONS_T)
    # print(list(y_train.values))
    # print(PERSONS)
    # local eval storage
    logic_g = []
    logic_p = []
    logic_pg = []
    # variable to store OHE which outperform default model
    outperformers = {}
    NbGood = 0
    for i in range(len(OHE)):
        # logic storage
        logic_i_g = []
        logic_i_p = []
        logic_i_pg = []
        # build the MLN for the variable i on training dataset
        copT = x_train.copy(deep=True)
        copT[target_variable] = y_train.copy(deep=True)
        MLN = build_mlg_with_class(
            copT, [OHE[i]], target_variable) \
            if (graphWithClass is True) \
            else build_mlg(copT, [OHE[i]])
        # print(value_clfs.iloc[:,])
        # save the graph
        save_graph(
            cwd=cwd + '/mlna_1',
            graph=MLN,
            name=f'{nominal_factor_colums[i]}_mln',
            rows_len=pd.concat([x_train, y_train], axis=1).shape[0],
            prefix=domain,
            cols_len=len(OHE[i])
        )

        extracts_g = {
            "Att_DEGREE_GLO" : [],
            "Att_INTRA_GLO" : [],
            "Att_INTER_GLO" : [],
            "Att_COMBINE_GLO" : [],
            "Att_M_INTRA_GLO" : [],
            "Att_M_INTER_GLO" : [],
            "Att_M_COMBINE_GLO" : []
        }
        extracts_p = {
            "Att_DEGREE_PER" : [], # GLO
            "Att_COMBINE_PER" : [],
            "YN_COMBINE_PER" : [],
            "YP_COMBINE_PER" : [],# COM
            "Att_INTER_PER" : [],
            "YN_INTER_PER" : [],
            "YP_INTER_PER" : [],#INTER
            "Att_INTRA_PER" : [],
            "YN_INTRA_PER" : [],
            "YP_INTRA_PER" : [],#INTRA
            "Att_M_COMBINE_PER" : [],
            "Att_M_INTER_PER" : [],
            "Att_M_INTRA_PER" : []
        }
        extracts_g_t = copy.deepcopy(extracts_g)
        extracts_p_t = copy.deepcopy(extracts_p)
        ##################################
        ####### Training Descriptor ######
        ##################################
        for borrower in PERSONS:
            # compute descriptors for current borrower
            current = extract_descriptors_from_graph_model(
                graph=removeEdge(MLN, 1, copT.loc[borrower,target_variable], borrower),
                y_graph=removeEdge(MLN, 1, copT.loc[borrower,target_variable], borrower),
                graphWithClass=graphWithClass,
                alpha=alpha,
                borrower=borrower,
                layers=1
            )
            # extract descriptors context
            # print(f"{list(current.keys())} <--> {list(extracts_g.keys())} <--> {list(extracts_p.keys())}")
            for key in list(current.keys()):
                if 'GLO' in key:
                    extracts_g[key].append(current[key])
                else:
                    extracts_p[key].append(current[key])
            current = None
        ##################################
        ####### Test Descriptor     ######
        ##################################
        for borrower in PERSONS_T:
            # compute descriptors for current borrower
            grf = add_specific_loan_in_mlg(MLN, x_test.loc[[borrower]],[OHE[i]])
            current = extract_descriptors_from_graph_model(
                graph=grf,
                y_graph=grf,
                graphWithClass=graphWithClass,
                alpha=alpha,
                borrower=borrower,
                layers=1
            )
            # extract descriptors context
            for key in list(current.keys()):
                if 'GLO' in key:
                    extracts_g_t[key].append(current[key])
                else:
                    extracts_p_t[key].append(current[key])
            current = None
        ########################################
        ####### Descriptors Normalisation ######
        ########################################
        # print('inside',graphWithClass)
        if graphWithClass is False:
            # Deleting a class descriptor using del
            for key in list(extracts_g.keys()):
                if 'Y' in key:
                    del extracts_g_t[key]
                    del extracts_g[key]
            for key in list(extracts_p.keys()):
                if 'Y' in key:
                    del extracts_p[key]
                    del extracts_p_t[key]

        # get the max value of each descriptor in both train and test dataset
        maxGDesc = standard_extraction(extracts_g, extracts_g.keys())
        # print(extracts_p.keys())
        maxPDesc = standard_extraction(extracts_p, extracts_p.keys())
        print(f"{maxGDesc} <------> {maxPDesc}")
        standard_extraction(extracts_g_t, extracts_g.keys(),maxGDesc)
        standard_extraction(extracts_p_t, extracts_p.keys(),maxPDesc)

        ##########################################
        ####### Descriptors Config Generator ######
        ##########################################
        config_df = generate_config_df(
            cwd=cwd,
            graphWithClass=graphWithClass,
            mlnL='/mlna_1',
            domain=domain,
            extracts_g=extracts_g,
            extracts_p=extracts_p,
            extracts_g_t=extracts_g_t,
            extracts_p_t=extracts_p_t,
            name=nominal_factor_colums[i]
        )

        # free used ressources
        extract_g_df = None
        extract_p_df = None
        extract_g_df_t = None
        extract_p_df_t = None
        MLN = None
        bipart_combine = None
        bipart_intra_pagerank = None
        bipart_inter_pagerank = None

        ##########################################
        ####### Build MlC config            ######
        ##########################################
        mlc_cf = build_MlC(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            graphWithClass=graphWithClass,
            config_df=config_df,
            logic_i_p=logic_i_p,
            fix_imbalance=fix_imbalance,
            target_variable=target_variable,
            clfs=clfs,
            domain=domain,
            verbose=verbose,
            cwd=cwd,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            withCost=withCost,
            financialOption=financialOption,
            logic_i_g=logic_i_g,
            logic_i_pg=logic_i_pg,
            mlnL='/mlna_1',
            original = original,
            name=nominal_factor_colums[i]
        )
        extracts_g = None
        extracts_p = None
        extracts_g_t = None
        extracts_p_t = None

        ## default - MLNa
        mlna = set()
        col_list = x_train.columns.to_list()
        for column in OHE[i]:
            if column in col_list:
                mlna.add(column)
            else:
                mlna.add(nominal_factor_colums[i])

        build_MCA(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            graphWithClass=graphWithClass,
            config_df=config_df,
            fix_imbalance=fix_imbalance,
            target_variable=target_variable,
            clfs=clfs,
            domain=domain,
            verbose=verbose,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            cwd=cwd,
            withCost=withCost,
            financialOption=financialOption,
            logic_i_g=logic_i_g,
            logic_i_p=logic_i_p,
            logic_i_pg=logic_i_pg,
            mlna=mlna,
            mlc_cf=mlc_cf,
            mlnL='/mlna_1',
            original = original,
            name=nominal_factor_colums[i]
        )

        bestp = logic_i_p[5 if graphWithClass else 1][1].sort_values(
            by="accuracy",
            axis=0,
            ascending=False
            # ,key=lambda row: abs(row)
        ).head(1)
        print(f"{bestp} best personalized model")
        bestf = default[1].sort_values(
            by="accuracy",
            axis=0,
            ascending=False
            # ,key=lambda row: abs(row)
        ).head(1)

        if (bestp["accuracy"][list(bestp.index)[0]] > bestf["accuracy"][list(bestf.index)[0]]):
            NbGood += 1
        print(
            f'{list(bestp.index)[0]}--{bestp["accuracy"][list(bestp.index)[0]]}--' +
            f'{round(((round(bestp["accuracy"][list(bestp.index)[0]], 4) - round(bestf["accuracy"][list(bestf.index)[0]], 4)) / round(bestf["accuracy"][list(bestf.index)[0]], 4)) * 100, 4)} gain personalized model')
        outperformers[i] = round(((round(bestp["accuracy"][list(bestp.index)[0]], 4) - round(
            bestf["accuracy"][list(bestf.index)[0]], 4)) / round(bestf["accuracy"][list(bestf.index)[0]], 4)) * 100, 4)
        ########### END
        #########################################################################
        logic_p = [*logic_p, *logic_i_p]
        logic_g = [*logic_g, *logic_i_g]
        logic_pg = [*logic_pg, *logic_i_pg]
        table_p = print_summary([default, *logic_i_p], modelD)
        table_g = print_summary([default, *logic_i_g], modelD)
        table_pg = print_summary([default, *logic_i_pg], modelD)
        create_file(
            content=table_p[1],
            cwd=cwd + '/mlna_1/personalized',
            prefix=domain,
            filename=f"mlna_for_{nominal_factor_colums[i]}",
            extension=".html"
        )
        create_file(
            content=table_g[1],
            cwd=cwd + '/mlna_1/global',
            prefix=domain,
            filename=f"mlna_for_{nominal_factor_colums[i]}",
            extension=".html"
        )
        create_file(
            content=table_pg[1],
            cwd=cwd + '/mlna_1/mixed',
            prefix=domain,
            filename=f"mlna_for_{nominal_factor_colums[i]}",
            extension=".html"
        )
        table_g = None
        table_p = None
        logic_i_g = None
        logic_i_p = None
        mlc_cf = None
    table_p = print_summary([default, *logic_p], modelD)
    table_g = print_summary([default, *logic_g], modelD)
    table_pg = print_summary([default, *logic_pg], modelD)
    create_file(
        content=table_p[1],
        cwd=cwd + '/mlna_1/personalized',
        prefix=domain,
        filename=f"mlna_for_all_categorial data",
        extension=".html"
    )
    create_file(
        content=table_g[1],
        cwd=cwd + '/mlna_1/global',
        prefix=domain,
        filename=f"mlna_for_all_categorial data",
        extension=".html"
    )
    create_file(
        content=table_pg[1],
        cwd=cwd + '/mlna_1/mixed',
        prefix=domain,
        filename=f"mlna_for_all_categorial data",
        extension=".html"
    )
    modelD = None
    PERSONS = None
    table_p = None
    table_g = None
    # del modelD
    # del PERSONS
    return (outperformers, NbGood)

def make_mlna_k_variable_v2(
        default,
        x_traini,
        x_testi,
        y_traini,
        y_testi,
        OHE,
        nominal_factor_colums,
        cwd,
        domain,
        fix_imbalance,
        target_variable,
        custom_color,
        modelD,
        duration_divider,
        rate_divider,
        verbose,
        clfs,
        alpha,
        withCost,
        financialOption,
        original,
        graphWithClass=True
):
    # local eval storage
    logic_g = []
    logic_p = []
    logic_pg = []
    ## visualization of result
    modelD = model_desc()
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)
    PERSONS_T = get_persons(x_test)

    for k in list([2]):  # for 1<k<|OHE[i]|+2
        # for k in [2]: # for 1<k<|OHE[i]|+2
        # for k in range(2:len(OHE)+1: # for 1<k<|OHE[i]|+2
        for layer_config in get_combinations(range(len(OHE)), k):  # create subsets of k index of OHE and fetch it
            # for layer_config in [[1,2]]: # create subsets of k index of OHE and fetch it
            # logic storage
            logic_i_g = []
            logic_i_p = []
            logic_i_pg = []
            copT = x_train.copy(deep=True)
            copT[target_variable] = y_train.copy(deep=True)
            # build the MLN for the variable i
            MLN = build_mlg_with_class(
                    copT, [OHE[i] for i in layer_config],
                    target_variable
                ) \
                if (graphWithClass is True) \
                else build_mlg(copT,[OHE[i] for i in layer_config]
                               )
            col_targeted = [f'{nominal_factor_colums[i]}' for i in layer_config]
            case_k = '_'.join(col_targeted)
            # save the graph
            save_graph(
                cwd=cwd + f'/mlna_{k}',
                graph=MLN,
                name=f'{case_k}_mln',
                rows_len=pd.concat([x_train, y_train], axis=1).shape[0],
                prefix=domain,
                cols_len=len(OHE)
            )
            extracts_g = {
                "Att_DEGREE_GLO": [],
                "Att_INTRA_GLO": [],
                "Att_INTER_GLO": [],
                "Att_COMBINE_GLO": [],
                "Att_M_INTRA_GLO": [],
                "Att_M_INTER_GLO": [],
                "Att_M_COMBINE_GLO": []
            }
            extracts_p = {
                "Att_DEGREE_PER": [],  # GLO
                "Att_COMBINE_PER": [],
                "YN_COMBINE_PER": [],
                "YP_COMBINE_PER": [],  # COM
                "Att_INTER_PER": [],
                "YN_INTER_PER": [],
                "YP_INTER_PER": [],  # INTER
                "Att_INTRA_PER": [],
                "YN_INTRA_PER": [],
                "YP_INTRA_PER": [],  # INTRA
                "Att_M_COMBINE_PER": [],
                "Att_M_INTER_PER": [],
                "Att_M_INTRA_PER": []
            }
            extracts_g_t = copy.deepcopy(extracts_g)
            extracts_p_t = copy.deepcopy(extracts_p)

            ##################################
            ####### Training Descriptor ######
            ##################################
            for borrower in PERSONS:
                # compute descriptors for current borrower
                current = extract_descriptors_from_graph_model(
                    graph=MLN,
                    y_graph=removeEdge(MLN, k, copT.loc[borrower, target_variable], borrower),
                    graphWithClass=graphWithClass,
                    alpha=alpha,
                    borrower=borrower,
                    layers=k
                )
                # extract descriptors context
                # print(f"{list(current.keys())} <--> {list(extracts_g.keys())} <--> {list(extracts_p.keys())}")
                for key in list(current.keys()):
                    if 'GLO' in key:
                        extracts_g[key].append(current[key])
                    else:
                        extracts_p[key].append(current[key])

            ##################################
            ####### Test Descriptor     ######
            ##################################
            for borrower in PERSONS_T:
                # compute descriptors for current borrower
                grf = add_specific_loan_in_mlg(MLN, x_test.loc[[borrower]], [OHE[i] for i in layer_config])
                current = extract_descriptors_from_graph_model(
                    graph=grf,
                    y_graph=grf,
                    graphWithClass=graphWithClass,
                    alpha=alpha,
                    borrower=borrower,
                    layers=k
                )
                # extract descriptors context
                for key in list(current.keys()):
                    if 'GLO' in key:
                        extracts_g_t[key].append(current[key])
                    else:
                        extracts_p_t[key].append(current[key])

            ########################################
            ####### Descriptors Normalisation ######
            ########################################
            if graphWithClass is False:
                # Deleting a class descriptor using del
                # for key in list(extracts_g_t.keys()):
                #     if 'Y' in key:
                #         del extracts_g_t[key]
                #         del extracts_g[key]
                # for key in list(extracts_p_t.keys()):
                #     if 'Y' in key:
                #         del extracts_p_t[key]
                #         del extracts_p_t[key]

                # Deleting a class descriptor using del
                for key in list(extracts_g.keys()):
                    if 'Y' in key:
                        del extracts_g_t[key]
                        del extracts_g[key]
                for key in list(extracts_p.keys()):
                    if 'Y' in key:
                        del extracts_p[key]
                        del extracts_p_t[key]

            # get the max value of each descriptor in both train and test dataset
            maxGDesc = standard_extraction(extracts_g, extracts_g.keys())
            maxPDesc = standard_extraction(extracts_p, extracts_p.keys())
            # print(f"{maxGDesc} <------> {maxPDesc}")
            standard_extraction(extracts_g_t, extracts_g.keys(), maxGDesc)
            standard_extraction(extracts_p_t, extracts_p.keys(), maxPDesc)

            ##########################################
            ####### Descriptors Config Generator ######
            ##########################################
            config_df = generate_config_df(
                cwd=cwd,
                graphWithClass=graphWithClass,
                mlnL=f'/mlna_{k}',
                domain=domain,
                extracts_g=extracts_g,
                extracts_p=extracts_p,
                extracts_g_t=extracts_g_t,
                extracts_p_t=extracts_p_t,
                name=case_k
            )

            # free used ressources
            extract_g_df = None
            extract_p_df = None
            extract_g_df_t = None
            extract_p_df_t = None
            MLN = None
            bipart_combine = None
            bipart_intra_pagerank = None
            bipart_inter_pagerank = None

            ##########################################
            ####### Build MlC config            ######
            ##########################################
            mlc_cf = build_MlC(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                graphWithClass=graphWithClass,
                config_df=config_df,
                logic_i_p=logic_i_p,
                fix_imbalance=fix_imbalance,
                target_variable=target_variable,
                clfs=clfs,
                domain=domain,
                verbose=verbose,
                cwd=cwd,
                duration_divider=duration_divider,
                rate_divider=rate_divider,
                withCost=withCost,
                financialOption=financialOption,
                logic_i_g=logic_i_g,
                logic_i_pg=logic_i_pg,
                mlnL= f'/mlna_{k}',
                original = original,
                name=case_k
            )
            extracts_g = None
            extracts_p = None
            extracts_g_t = None
            extracts_p_t = None

            ## default - MLNa
            mlna = set()
            col_list = x_train.columns.to_list()
            for i in layer_config:
                for column in OHE[i]:
                    if column in col_list:
                        mlna.add(column)
                    else:
                        mlna.add(column.split("__")[-1])

            build_MCA(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                graphWithClass=graphWithClass,
                config_df=config_df,
                fix_imbalance=fix_imbalance,
                target_variable=target_variable,
                clfs=clfs,
                domain=domain,
                verbose=verbose,
                cwd=cwd,
                duration_divider=duration_divider,
                rate_divider=rate_divider,
                withCost=withCost,
                financialOption=financialOption,
                logic_i_g=logic_i_g,
                logic_i_p=logic_i_p,
                logic_i_pg=logic_i_pg,
                mlna=mlna,
                mlc_cf=mlc_cf,
                mlnL=f'/mlna_{k}',
                original= original,
                name=case_k
            )

            ########### END
            #########################################################################
            logic_p = [*logic_p, *logic_i_p]
            logic_g = [*logic_g, *logic_i_g]
            logic_pg = [*logic_pg, *logic_i_pg]
            table_p = print_summary([default, *logic_i_p], modelD)
            table_g = print_summary([default, *logic_i_g], modelD)
            table_pg = print_summary([default, *logic_i_pg], modelD)
            create_file(
                content=table_p[1],
                cwd=cwd + f'/mlna_{k}/personalized',
                prefix=domain,
                filename=f"mlna_for_{case_k}",
                extension=".html"
            )
            create_file(
                content=table_g[1],
                cwd=cwd + f'/mlna_{k}/global',
                prefix=domain,
                filename=f"mlna_for_{case_k}",
                extension=".html"
            )
            create_file(
                content=table_pg[1],
                cwd=cwd + f'/mlna_{k}/mixed',
                prefix=domain,
                filename=f"mlna_for_{case_k}",
                extension=".html"
            )
            table_g = None
            table_p = None
            logic_i_g = None
            logic_i_p = None
            mlc_cf = None
        table_p = print_summary([default, *logic_p], modelD)
        table_g = print_summary([default, *logic_g], modelD)
        table_pg = print_summary([default, *logic_pg], modelD)
        create_file(
            content=table_p[1],
            cwd=cwd + f'/mlna_{k}/personalized',
            prefix=domain,
            filename=f"mlna_for_all_categorial data",
            extension=".html"
        )
        create_file(
            content=table_g[1],
            cwd=cwd + f'/mlna_{k}/global',
            prefix=domain,
            filename=f"mlna_for_all_categorial data",
            extension=".html"
        )
        create_file(
            content=table_pg[1],
            cwd=cwd + f'/mlna_{k}/mixed',
            prefix=domain,
            filename=f"mlna_for_all_categorial data",
            extension=".html"
        )
        modelD = None
        PERSONS = None
        table_p = None
        table_g = None
        # del modelD
        # del PERSONS


def make_mlna_top_k_variable_v2(
        default,
        x_traini,
        x_testi,
        y_traini,
        y_testi,
        OHE,
        nominal_factor_colums,
        cwd,
        domain,
        fix_imbalance,
        target_variable,
        custom_color,
        modelD,
        verbose,
        clfs,
        alpha,
        withCost,
        duration_divider,
        rate_divider,
        financialOption,
        original,
        graphWithClass=False,
        topR=[]
):
    # local eval storage
    logic_g = []
    logic_p = []
    logic_pg = []
    ## visualization of result
    modelD = model_desc()
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)
    PERSONS_T = get_persons(x_test)
    exitingMLNB = [dirnames for _, dirnames, _ in os.walk(f'{cwd}')][0]
    exitingMLNB = sorted([int(el.split("_")[1]) for el in exitingMLNB if "_b" in el])
    print(exitingMLNB,"//", alpha)
    BexitingMLNB = exitingMLNB[-1] if len(exitingMLNB) > 0 else None

    start = BexitingMLNB + 1 if len(exitingMLNB) > 0 else 2
    for k in range(start, len(topR) + 1):  # for 1<k<|OHE[i]|+2
        # for k in [2]: # for 1<k<|OHE[i]|+2
        # for k in range(2:len(OHE)+1: # for 1<k<|OHE[i]|+2
        layer_config = topR[:k]  # create subsets of k index of OHE and fetch it
        print(layer_config)
        # for layer_config in [[1,2]]: # create subsets of k index of OHE and fetch it
        # logic storage
        logic_i_g = []
        logic_i_p = []
        logic_i_pg = []
        copT = x_train.copy(deep=True)
        copT[target_variable] = y_train.copy(deep=True)
        # build the MLN for the variable i
        MLN = build_mlg_with_class(copT, [OHE[i] for i in layer_config],
                                   target_variable) if (graphWithClass is True) else build_mlg(
            copT, [OHE[i] for i in layer_config])
        col_targeted = [f'{nominal_factor_colums[i]}' for i in layer_config]
        case_k = '_'.join(col_targeted)
        # save the graph
        save_graph(
            cwd=cwd + f'/mlna_{k}_b',
            graph=MLN,
            name=f'{case_k}_mln',
            rows_len=copT.shape[0],
            prefix=domain,
            cols_len=len(OHE)
        )
        extracts_g = {
            "Att_DEGREE_GLO": [],
            "Att_INTRA_GLO": [],
            "Att_INTER_GLO": [],
            "Att_COMBINE_GLO": [],
            "Att_M_INTRA_GLO": [],
            "Att_M_INTER_GLO": [],
            "Att_M_COMBINE_GLO": []
        }
        extracts_p = {
            "Att_DEGREE_PER": [],  # GLO
            "Att_COMBINE_PER": [],
            "YN_COMBINE_PER": [],
            "YP_COMBINE_PER": [],  # COM
            "Att_INTER_PER": [],
            "YN_INTER_PER": [],
            "YP_INTER_PER": [],  # INTER
            "Att_INTRA_PER": [],
            "YN_INTRA_PER": [],
            "YP_INTRA_PER": [],  # INTRA
            "Att_M_COMBINE_PER": [],
            "Att_M_INTER_PER": [],
            "Att_M_INTRA_PER": []
        }
        extracts_g_t = copy.deepcopy(extracts_g)
        extracts_p_t = copy.deepcopy(extracts_p)

        ##################################
        ####### Training Descriptor ######
        ##################################
        for borrower in PERSONS:
            # compute descriptors for current borrower
            current = extract_descriptors_from_graph_model(
                graph=MLN,
                y_graph=removeEdge(MLN, k, copT.loc[borrower, target_variable], borrower),
                graphWithClass=graphWithClass,
                alpha=alpha,
                borrower=borrower,
                layers=k
            )
            # extract descriptors context
            # print(f"{list(current.keys())} <--> {list(extracts_g.keys())} <--> {list(extracts_p.keys())}")
            for key in list(current.keys()):
                if 'GLO' in key:
                    extracts_g[key].append(current[key])
                else:
                    extracts_p[key].append(current[key])

        ##################################
        ####### Test Descriptor     ######
        ##################################
        for borrower in PERSONS_T:
            # compute descriptors for current borrower
            grf = add_specific_loan_in_mlg(MLN, x_test.loc[[borrower]], [OHE[i] for i in layer_config])
            current = extract_descriptors_from_graph_model(
                graph=grf,
                y_graph=grf,
                graphWithClass=graphWithClass,
                alpha=alpha,
                borrower=borrower,
                layers=k
            )
            # extract descriptors context
            for key in list(current.keys()):
                if 'GLO' in key:
                    extracts_g_t[key].append(current[key])
                else:
                    extracts_p_t[key].append(current[key])

        ########################################
        ####### Descriptors Normalisation ######
        ########################################
        if graphWithClass is False:
            # Deleting a class descriptor using del
            for key in list(extracts_g_t.keys()):
                if 'Y' in key:
                    del extracts_g_t[key]
                    del extracts_g[key]
            for key in list(extracts_p_t.keys()):
                if 'Y' in key:
                    del extracts_p[key]
                    del extracts_p_t[key]

        # get the max value of each descriptor in both train and test dataset
        # maxGdesc = get_maximun_std_descriptor(extracts_g, extracts_g_t, extracts_g.keys())
        # maxPdesc = get_maximun_std_descriptor(extracts_p, extracts_p_t, extracts_p.keys())
        # print(f"{maxGDesc} <------> {maxPDesc}")
        # standard_extraction(extracts_g, extracts_g.keys(), maxGdesc)
        # standard_extraction(extracts_p, extracts_p.keys(), maxPdesc)
        # # print(f"{maxGDesc} <------> {maxPDesc}")
        # standard_extraction(extracts_g_t, extracts_g.keys(), maxGdesc)
        # standard_extraction(extracts_p_t, extracts_p.keys(), maxPdesc)

        maxGDesc = standard_extraction(extracts_g, extracts_g.keys())
        maxPDesc = standard_extraction(extracts_p, extracts_p.keys())
        # print(f"{maxGDesc} <------> {maxPDesc}")
        standard_extraction(extracts_g_t, extracts_g.keys(), maxGDesc)
        standard_extraction(extracts_p_t, extracts_p.keys(), maxPDesc)
        ##########################################
        ####### Descriptors Config Generator ######
        ##########################################
        config_df = generate_config_df(
            cwd=cwd,
            graphWithClass=graphWithClass,
            mlnL=f'/mlna_{k}_b',
            domain=domain,
            extracts_g=extracts_g,
            extracts_p=extracts_p,
            extracts_g_t=extracts_g_t,
            extracts_p_t=extracts_p_t,
            name=case_k
        )

        # free used ressources
        extract_g_df = None
        extract_p_df = None
        extract_g_df_t = None
        extract_p_df_t = None
        MLN = None
        bipart_combine = None
        bipart_intra_pagerank = None
        bipart_inter_pagerank = None

        ##########################################
        ####### Build MlC config            ######
        ##########################################
        mlc_cf = build_MlC(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            graphWithClass=graphWithClass,
            config_df=config_df,
            logic_i_p=logic_i_p,
            fix_imbalance=fix_imbalance,
            target_variable=target_variable,
            clfs=clfs,
            domain=domain,
            verbose=verbose,
            cwd=cwd,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            withCost=withCost,
            financialOption=financialOption,
            logic_i_g=logic_i_g,
            logic_i_pg=logic_i_pg,
            mlnL=f'/mlna_{k}_b',
            original = original,
            name=case_k
        )
        extracts_g = None
        extracts_p = None
        extracts_g_t = None
        extracts_p_t = None

        ## default - MLNa
        mlna = set()
        col_list = x_train.columns.to_list()
        for i in layer_config:
            for column in OHE[i]:
                if column in col_list:
                    mlna.add(column)
                else:
                    mlna.add(column.split("__")[-1])

        build_MCA(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            graphWithClass=graphWithClass,
            config_df=config_df,
            fix_imbalance=fix_imbalance,
            target_variable=target_variable,
            clfs=clfs,
            domain=domain,
            verbose=verbose,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            cwd=cwd,
            withCost=withCost,
            financialOption=financialOption,
            logic_i_g=logic_i_g,
            logic_i_p=logic_i_p,
            logic_i_pg=logic_i_pg,
            mlna=mlna,
            mlc_cf=mlc_cf,
            mlnL=f'/mlna_{k}_b',
            original = original,
            name=case_k
        )

        ########### END
        #########################################################################
        logic_p = [*logic_p, *logic_i_p]
        logic_g = [*logic_g, *logic_i_g]
        logic_pg = [*logic_pg, *logic_i_pg]
        table_p = print_summary([default, *logic_i_p], modelD)
        table_g = print_summary([default, *logic_i_g], modelD)
        table_pg = print_summary([default, *logic_i_pg], modelD)
        create_file(
            content=table_p[1],
            cwd=cwd + f'/mlna_{k}_b/personalized',
            prefix=domain,
            filename=f"mlna_for_{case_k}",
            extension=".html"
        )
        create_file(
            content=table_g[1],
            cwd=cwd + f'/mlna_{k}_b/global',
            prefix=domain,
            filename=f"mlna_for_{case_k}",
            extension=".html"
        )
        create_file(
            content=table_pg[1],
            cwd=cwd + f'/mlna_{k}_b/mixed',
            prefix=domain,
            filename=f"mlna_for_{case_k}",
            extension=".html"
        )
        table_g = None
        table_p = None
        logic_i_g = None
        logic_i_p = None
        mlc_cf = None
    if start != len(topR) + 1:
        table_p = print_summary([default, *logic_p], modelD)
        table_g = print_summary([default, *logic_g], modelD)
        table_pg = print_summary([default, *logic_pg], modelD)
        create_file(
            content=table_p[1],
            cwd=cwd + f'/mlna_{k}_b/personalized',
            prefix=domain,
            filename=f"mlna_for_all_categorial data",
            extension=".html"
        )
        create_file(
            content=table_g[1],
            cwd=cwd + f'/mlna_{k}_b/global',
            prefix=domain,
            filename=f"mlna_for_all_categorial data",
            extension=".html"
        )
        create_file(
            content=table_pg[1],
            cwd=cwd + f'/mlna_{k}_b/mixed',
            prefix=domain,
            filename=f"mlna_for_all_categorial data",
            extension=".html"
        )
    modelD = None
    PERSONS = None
    table_p = None
    table_g = None
    # del modelD
    # del PERSONS

# @profile
def make_mlna_1_variable(
        default,
        OHE,
        nominal_factor_colums,
        cwd,
        domain,
        fix_imbalance,
        target_variable,
        custom_color,
        modelD,
        verbose,
        clfs,
        alpha,
        withCost,
        x_traini,
        x_testi,
        y_traini,
        y_testi,
        financialOption,
        graphWithClass=False
):
    """
    Multilayers networks analysis with one attribut layer
    Parameters
    ----------
    default
    OHE
    nominal_factor_colums
    cwd
    domain
    fix_imbalance
    target_variable
    custom_color
    modelD
    verbose
    clfs
    alpha
    withCost
    x_traini
    x_testi
    y_traini
    y_testi
    financialOption
    graphWithClass

    Returns
    -------

    """
    ## visualization of result
    modelD = model_desc()
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)
    # print(PERSONS)
    # print(PERSONS_T)
    # print(list(y_train.values))
    # print(PERSONS)
    # local eval storage
    logic_g = []
    logic_p = []
    logic_pg = []
    # variable to store OHE which outperform default model
    outperformers = {}
    NbGood = 0
    for i in range(len(OHE)):
        # logic storage
        logic_i_g = []
        logic_i_p = []
        logic_i_pg = []
        # build the MLN for the variable i on training dataset
        copT = x_train.copy(deep=True)
        copT[target_variable] = y_train.copy(deep=True)
        MLN = build_mlg_with_class(
            copT, [OHE[i]], target_variable) \
            if (graphWithClass is True) \
            else build_mlg(copT, [OHE[i]])
        # print(value_clfs.iloc[:,])
        # save the graph
        save_graph(
            cwd=cwd + '/mlna_1',
            graph=MLN,
            name=f'{nominal_factor_colums[i]}_mln',
            rows_len=pd.concat([x_train, y_train], axis=1).shape[0],
            prefix=domain,
            cols_len=len(OHE[i])
        )

        # Binome page rank
        bipart_combine = nx.pagerank(MLN, alpha=alpha)

        # get intra page rank
        bipart_intra_pagerank = nx.pagerank(
            MLN,
            personalization=compute_personlization(get_intra_node_label(MLN), MLN),
            alpha=alpha
        )

        # get inter page rank
        bipart_inter_pagerank = nx.pagerank(
            MLN,
            personalization=compute_personlization(get_inter_node_label(MLN), MLN),
            alpha=alpha
        )

        # extract centrality degree from MLN
        extracts_g = {}
        extracts_p = {}


        """
        Compute descriptors for training set
        """
        extracts_g[f'Att_DEGREE_GLO'] = [
            get_number_of_borrowers_with_same_n_layer_value(borrower=index, graph=MLN, layer_nber=0)[1] for index in
            PERSONS
        ]
        extracts_p[f'Att_DEGREE_PER'] = extracts_g[f'Att_DEGREE_GLO']
        extracts_g[f'Att_INTRA_GLO'] = get_max_borrower_pr(bipart_intra_pagerank)[0]
        extracts_g[f'Att_INTER_GLO'] = get_max_borrower_pr(bipart_inter_pagerank)[0]
        extracts_g[f'Att_COMBINE_GLO'] = get_max_borrower_pr(bipart_combine)[0]
        extracts_p[f'Att_COMBINE_PER'] = [get_max_borrower_pr(nx.pagerank(MLN, personalization=compute_personlization(
            get_combine_perso_nodes_label(MLN, [val], 1)[0][0], MLN), alpha=alpha))[1][val] for index, val in
                                          enumerate(PERSONS)]
        if graphWithClass is True:
            extracts_p[f'YN_COMBINE_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                         personalization=compute_personlization(
                                             get_combine_perso_nodes_label(
                                                 removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                                 [
                                                     val],
                                                 1)[
                                                 0][
                                                 0],
                                             removeEdge(MLN, 1, copT.loc[val,target_variable], val)),
                                         alpha=alpha))[
                    0] for index, val in
                enumerate(PERSONS)]
            extracts_p[f'YP_COMBINE_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                         personalization=compute_personlization(
                                             get_combine_perso_nodes_label(
                                                 removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                                 [
                                                     val],
                                                 1)[
                                                 0][
                                                 0],
                                             removeEdge(MLN, 1, copT.loc[val,target_variable], val)),
                                         alpha=alpha))[
                    1] for index, val in
                enumerate(PERSONS)]
        extracts_p[f'Att_INTER_PER'] = [get_max_borrower_pr(nx.pagerank(MLN, personalization=compute_personlization(
                get_inter_perso_nodes_label(MLN, [val], 1)[0][0], MLN), alpha=alpha))[1][val] for index, val in
                                            enumerate(PERSONS)]
        if graphWithClass == True:
            extracts_p[f'YN_INTER_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                         personalization=compute_personlization(
                                             get_inter_perso_nodes_label(
                                                 removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                                 [
                                                     val],
                                                 1)[
                                                 0][
                                                 0],
                                             removeEdge(MLN, 1, copT.loc[val,target_variable], val)),
                                         alpha=alpha))[
                    0] for index, val in
                enumerate(PERSONS)]
            extracts_p[f'YP_INTER_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                         personalization=compute_personlization(
                                             get_inter_perso_nodes_label(
                                                 removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                                 [
                                                     val],
                                                 1)[
                                                 0][
                                                 0],
                                             removeEdge(MLN, 1, copT.loc[val,target_variable], val)),
                                         alpha=alpha))[
                    1] for index, val in
                enumerate(PERSONS)]
        extracts_p[f'Att_INTRA_PER'] = [get_max_borrower_pr(nx.pagerank(MLN, personalization=compute_personlization(
            get_intra_perso_nodes_label(MLN, [val], 1)[0][0], MLN), alpha=alpha))[1][val] for index, val in
                                        enumerate(PERSONS)]
        if graphWithClass == True:
            extracts_p[f'YN_INTRA_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                         personalization=compute_personlization(
                                             get_intra_perso_nodes_label(
                                                 removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                                 [
                                                     val],
                                                 1)[
                                                 0][
                                                 0],
                                             removeEdge(MLN, 1, copT.loc[val,target_variable], val)),
                                         alpha=alpha))[
                    0] for index, val in
                enumerate(PERSONS)]
            extracts_p[f'YP_INTRA_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                         personalization=compute_personlization(
                                             get_intra_perso_nodes_label(
                                                 removeEdge(MLN, 1, copT.loc[val,target_variable], val),
                                                 [
                                                     val],
                                                 1)[
                                                 0][
                                                 0],
                                             removeEdge(MLN, 1, copT.loc[val,target_variable], val)),
                                         alpha=alpha))[
                    1] for index, val in
                enumerate(PERSONS)]
        extracts_g[f'Att_M_INTRA_GLO'] = [get_max_modality_pagerank_score(index, MLN, 1, bipart_intra_pagerank) for
                                          index in PERSONS]
        extracts_g[f'Att_M_INTER_GLO'] = [get_max_modality_pagerank_score(index, MLN, 1, bipart_inter_pagerank) for
                                          index in PERSONS]
        extracts_g[f'Att_M_COMBINE_GLO'] = [get_max_modality_pagerank_score(index, MLN, 1, bipart_combine) for index in
                                            PERSONS]
        extracts_p[f'Att_M_COMBINE_PER'] = [get_max_modality_pagerank_score(val, MLN, 1, nx.pagerank(MLN,
                                                                                                     personalization=compute_personlization(
                                                                                                         get_combine_perso_nodes_label(
                                                                                                             MLN, [val],
                                                                                                             1)[0][0],
                                                                                                         MLN),
                                                                                                     alpha=alpha)) for
                                            index, val in enumerate(PERSONS)]
        extracts_p[f'Att_M_INTER_PER'] = [get_max_modality_pagerank_score(val, MLN, 1, nx.pagerank(MLN,
                                                                                                   personalization=compute_personlization(
                                                                                                       get_inter_perso_nodes_label(
                                                                                                           MLN, [val],
                                                                                                           1)[0][0],
                                                                                                       MLN),
                                                                                                   alpha=alpha)) for
                                          index, val in enumerate(PERSONS)]
        extracts_p[f'Att_M_INTRA_PER'] = [get_max_modality_pagerank_score(val, MLN, 1, nx.pagerank(MLN,
                                                                                                   personalization=compute_personlization(
                                                                                                       get_intra_perso_nodes_label(
                                                                                                           MLN, [val],
                                                                                                           1)[0][0],
                                                                                                       MLN),
                                                                                                   alpha=alpha)) for
                                          index, val in enumerate(PERSONS)]

        # print(f"""
        #         {extracts_p}
        #         {extracts_g}
        #         """)
        standard_extraction(extracts_g, extracts_g.keys())
        standard_extraction(extracts_p, extracts_p.keys())
        extracts_g_t,extracts_p_t = get_test_examples_descriptors_from_graph_model(
            MLN=MLN,
            x_test=x_test,
            graphWithClass=graphWithClass,
            OHE=OHE,
            i=i,
            alpha=alpha
        )


        """
        Merge descriptors ro build finals' sets
        """
        ## default - MLNa
        mlna = set()
        col_list = x_train.columns.to_list()
        for column in OHE[i]:
            if column in col_list:
                mlna.add(column)
            else:
                mlna.add(nominal_factor_colums[i])

        # classic
        # plot_features_importance_as_barh(
        # 	    data= default[1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= "Classic features importance",
        # 	    prefix= domain,
        # 	    cwd= cwd+'/mlna_1',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)

        # save descriptors
        """
        train
        """
        extract_MX_GLO_df = pd.DataFrame(extracts_g)
        extract_MX_PER_df = pd.DataFrame({key: extracts_p[key] for key in extracts_p.keys() if not ("Y" in key)})
        if graphWithClass == True:
            extract_MY_PER_df = pd.DataFrame({key: extracts_p[key] for key in extracts_p.keys() if ("Y" in key)})
            extract_MXY_PER_df = pd.DataFrame(extracts_p)

        """
        test
        """
        extract_MX_GLO_df_t = pd.DataFrame(extracts_g_t)
        extract_MX_PER_df_t = pd.DataFrame({key: extracts_p_t[key] for key in extracts_p_t.keys() if not ("Y" in key)})
        if graphWithClass == True:
            extract_MY_PER_df_t = pd.DataFrame({key: extracts_p_t[key] for key in extracts_p_t.keys() if ("Y" in key)})
            extract_MXY_PER_df_t = pd.DataFrame(extracts_p_t)

        save_dataset(
            cwd=cwd + '/mlna_1/global',
            dataframe=extract_MX_GLO_df,
            name=f'{domain}_extract_MX_GLO_df_for_{nominal_factor_colums[i]}',
            prefix=domain,
            sep=','
        )
        save_dataset(
            cwd=cwd + '/mlna_1/personalized',
            dataframe=extract_MX_PER_df,
            name=f'{domain}_extract_MX_PER_df_for_{nominal_factor_colums[i]}',
            prefix=domain,
            sep=','
        )
        save_dataset(
            cwd=cwd + '/mlna_1/global',
            dataframe=extract_MX_GLO_df_t,
            name=f'{domain}_extract_MX_GLO_df_t_for_{nominal_factor_colums[i]}',
            prefix=domain,
            sep=','
        )
        save_dataset(
            cwd=cwd + '/mlna_1/personalized',
            dataframe=extract_MX_PER_df_t,
            name=f'{domain}_extract_MX_PER_df_t_for_{nominal_factor_colums[i]}',
            prefix=domain,
            sep=','
        )
        if graphWithClass == True:
            save_dataset(
                cwd=cwd + '/mlna_1/personalized',
                dataframe=extract_MY_PER_df,
                name=f'{domain}_extract_MY_PER_df_for_{nominal_factor_colums[i]}',
                prefix=domain,
                sep=','
            )
            save_dataset(
                cwd=cwd + '/mlna_1/personalized',
                dataframe=extract_MXY_PER_df,
                name=f'{domain}_extract_MXY_PER_df_for_{nominal_factor_colums[i]}',
                prefix=domain,
                sep=','
            )
            save_dataset(
                cwd=cwd + '/mlna_1/personalized',
                dataframe=extract_MY_PER_df_t,
                name=f'{domain}_extract_MY_PER_df_t_for_{nominal_factor_colums[i]}',
                prefix=domain,
                sep=','
            )
            save_dataset(
                cwd=cwd + '/mlna_1/personalized',
                dataframe=extract_MXY_PER_df_t,
                name=f'{domain}_extract_MXY_PER_df_t_for_{nominal_factor_colums[i]}',
                prefix=domain,
                sep=','
            )
        extract_g_df = None
        extract_p_df = None
        extract_g_df_t = None
        extract_p_df_t = None
        MLN = None
        bipart_combine = None
        bipart_intra_pagerank = None
        bipart_inter_pagerank = None
        # del extract_g_df
        # del extract_p_df
        # del MLN
        # del bipart_combine
        # del bipart_intra_pagerank
        # del bipart_inter_pagerank
        ## default + MLN
        # inject descriptors
        """
        train
        """
        MlC_GLO_MX = inject_features_extracted(x_train, extract_MX_GLO_df)

        MlC_PER_MX = inject_features_extracted(x_train, extract_MX_PER_df)
        if graphWithClass == True:
            MlC_PER_MY = inject_features_extracted(x_train, extract_MY_PER_df)
            MlC_PER_MXY = inject_features_extracted(x_train, extract_MXY_PER_df)

        MlC_GAP_MX = inject_features_extracted(MlC_GLO_MX, extract_MX_PER_df)
        if graphWithClass == True:
            MlC_GAP_MY = inject_features_extracted(MlC_GLO_MX, extract_MY_PER_df)
            MlC_GAP_MXY = inject_features_extracted(MlC_GLO_MX, extract_MXY_PER_df)


        """
        test
        """
        MlC_GLO_MX_T = inject_features_extracted(x_test, extract_MX_GLO_df_t)

        MlC_PER_MX_T = inject_features_extracted(x_test, extract_MX_PER_df_t)
        if graphWithClass == True:
            MlC_PER_MY_T = inject_features_extracted(x_test, extract_MY_PER_df_t)
            MlC_PER_MXY_T = inject_features_extracted(x_test, extract_MXY_PER_df_t)

        MlC_GAP_MX_T = inject_features_extracted(MlC_GLO_MX_T, extract_MX_PER_df_t)
        if graphWithClass == True:
            MlC_GAP_MY_T = inject_features_extracted(MlC_GLO_MX_T, extract_MY_PER_df_t)
            MlC_GAP_MXY_T = inject_features_extracted(MlC_GLO_MX_T, extract_MXY_PER_df_t)
        extracts_g = None
        extracts_p = None
        extracts_g_t = None
        extracts_p_t = None
        # del extracts_g
        # del extracts_p
        # print(MlC_PER_MX.isnull().sum())
        # print(MlC_PER_MX_T.isnull().sum())
        logic_i_p.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini = MlC_PER_MX,
                x_testi = MlC_PER_MX_T,
                y_traini = y_train,
                y_testi = y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MlC_PER_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + '/mlna_1/personalized',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        if graphWithClass == True:
            logic_i_p.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini = MlC_PER_MY,
                    x_testi = MlC_PER_MY_T,
                    y_traini = y_train,
                    y_testi = y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MlC_PER_MY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + '/mlna_1/personalized',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            logic_i_p.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini = MlC_PER_MXY,
                    x_testi = MlC_PER_MXY_T,
                    y_traini = y_train,
                    y_testi = y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MlC_PER_MXY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + '/mlna_1/personalized',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
        # print(f"""
        # {MlC_GLO_MX.columns}
        # {len(MlC_GLO_MX.columns)}
        # {MlC_GLO_MX_T.columns}
        # {len(MlC_GLO_MX_T.columns)}
        # """)
        logic_i_g.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini = MlC_GLO_MX,
                x_testi = MlC_GLO_MX_T,
                y_traini = y_train,
                y_testi = y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MlC_GLO_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + '/mlna_1/global',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        logic_i_pg.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini = MlC_GAP_MX,
                x_testi = MlC_GAP_MX_T,
                y_traini = y_train,
                y_testi = y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MlC_GAP_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + '/mlna_1/mixed',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        if graphWithClass == True:
            logic_i_pg.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini = MlC_GAP_MY,
                    x_testi = MlC_GAP_MY_T,
                    y_traini = y_train,
                    y_testi = y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MlC_GAP_MY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + '/mlna_1/mixed',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            logic_i_pg.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini = MlC_GAP_MXY,
                    x_testi = MlC_GAP_MXY_T,
                    y_traini = y_train,
                    y_testi = y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MlC_GAP_MXY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + '/mlna_1/mixed',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )

        # classic + mln
        # plot_features_importance_as_barh(
        # 	    data= logic_i_p[-1][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+'/mlna_1/personalized',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)
        # plot_features_importance_as_barh(
        # 	    data= logic_i_g[-1][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+'/mlna_1/global',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)
        # plot_features_importance_as_barh(
        # 	    data= logic_i_pg[-1][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+'/mlna_1/mixed',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)

        # VALUE_MINUS_MLNa = value_clfs.drop(list(mlna), axis=1)
        #
        # logic_i_g.append(
        # 	make_builder(
        # 		fix_imbalance=fix_imbalance,
        # 		DATA_OVER=VALUE_MINUS_MLNa,
        # 		target_variable=target_variable,
        # 		clfs=clfs,
        # 		domain= f'classic_-_mlna',
        # 		prefix=domain,
        # 		verbose=verbose,
        # 		cwd= cwd+'/mlna_1'
        # 		)
        # 	)
        # logic_i_p.append(logic_i_g[-1])
        # logic_i_pg.append(logic_i_g[-1])
        # classic - mln attribut
        # plot_features_importance_as_barh(
        # 	    data= logic_i_g[-1][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic - mln attributs features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+'/mlna_1',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)

        ## default + MLN - MLNa
        """
        train
        """
        MCA_GLO_MX = MlC_GLO_MX.drop(list(mlna), axis=1)

        MCA_PER_MX = MlC_PER_MX.drop(list(mlna), axis=1)
        if graphWithClass == True:
            MCA_PER_MY = MlC_PER_MY.drop(list(mlna), axis=1)
            MCA_PER_MXY = MlC_PER_MXY.drop(list(mlna), axis=1)

        MCA_GAP_MX = MlC_GAP_MX.drop(list(mlna), axis=1)
        if graphWithClass == True:
            MCA_GAP_MY = MlC_GAP_MY.drop(list(mlna), axis=1)
            MCA_GAP_MXY = MlC_GAP_MXY.drop(list(mlna), axis=1)

        """
        test
        """
        MCA_GLO_MX_T = MlC_GLO_MX_T.drop(list(mlna), axis=1)

        MCA_PER_MX_T = MlC_PER_MX_T.drop(list(mlna), axis=1)
        if graphWithClass == True:
            MCA_PER_MY_T = MlC_PER_MY_T.drop(list(mlna), axis=1)
            MCA_PER_MXY_T = MlC_PER_MXY_T.drop(list(mlna), axis=1)

        MCA_GAP_MX_T = MlC_GAP_MX_T.drop(list(mlna), axis=1)
        if graphWithClass == True:
            MCA_GAP_MY_T = MlC_GAP_MY_T.drop(list(mlna), axis=1)
            MCA_GAP_MXY_T = MlC_GAP_MXY_T.drop(list(mlna), axis=1)

        logic_i_p.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini = MCA_PER_MX,
                x_testi = MCA_PER_MX_T,
                y_traini = y_train,
                y_testi = y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MCA_PER_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + '/mlna_1/personalized',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        if graphWithClass == True:
            logic_i_p.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini = MCA_PER_MY,
                    x_testi = MCA_PER_MY_T,
                    y_traini = y_train,
                    y_testi = y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MCA_PER_MY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + '/mlna_1/personalized',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            logic_i_p.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini = MCA_PER_MXY,
                    x_testi = MCA_PER_MXY_T,
                    y_traini = y_train,
                    y_testi = y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MCA_PER_MXY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + '/mlna_1/personalized',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
        logic_i_g.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini = MCA_GLO_MX,
                x_testi = MCA_GLO_MX_T,
                y_traini = y_train,
                y_testi = y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MCA_GLO_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + '/mlna_1/global',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        logic_i_pg.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini = MCA_GAP_MX,
                x_testi = MCA_GAP_MX_T,
                y_traini = y_train,
                y_testi = y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MCA_GAP_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + '/mlna_1/mixed',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        if graphWithClass == True:
            logic_i_pg.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini = MCA_GAP_MY,
                    x_testi = MCA_GAP_MY_T,
                    y_traini = y_train,
                    y_testi = y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MCA_GAP_MY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + '/mlna_1/mixed',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            logic_i_pg.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini = MCA_GAP_MXY,
                    x_testi = MCA_GAP_MXY_T,
                    y_traini = y_train,
                    y_testi = y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MCA_GAP_MXY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + '/mlna_1/mixed',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
        # classic + mln - mln attribut
        # plot_features_importance_as_barh(
        # 	    data= logic_i_p[-1][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln - mln attributs features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+'/mlna_1/personalized',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)
        # plot_features_importance_as_barh(
        # 	    data= logic_i_g[-1][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln - mln attributs features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+'/mlna_1/global',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)
        # plot_features_importance_as_barh(
        # 	    data= logic_i_pg[-1][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln - mln attributs features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+'/mlna_1/mixed',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)

        # print html report for analysis of variable i
        # logic_i=[default,*logic_i]

        #######################################################################
        ######### OHE selection
        # find the best accuracy in between all training set
        # compare it to the best accuracy in default training
        # if greater, save the index and the gain
        bestp = logic_i_p[5 if graphWithClass else 1][1].sort_values(
            by="accuracy",
            axis=0,
            ascending=False
            # ,key=lambda row: abs(row)
        ).head(1)
        print(f"{bestp} best personalized model")
        bestf = default[1].sort_values(
            by="accuracy",
            axis=0,
            ascending=False
            # ,key=lambda row: abs(row)
        ).head(1)

        if (bestp["accuracy"][list(bestp.index)[0]] > bestf["accuracy"][list(bestf.index)[0]]):
            NbGood += 1
        print(
            f'{list(bestp.index)[0]}--{bestp["accuracy"][list(bestp.index)[0]]}--' +
            f'{round(((round(bestp["accuracy"][list(bestp.index)[0]], 4) - round(bestf["accuracy"][list(bestf.index)[0]], 4)) / round(bestf["accuracy"][list(bestf.index)[0]], 4)) * 100, 4)} gain personalized model')
        outperformers[i] = round(((round(bestp["accuracy"][list(bestp.index)[0]], 4) - round(
            bestf["accuracy"][list(bestf.index)[0]], 4)) / round(bestf["accuracy"][list(bestf.index)[0]], 4)) * 100, 4)
        ########### END
        #########################################################################
        logic_p = [*logic_p, *logic_i_p]
        logic_g = [*logic_g, *logic_i_g]
        logic_pg = [*logic_pg, *logic_i_pg]
        table_p = print_summary([default, *logic_i_p], modelD)
        table_g = print_summary([default, *logic_i_g], modelD)
        table_pg = print_summary([default, *logic_i_pg], modelD)
        create_file(
            content=table_p[1],
            cwd=cwd + '/mlna_1/personalized',
            prefix=domain,
            filename=f"mlna_for_{nominal_factor_colums[i]}",
            extension=".html"
        )
        create_file(
            content=table_g[1],
            cwd=cwd + '/mlna_1/global',
            prefix=domain,
            filename=f"mlna_for_{nominal_factor_colums[i]}",
            extension=".html"
        )
        create_file(
            content=table_pg[1],
            cwd=cwd + '/mlna_1/mixed',
            prefix=domain,
            filename=f"mlna_for_{nominal_factor_colums[i]}",
            extension=".html"
        )
        table_g = None
        table_p = None
        logic_i_g = None
        logic_i_p = None
        MCA_GLO_MX = None

        MCA_PER_MX = None
        MCA_PER_MY = None
        MCA_PER_MXY = None

        MCA_GAP_MX = None
        MCA_GAP_MY = None
        MCA_GAP_MXY = None

        MlC_GLO_MX = None

        MlC_PER_MX = None
        MlC_PER_MY = None
        MlC_PER_MXY = None

        MlC_GAP_MX = None
        MlC_GAP_MY = None
        MlC_GAP_MXY = None

    # del table_g
    # del table_p
    # del logic_i_g
    # del logic_i_p
    # del VALUE_MLN_G
    # del VALUE_MLN_P
    # del VALUE_MLN_MINUS_MLN_P
    # del VALUE_MLN_MINUS_MLN_G
    # del VALUE_MINUS_MLNa
    table_p = print_summary([default, *logic_p], modelD)
    table_g = print_summary([default, *logic_g], modelD)
    table_pg = print_summary([default, *logic_pg], modelD)
    create_file(
        content=table_p[1],
        cwd=cwd + '/mlna_1/personalized',
        prefix=domain,
        filename=f"mlna_for_all_categorial data",
        extension=".html"
    )
    create_file(
        content=table_g[1],
        cwd=cwd + '/mlna_1/global',
        prefix=domain,
        filename=f"mlna_for_all_categorial data",
        extension=".html"
    )
    create_file(
        content=table_pg[1],
        cwd=cwd + '/mlna_1/mixed',
        prefix=domain,
        filename=f"mlna_for_all_categorial data",
        extension=".html"
    )
    modelD = None
    PERSONS = None
    table_p = None
    table_g = None
    # del modelD
    # del PERSONS
    return (outperformers, NbGood)

# @profile
def make_mlna_k_variable(
        default,
        x_traini,
        x_testi,
        y_traini,
        y_testi,
        OHE,
        nominal_factor_colums,
        cwd,
        domain,
        fix_imbalance,
        target_variable,
        custom_color,
        modelD,
        verbose,
        clfs,
        alpha,
        withCost,
        financialOption,
        graphWithClass=True
):
    """
    """

    # local eval storage
    logic_g = []
    logic_p = []
    logic_pg = []
    ## visualization of result
    modelD = model_desc()
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)

    for k in list(set([len(OHE)])):  # for 1<k<|OHE[i]|+2
        # for k in [2]: # for 1<k<|OHE[i]|+2
        # for k in range(2:len(OHE)+1: # for 1<k<|OHE[i]|+2
        for layer_config in get_combinations(range(len(OHE)), k):  # create subsets of k index of OHE and fetch it
            # for layer_config in [[1,2]]: # create subsets of k index of OHE and fetch it
            # logic storage
            logic_i_g = []
            logic_i_p = []
            logic_i_pg = []
            copT = x_train.copy(deep=True)
            copT[target_variable] = y_train.copy(deep=True)
            # build the MLN for the variable i
            MLN = build_mlg_with_class(copT, [OHE[i] for i in layer_config],
                                       target_variable) if (graphWithClass is True) else build_mlg(copT,
                                                                                                   [OHE[i] for i in
                                                                                                    layer_config])
            col_targeted = [f'{nominal_factor_colums[i]}' for i in layer_config]
            case_k = '_'.join(col_targeted)
            # save the graph
            save_graph(
                cwd=cwd + f'/mlna_{k}',
                graph=MLN,
                name=f'{case_k}_mln',
                rows_len=pd.concat([x_train, y_train], axis=1).shape[0],
                prefix=domain,
                cols_len=len(OHE)
            )

            # Binome page rank
            bipart_combine = nx.pagerank(MLN, alpha=alpha)

            # get intra page rank
            bipart_intra_pagerank = nx.pagerank(
                MLN,
                personalization=compute_personlization(get_intra_node_label(MLN), MLN),
                alpha=alpha
            )

            # get inter page rank
            bipart_inter_pagerank = nx.pagerank(
                MLN,
                personalization=compute_personlization(get_inter_node_label(MLN), MLN),
                alpha=alpha
            )

            # extract centrality degree from MLN
            extracts_g = {}
            extracts_p = {}

            extracts_g[f'Att_DEGREE_GLO'] = [
                get_number_of_borrowers_with_same_custom_layer_value(borrower=index, graph=MLN, custom_layer=list(range(k)))[
                    1] for index in PERSONS
            ]
            extracts_p[f'Att_DEGREE_PER'] = extracts_g[f'Att_DEGREE_GLO']
            extracts_g[f'Att_INTRA_GLO'] = get_max_borrower_pr(bipart_intra_pagerank)[0]
            extracts_g[f'Att_INTER_GLO'] = get_max_borrower_pr(bipart_inter_pagerank)[0]
            extracts_g[f'Att_COMBINE_GLO'] = get_max_borrower_pr(bipart_combine)[0]
            extracts_p[f'Att_COMBINE_PER'] = [get_max_borrower_pr(nx.pagerank(MLN,
                                                                              personalization=compute_personlization(
                                                                                  get_combine_perso_nodes_label(MLN,
                                                                                                                [val],
                                                                                                                k)[0][
                                                                                      0], MLN), alpha=alpha))[1][val] for
                                              index, val in enumerate(PERSONS)]
            if graphWithClass == True:
                extracts_p[f'YN_COMBINE_PER'] = [
                    get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                             personalization=compute_personlization(
                                                 get_combine_perso_nodes_label(
                                                     removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[
                                                     0][0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                             alpha=alpha))[0] for
                    index, val in enumerate(PERSONS)]
                extracts_p[f'YP_COMBINE_PER'] = [
                    get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                             personalization=compute_personlization(
                                                 get_combine_perso_nodes_label(
                                                     removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[
                                                     0][0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                             alpha=alpha))[1] for
                    index, val in enumerate(PERSONS)]

            extracts_p[f'Att_INTER_PER'] = [get_max_borrower_pr(nx.pagerank(MLN, personalization=compute_personlization(
                get_inter_perso_nodes_label(MLN, [val], k)[0][0], MLN), alpha=alpha))[1][val] for index, val in
                                            enumerate(PERSONS)]
            if graphWithClass == True:
                extracts_p[f'YN_INTER_PER'] = [
                    get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                             personalization=compute_personlization(
                                                 get_inter_perso_nodes_label(
                                                     removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[0][
                                                     0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                             alpha=alpha))[0] for
                    index, val in enumerate(PERSONS)]
                extracts_p[f'YP_INTER_PER'] = [
                    get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                             personalization=compute_personlization(
                                                 get_inter_perso_nodes_label(
                                                     removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[0][
                                                     0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                             alpha=alpha))[1] for
                    index, val in enumerate(PERSONS)]

            extracts_p[f'Att_INTRA_PER'] = [get_max_borrower_pr(nx.pagerank(MLN, personalization=compute_personlization(
                get_intra_perso_nodes_label(MLN, [val], k)[0][0], MLN), alpha=alpha))[1][val] for index, val in
                                            enumerate(PERSONS)]
            if graphWithClass == True:
                extracts_p[f'YN_INTER_PER'] = [
                    get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                             personalization=compute_personlization(
                                                 get_intra_perso_nodes_label(
                                                     removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[0][
                                                     0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                             alpha=alpha))[0] for
                    index, val in enumerate(PERSONS)]
                extracts_p[f'YP_INTER_PER'] = [
                    get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                             personalization=compute_personlization(
                                                 get_intra_perso_nodes_label(
                                                     removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[0][
                                                     0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                             alpha=alpha))[1] for
                    index, val in enumerate(PERSONS)]

            extracts_g[f'Att_M_INTRA_GLO'] = [get_max_modality_pagerank_score(index, MLN, k, bipart_intra_pagerank) for
                                              index in PERSONS]
            extracts_g[f'Att_M_INTER_GLO'] = [get_max_modality_pagerank_score(index, MLN, k, bipart_inter_pagerank) for
                                              index in PERSONS]
            extracts_g[f'Att_M_COMBINE_GLO'] = [get_max_modality_pagerank_score(index, MLN, k, bipart_combine) for index
                                                in PERSONS]
            extracts_p[f'Att_M_COMBINE_PER'] = [get_max_modality_pagerank_score(val, MLN, k, nx.pagerank(MLN,
                                                                                                         personalization=compute_personlization(
                                                                                                             get_combine_perso_nodes_label(
                                                                                                                 MLN,
                                                                                                                 [val],
                                                                                                                 k)[0][
                                                                                                                 0],
                                                                                                             MLN),
                                                                                                         alpha=alpha))
                                                for index, val in enumerate(PERSONS)]
            extracts_p[f'Att_M_INTER_PER'] = [get_max_modality_pagerank_score(val, MLN, k, nx.pagerank(MLN,
                                                                                                       personalization=compute_personlization(
                                                                                                           get_inter_perso_nodes_label(
                                                                                                               MLN,
                                                                                                               [val],
                                                                                                               k)[0][0],
                                                                                                           MLN),
                                                                                                       alpha=alpha)) for
                                              index, val in enumerate(PERSONS)]
            extracts_p[f'Att_M_INTRA_PER'] = [get_max_modality_pagerank_score(val, MLN, k, nx.pagerank(MLN,
                                                                                                       personalization=compute_personlization(
                                                                                                           get_intra_perso_nodes_label(
                                                                                                               MLN,
                                                                                                               [val],
                                                                                                               k)[0][0],
                                                                                                           MLN),
                                                                                                       alpha=alpha)) for
                                              index, val in enumerate(PERSONS)]
            # standardization
            print(f"""
                {extracts_p}
                {extracts_g}
                """)
            standard_extraction(extracts_p, extracts_p.keys())
            standard_extraction(extracts_g, extracts_g.keys())
            extracts_g_t, extracts_p_t = get_test_examples_descriptors_from_graph_model(
                MLN=MLN,
                x_test=x_test,
                graphWithClass=graphWithClass,
                OHE=OHE,
                i=k,
                alpha=alpha,
                layers=k,
                layer_config=layer_config,
            )
            # save descriptors
            """
            train
            """
            print(f"""
                    {extracts_p}
                    {extracts_g}
                    """)
            extract_MX_GLO_df = pd.DataFrame(extracts_g)
            extract_MX_PER_df = pd.DataFrame({key: extracts_p[key] for key in extracts_p.keys() if not ("Y" in key)})
            if graphWithClass == True:
                extract_MY_PER_df = pd.DataFrame({key: extracts_p[key] for key in extracts_p.keys() if ("Y" in key)})
                extract_MXY_PER_df = pd.DataFrame(extracts_p)

            """
            test
            """
            extract_MX_GLO_df_t = pd.DataFrame(extracts_g_t)
            extract_MX_PER_df_t = pd.DataFrame(
                {key: extracts_p_t[key] for key in extracts_p_t.keys() if not ("Y" in key)})
            if graphWithClass == True:
                extract_MY_PER_df_t = pd.DataFrame(
                    {key: extracts_p_t[key] for key in extracts_p_t.keys() if ("Y" in key)})
                extract_MXY_PER_df_t = pd.DataFrame(extracts_p_t)

            save_dataset(
                cwd=cwd + f'/mlna_{k}/global',
                dataframe=extract_MX_GLO_df,
                name=f'{domain}_extract_MX_GLO_df_for_{case_k}',
                prefix=domain,
                sep=','
            )
            save_dataset(
                cwd=cwd + f'/mlna_{k}/personalized',
                dataframe=extract_MX_PER_df,
                name=f'{domain}_extract_MX_PER_df_for_{case_k}',
                prefix=domain,
                sep=','
            )
            save_dataset(
                cwd=cwd + f'/mlna_{k}/global',
                dataframe=extract_MX_GLO_df_t,
                name=f'{domain}_extract_MX_GLO_df_t_for_{case_k}',
                prefix=domain,
                sep=','
            )
            save_dataset(
                cwd=cwd + f'/mlna_{k}/personalized',
                dataframe=extract_MX_PER_df_t,
                name=f'{domain}_extract_MX_PER_df_t_for_{case_k}',
                prefix=domain,
                sep=','
            )
            if graphWithClass == True:
                save_dataset(
                    cwd=cwd + f'/mlna_{k}/personalized',
                    dataframe=extract_MY_PER_df,
                    name=f'{domain}_extract_MY_PER_df_for_{case_k}',
                    prefix=domain,
                    sep=','
                )
                save_dataset(
                    cwd=cwd + f'/mlna_{k}/personalized',
                    dataframe=extract_MXY_PER_df,
                    name=f'{domain}_extract_MXY_PER_df_for_{case_k}',
                    prefix=domain,
                    sep=','
                )
                save_dataset(
                    cwd=cwd + f'/mlna_{k}/personalized',
                    dataframe=extract_MY_PER_df_t,
                    name=f'{domain}_extract_MY_PER_df_t_for_{case_k}',
                    prefix=domain,
                    sep=','
                )
                save_dataset(
                    cwd=cwd + f'/mlna_{k}/personalized',
                    dataframe=extract_MXY_PER_df_t,
                    name=f'{domain}_extract_MXY_PER_df_t_for_{case_k}',
                    prefix=domain,
                    sep=','
                )
            extract_g_df = None
            extract_p_df = None
            MLN = None
            bipart_combine = None
            bipart_intra_pagerank = None
            bipart_inter_pagerank = None
            # del extract_g_df
            # del extract_p_df
            # del MLN
            # del bipart_combine
            # del bipart_intra_pagerank
            # del bipart_inter_pagerank

            mlna = set()
            col_list = x_train.columns.to_list()
            for i in layer_config:
                for column in OHE[i]:
                    if column in col_list:
                        mlna.add(column)
                    else:
                        mlna.add(column.split("__")[-1])
            # classic
            # plot_features_importance_as_barh(
            # 	    data= default[1],
            # 	    getColor= custom_color,
            # 	    modelDictName= modelD,
            # 	    plotTitle= "Classic features importance",
            # 	    prefix= domain,
            # 	    cwd= cwd+f'/mlna_{k}',
            # 	    graph_a= list(mlna),
            # 	    save= True
            # 	)
            ## default + MLN
            # inject descriptors
            """
            train
            """
            MlC_GLO_MX = inject_features_extracted(x_train, extract_MX_GLO_df)

            MlC_PER_MX = inject_features_extracted(x_train, extract_MX_PER_df)
            if graphWithClass == True:
                MlC_PER_MY = inject_features_extracted(x_train, extract_MY_PER_df)
                MlC_PER_MXY = inject_features_extracted(x_train, extract_MXY_PER_df)

            MlC_GAP_MX = inject_features_extracted(MlC_GLO_MX, extract_MX_PER_df)
            if graphWithClass == True:
                MlC_GAP_MY = inject_features_extracted(MlC_GLO_MX, extract_MY_PER_df)
                MlC_GAP_MXY = inject_features_extracted(MlC_GLO_MX, extract_MXY_PER_df)

            """
            test
            """
            MlC_GLO_MX_T = inject_features_extracted(x_test, extract_MX_GLO_df_t)

            MlC_PER_MX_T = inject_features_extracted(x_test, extract_MX_PER_df_t)
            if graphWithClass == True:
                MlC_PER_MY_T = inject_features_extracted(x_test, extract_MY_PER_df_t)
                MlC_PER_MXY_T = inject_features_extracted(x_test, extract_MXY_PER_df_t)

            MlC_GAP_MX_T = inject_features_extracted(MlC_GLO_MX_T, extract_MX_PER_df_t)
            if graphWithClass == True:
                MlC_GAP_MY_T = inject_features_extracted(MlC_GLO_MX_T, extract_MY_PER_df_t)
                MlC_GAP_MXY_T = inject_features_extracted(MlC_GLO_MX_T, extract_MXY_PER_df_t)
            extracts_g = None
            extracts_p = None
            extracts_g_t = None
            extracts_p_t = None
            # del extracts_g
            # del extracts_p
            # print(MlC_PER_MX.isnull().sum())
            # print(MlC_PER_MX_T.isnull().sum())
            logic_i_p.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MlC_PER_MX,
                    x_testi=MlC_PER_MX_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MlC_PER_MX',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}/personalized',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            if graphWithClass == True:
                logic_i_p.append(
                    make_builder(
                        fix_imbalance=fix_imbalance,
                        DATA_OVER=None,
                        x_traini=MlC_PER_MY,
                        x_testi=MlC_PER_MY_T,
                        y_traini=y_train,
                        y_testi=y_test,
                        target_variable=target_variable,
                        clfs=clfs,
                        domain=f'MlC_PER_MY',
                        prefix=domain,
                        verbose=verbose,
                        cwd=cwd + f'/mlna_{k}/personalized',
                        withCost=withCost,
                        financialOption=financialOption
                    )
                )
                logic_i_p.append(
                    make_builder(
                        fix_imbalance=fix_imbalance,
                        DATA_OVER=None,
                        x_traini=MlC_PER_MXY,
                        x_testi=MlC_PER_MXY_T,
                        y_traini=y_train,
                        y_testi=y_test,
                        target_variable=target_variable,
                        clfs=clfs,
                        domain=f'MlC_PER_MXY',
                        prefix=domain,
                        verbose=verbose,
                        cwd=cwd + f'/mlna_{k}/personalized',
                        withCost=withCost,
                        financialOption=financialOption
                    )
                )
            # print(f"""
            #         {MlC_GLO_MX.columns}
            #         {len(MlC_GLO_MX.columns)}
            #         {MlC_GLO_MX_T.columns}
            #         {len(MlC_GLO_MX_T.columns)}
            #         """)
            logic_i_g.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MlC_GLO_MX,
                    x_testi=MlC_GLO_MX_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MlC_GLO_MX',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}/global',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            logic_i_pg.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MlC_GAP_MX,
                    x_testi=MlC_GAP_MX_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MlC_GAP_MX',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}/mixed',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            if graphWithClass == True:
                logic_i_pg.append(
                    make_builder(
                        fix_imbalance=fix_imbalance,
                        DATA_OVER=None,
                        x_traini=MlC_GAP_MY,
                        x_testi=MlC_GAP_MY_T,
                        y_traini=y_train,
                        y_testi=y_test,
                        target_variable=target_variable,
                        clfs=clfs,
                        domain=f'MlC_GAP_MY',
                        prefix=domain,
                        verbose=verbose,
                        cwd=cwd + f'/mlna_{k}/mixed',
                        withCost=withCost,
                        financialOption=financialOption
                    )
                )
                logic_i_pg.append(
                    make_builder(
                        fix_imbalance=fix_imbalance,
                        DATA_OVER=None,
                        x_traini=MlC_GAP_MXY,
                        x_testi=MlC_GAP_MXY_T,
                        y_traini=y_train,
                        y_testi=y_test,
                        target_variable=target_variable,
                        clfs=clfs,
                        domain=f'MlC_GAP_MXY',
                        prefix=domain,
                        verbose=verbose,
                        cwd=cwd + f'/mlna_{k}/mixed',
                        withCost=withCost,
                        financialOption=financialOption
                    )
                )

            # classic + mln
            # plot_features_importance_as_barh(
            # 	    data= logic_i_g[0][1],
            # 	    getColor= custom_color,
            # 	    modelDictName= modelD,
            # 	    plotTitle= f"Classic + mln features importance for",
            # 	    prefix= domain,
            # 	    cwd= cwd+f'/mlna_{k}/global',
            # 	    graph_a= list(mlna),
            # 	    save= True
            # 	)
            # plot_features_importance_as_barh(
            # 	    data= logic_i_p[0][1],
            # 	    getColor= custom_color,
            # 	    modelDictName= modelD,
            # 	    plotTitle= f"Classic + mln features importance for",
            # 	    prefix= domain,
            # 	    cwd= cwd+f'/mlna_{k}/personalized',
            # 	    graph_a= list(mlna),
            # 	    save= True
            # 	)
            # plot_features_importance_as_barh(
            # 	    data= logic_i_pg[0][1],
            # 	    getColor= custom_color,
            # 	    modelDictName= modelD,
            # 	    plotTitle= f"Classic + mln features importance for",
            # 	    prefix= domain,
            # 	    cwd= cwd+f'/mlna_{k}/mixed',
            # 	    graph_a= list(mlna),
            # 	    save= True
            # 	)

            ## default - MLNa

            # is_full = list(set(col_list) - set(list(mlna)))
            # print(f"{col_list} + {mlna} + {is_full}")
            # if len(is_full) > 1: #  check if operation is possible
            # 	#print(typeof(mlna))
            #
            # 	VALUE_MINUS_MLNa = value_clfs.drop(list(mlna), axis=1)
            #
            # 	logic_i_g.append(
            # 		make_builder(
            # 			fix_imbalance=fix_imbalance,
            # 			DATA_OVER=VALUE_MINUS_MLNa,
            # 			target_variable=target_variable,
            # 			clfs=clfs,
            # 			domain= f'classic_-_mlna',
            # 			prefix=domain,
            # 			verbose=verbose,
            # 			cwd= cwd+f'/mlna_{k}'
            # 			)
            # 		)
            # 	logic_i_p.append(logic_i_g[-1])
            # 	logic_i_pg.append(logic_i_g[-1])
            # classic - mln attribut
            # plot_features_importance_as_barh(
            # 	    data= logic_i_g[-1][1],
            # 	    getColor= custom_color,
            # 	    modelDictName= modelD,
            # 	    plotTitle= f"Classic - mln attributs features importance for",
            # 	    prefix= domain,
            # 	    cwd= cwd+f'/mlna_{k}',
            # 	    graph_a= list(mlna),
            # 	    save= True
            # 	)
            # print(f"{typeof(mlna)} {mmlna}")

            ## default + MLN - MLNa
            """
            train
            """
            MCA_GLO_MX = MlC_GLO_MX.drop(list(mlna), axis=1)

            MCA_PER_MX = MlC_PER_MX.drop(list(mlna), axis=1)
            if graphWithClass == True:
                MCA_PER_MY = MlC_PER_MY.drop(list(mlna), axis=1)
                MCA_PER_MXY = MlC_PER_MXY.drop(list(mlna), axis=1)

            MCA_GAP_MX = MlC_GAP_MX.drop(list(mlna), axis=1)
            if graphWithClass == True:
                MCA_GAP_MY = MlC_GAP_MY.drop(list(mlna), axis=1)
                MCA_GAP_MXY = MlC_GAP_MXY.drop(list(mlna), axis=1)

            """
            test
            """
            MCA_GLO_MX_T = MlC_GLO_MX_T.drop(list(mlna), axis=1)

            MCA_PER_MX_T = MlC_PER_MX_T.drop(list(mlna), axis=1)
            if graphWithClass == True:
                MCA_PER_MY_T = MlC_PER_MY_T.drop(list(mlna), axis=1)
                MCA_PER_MXY_T = MlC_PER_MXY_T.drop(list(mlna), axis=1)

            MCA_GAP_MX_T = MlC_GAP_MX_T.drop(list(mlna), axis=1)
            if graphWithClass == True:
                MCA_GAP_MY_T = MlC_GAP_MY_T.drop(list(mlna), axis=1)
                MCA_GAP_MXY_T = MlC_GAP_MXY_T.drop(list(mlna), axis=1)

            logic_i_p.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MCA_PER_MX,
                    x_testi=MCA_PER_MX_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MCA_PER_MX',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}/personalized',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            if graphWithClass == True:
                logic_i_p.append(
                    make_builder(
                        fix_imbalance=fix_imbalance,
                        DATA_OVER=None,
                        x_traini=MCA_PER_MY,
                        x_testi=MCA_PER_MY_T,
                        y_traini=y_train,
                        y_testi=y_test,
                        target_variable=target_variable,
                        clfs=clfs,
                        domain=f'MCA_PER_MY',
                        prefix=domain,
                        verbose=verbose,
                        cwd=cwd + f'/mlna_{k}/personalized',
                        withCost=withCost,
                        financialOption=financialOption
                    )
                )
                logic_i_p.append(
                    make_builder(
                        fix_imbalance=fix_imbalance,
                        DATA_OVER=None,
                        x_traini=MCA_PER_MXY,
                        x_testi=MCA_PER_MXY_T,
                        y_traini=y_train,
                        y_testi=y_test,
                        target_variable=target_variable,
                        clfs=clfs,
                        domain=f'MCA_PER_MXY',
                        prefix=domain,
                        verbose=verbose,
                        cwd=cwd + f'/mlna_{k}/personalized',
                        withCost=withCost,
                        financialOption=financialOption
                    )
                )
            logic_i_g.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MCA_GLO_MX,
                    x_testi=MCA_GLO_MX_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MCA_GLO_MX',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}/global',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            logic_i_pg.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MCA_GAP_MX,
                    x_testi=MCA_GAP_MX_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MCA_GAP_MX',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}/mixed',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            if graphWithClass == True:
                logic_i_pg.append(
                    make_builder(
                        fix_imbalance=fix_imbalance,
                        DATA_OVER=None,
                        x_traini=MCA_GAP_MY,
                        x_testi=MCA_GAP_MY_T,
                        y_traini=y_train,
                        y_testi=y_test,
                        target_variable=target_variable,
                        clfs=clfs,
                        domain=f'MCA_GAP_MY',
                        prefix=domain,
                        verbose=verbose,
                        cwd=cwd + f'/mlna_{k}/mixed',
                        withCost=withCost,
                        financialOption=financialOption
                    )
                )
                logic_i_pg.append(
                    make_builder(
                        fix_imbalance=fix_imbalance,
                        DATA_OVER=None,
                        x_traini=MCA_GAP_MXY,
                        x_testi=MCA_GAP_MXY_T,
                        y_traini=y_train,
                        y_testi=y_test,
                        target_variable=target_variable,
                        clfs=clfs,
                        domain=f'MCA_GAP_MXY',
                        prefix=domain,
                        verbose=verbose,
                        cwd=cwd + f'/mlna_{k}/mixed',
                        withCost=withCost,
                        financialOption=financialOption
                    )
                )
            # classic + mln - mln attribut
            # plot_features_importance_as_barh(
            # 	    data= logic_i_p[-1][1],
            # 	    getColor= custom_color,
            # 	    modelDictName= modelD,
            # 	    plotTitle= f"Classic + mln - mln attributs features importance for",
            # 	    prefix= domain,
            # 	    cwd= cwd+f'/mlna_{k}/personalized',
            # 	    graph_a= list(mlna),
            # 	    save= True
            # 	)
            # plot_features_importance_as_barh(
            # 	    data= logic_i_g[-1][1],
            # 	    getColor= custom_color,
            # 	    modelDictName= modelD,
            # 	    plotTitle= f"Classic + mln - mln attributs features importance for",
            # 	    prefix= domain,
            # 	    cwd= cwd+f'/mlna_{k}/global',
            # 	    graph_a= list(mlna),
            # 	    save= True
            # 	)
            # plot_features_importance_as_barh(
            # 	    data= logic_i_pg[-1][1],
            # 	    getColor= custom_color,
            # 	    modelDictName= modelD,
            # 	    plotTitle= f"Classic + mln - mln attributs features importance for",
            # 	    prefix= domain,
            # 	    cwd= cwd+f'/mlna_{k}/mixed',
            # 	    graph_a= list(mlna),
            # 	    save= True
            # 	)

            # print html report for analysis of variable i
            # logic_i=[default,*logic_i]
            logic_p = [*logic_p, *logic_i_p]
            logic_g = [*logic_g, *logic_i_g]
            logic_pg = [*logic_pg, *logic_i_pg]
            table_g = print_summary([default, *logic_i_g], modelD)
            table_p = print_summary([default, *logic_i_p], modelD)
            table_pg = print_summary([default, *logic_i_pg], modelD)
            # del logic_p
            # del logic_g
            create_file(
                content=table_g[1],
                cwd=cwd + f'/mlna_{k}/global',
                prefix=domain,
                filename=f"mlna_for_{case_k}",
                extension=".html"
            )
            create_file(
                content=table_p[1],
                cwd=cwd + f'/mlna_{k}/personalized',
                prefix=domain,
                filename=f"mlna_for_{case_k}",
                extension=".html"
            )
            create_file(
                content=table_pg[1],
                cwd=cwd + f'/mlna_{k}/mixed',
                prefix=domain,
                filename=f"mlna_for_{case_k}",
                extension=".html"
            )
            table_g = None
            table_p = None
            logic_i_g = None
            logic_i_p = None
            VALUE_MLN_G = None
            VALUE_MLN_P = None
            VALUE_MLN_MINUS_MLN_P = None
            VALUE_MLN_MINUS_MLN_G = None
            VALUE_MINUS_MLNa = None
        # del table_g
        # del table_p
        # del logic_i_g
        # del logic_i_p
        # del VALUE_MLN_G
        # del VALUE_MLN_P
        # del VALUE_MLN_MINUS_MLN_P
        # del VALUE_MLN_MINUS_MLN_G
        # del VALUE_MINUS_MLNa
        table_p = print_summary([default, *logic_p], modelD)
        table_g = print_summary([default, *logic_g], modelD)
        table_pg = print_summary([default, *logic_pg], modelD)
        create_file(
            content=table_p[1],
            cwd=cwd + f'/mlna_{k}/personalized',
            prefix=domain,
            filename=f"mlna_for_all_{k}_combination_categorial_data",
            extension=".html"
        )
        create_file(
            content=table_g[1],
            cwd=cwd + f'/mlna_{k}/global',
            prefix=domain,
            filename=f"mlna_for_all_{k}_combination_categorial_data",
            extension=".html"
        )
        create_file(
            content=table_pg[1],
            cwd=cwd + f'/mlna_{k}/mixed',
            prefix=domain,
            filename=f"mlna_for_all_{k}_combination_categorial_data",
            extension=".html"
        )
        modelD = None
        PERSONS = None
        table_p = None
        table_g = None

# return logic
def make_mlna_top_k_variable(
        default,
        x_traini,
        x_testi,
        y_traini,
        y_testi,
        OHE,
        nominal_factor_colums,
        cwd,
        domain,
        fix_imbalance,
        target_variable,
        custom_color,
        modelD,
        verbose,
        clfs,
        alpha,
        withCost,
        financialOption,
        graphWithClass=False,
        topR=[]
):
    """
    """

    # local eval storage
    logic_g = []
    logic_p = []
    logic_pg = []
    ## visualization of result
    modelD = model_desc()
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)

    for k in range(2, len(topR) + 1):  # for 1<k<|OHE[i]|+2
        # for k in [2]: # for 1<k<|OHE[i]|+2
        # for k in range(2:len(OHE)+1: # for 1<k<|OHE[i]|+2
        layer_config = topR[:k]  # create subsets of k index of OHE and fetch it
        print(layer_config)
        # for layer_config in [[1,2]]: # create subsets of k index of OHE and fetch it
        # logic storage
        logic_i_g = []
        logic_i_p = []
        logic_i_pg = []
        copT = x_train.copy(deep=True)
        copT[target_variable] = y_train.copy(deep=True)
        # build the MLN for the variable i
        MLN = build_mlg_with_class(copT, [OHE[i] for i in layer_config],
                                   target_variable) if (graphWithClass is True) else build_mlg(
            copT, [OHE[i] for i in layer_config])
        col_targeted = [f'{nominal_factor_colums[i]}' for i in layer_config]
        case_k = '_'.join(col_targeted)
        # save the graph
        save_graph(
            cwd=cwd + f'/mlna_{k}_b',
            graph=MLN,
            name=f'{case_k}_mln',
            rows_len=copT.shape[0],
            prefix=domain,
            cols_len=len(OHE)
        )

        # Binome page rank
        bipart_combine = nx.pagerank(MLN, alpha=alpha)

        # get intra page rank
        bipart_intra_pagerank = nx.pagerank(
            MLN,
            personalization=compute_personlization(get_intra_node_label(MLN), MLN),
            alpha=alpha
        )

        # get inter page rank
        bipart_inter_pagerank = nx.pagerank(
            MLN,
            personalization=compute_personlization(get_inter_node_label(MLN), MLN),
            alpha=alpha
        )

        # extract centrality degree from MLN
        extracts_g = {}
        extracts_p = {}

        extracts_g[f'Att_DEGREE_GLO'] = [
            get_number_of_borrowers_with_same_custom_layer_value(borrower=index, graph=MLN, custom_layer=list(range(k)))[
                1] for index in PERSONS
        ]
        extracts_p[f'Att_DEGREE_PER'] = extracts_g[f'Att_DEGREE_GLO']
        extracts_g[f'Att_INTRA_GLO'] = get_max_borrower_pr(bipart_intra_pagerank)[0]
        extracts_g[f'Att_INTER_GLO'] = get_max_borrower_pr(bipart_inter_pagerank)[0]
        extracts_g[f'Att_COMBINE_GLO'] = get_max_borrower_pr(bipart_combine)[0]
        extracts_p[f'Att_COMBINE_PER'] = [get_max_borrower_pr(nx.pagerank(MLN,
                                                                          personalization=compute_personlization(
                                                                              get_combine_perso_nodes_label(
                                                                                  MLN, [val], k)[
                                                                                  0][0], MLN),
                                                                          alpha=alpha))[1][val] for
                                          index, val in enumerate(PERSONS)]
        if graphWithClass == True:
            extracts_p[f'YN_COMBINE_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                         personalization=compute_personlization(
                                             get_combine_perso_nodes_label(
                                                 removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[
                                                 0][0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                         alpha=alpha))[0] for
                index, val in enumerate(PERSONS)]
            extracts_p[f'YP_COMBINE_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                         personalization=compute_personlization(
                                             get_combine_perso_nodes_label(
                                                 removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[
                                                 0][0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                         alpha=alpha))[1] for
                index, val in enumerate(PERSONS)]

        extracts_p[f'Att_INTER_PER'] = [
            get_max_borrower_pr(nx.pagerank(MLN,
                                            personalization=compute_personlization(
                                                get_inter_perso_nodes_label(
                                                    MLN, [val], k)[0][
                                                    0], MLN),
                                            alpha=alpha))[1][val] for
            index, val in enumerate(PERSONS)]
        if graphWithClass == True:
            extracts_p[f'YN_INTER_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                         personalization=compute_personlization(
                                             get_inter_perso_nodes_label(
                                                 removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[0][
                                                 0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                         alpha=alpha))[0] for
                index, val in enumerate(PERSONS)]
            extracts_p[f'YP_INTER_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                         personalization=compute_personlization(
                                             get_inter_perso_nodes_label(
                                                 removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[0][
                                                 0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                         alpha=alpha))[1] for
                index, val in enumerate(PERSONS)]

        extracts_p[f'Att_INTRA_PER'] = [
            get_max_borrower_pr(nx.pagerank(MLN,
                                            personalization=compute_personlization(
                                                get_intra_perso_nodes_label(
                                                    MLN, [val], k)[0][
                                                    0], MLN),
                                            alpha=alpha))[1][val] for
            index, val in enumerate(PERSONS)]
        if graphWithClass == True:
            extracts_p[f'YN_INTRA_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                         personalization=compute_personlization(
                                             get_intra_perso_nodes_label(
                                                 removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[0][
                                                 0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                         alpha=alpha))[0] for
                index, val in enumerate(PERSONS)]
            extracts_p[f'YP_INTRA_PER'] = [
                get_class_pr(nx.pagerank(removeEdge(MLN, k, copT.loc[val, target_variable], val),
                                         personalization=compute_personlization(
                                             get_intra_perso_nodes_label(
                                                 removeEdge(MLN, k, copT.loc[val, target_variable], val), [val], k)[0][
                                                 0], removeEdge(MLN, k, copT.loc[val, target_variable], val)),
                                         alpha=alpha))[1] for
                index, val in enumerate(PERSONS)]

        extracts_g[f'Att_M_INTRA_GLO'] = [
            get_max_modality_pagerank_score(index, MLN, k, bipart_intra_pagerank) for index in PERSONS]
        extracts_g[f'Att_M_INTER_GLO'] = [
            get_max_modality_pagerank_score(index, MLN, k, bipart_inter_pagerank) for index in PERSONS]
        extracts_g[f'Att_M_COMBINE_GLO'] = [
            get_max_modality_pagerank_score(index, MLN, k, bipart_combine) for index in PERSONS]
        extracts_p[f'Att_M_COMBINE_PER'] = [get_max_modality_pagerank_score(val, MLN, k,
                                                                            nx.pagerank(MLN,
                                                                                        personalization=compute_personlization(
                                                                                            get_combine_perso_nodes_label(
                                                                                                MLN,
                                                                                                [
                                                                                                    val],
                                                                                                k)[
                                                                                                0][
                                                                                                0],
                                                                                            MLN),
                                                                                        alpha=alpha))
                                            for index, val in enumerate(PERSONS)]
        extracts_p[f'Att_M_INTER_PER'] = [get_max_modality_pagerank_score(val, MLN, k,
                                                                          nx.pagerank(MLN,
                                                                                      personalization=compute_personlization(
                                                                                          get_inter_perso_nodes_label(
                                                                                              MLN,
                                                                                              [
                                                                                                  val],
                                                                                              k)[
                                                                                              0][
                                                                                              0],
                                                                                          MLN),
                                                                                      alpha=alpha))
                                          for index, val in enumerate(PERSONS)]
        extracts_p[f'Att_M_INTRA_PER'] = [get_max_modality_pagerank_score(val, MLN, k,
                                                                          nx.pagerank(MLN,
                                                                                      personalization=compute_personlization(
                                                                                          get_intra_perso_nodes_label(
                                                                                              MLN,
                                                                                              [
                                                                                                  val],
                                                                                              k)[
                                                                                              0][
                                                                                              0],
                                                                                          MLN),
                                                                                      alpha=alpha))
                                          for index, val in enumerate(PERSONS)]
        # standardization
        # print(f"""
        #     {extracts_p}
        #     {extracts_g}
        #     """)
        standard_extraction(extracts_p, extracts_p.keys())
        standard_extraction(extracts_g, extracts_g.keys())
        extracts_g_t, extracts_p_t = get_test_examples_descriptors_from_graph_model(
            MLN=MLN,
            x_test=x_test,
            graphWithClass=graphWithClass,
            OHE=OHE,
            i=k,
            alpha=alpha,
            layers=k,
            layer_config=layer_config,
        )
        # save descriptors
        """
        train
        """
        extract_MX_GLO_df = pd.DataFrame(extracts_g)
        extract_MX_PER_df = pd.DataFrame({key: extracts_p[key] for key in extracts_p.keys() if not ("Y" in key)})
        if graphWithClass == True:
            extract_MY_PER_df = pd.DataFrame({key: extracts_p[key] for key in extracts_p.keys() if ("Y" in key)})
            extract_MXY_PER_df = pd.DataFrame(extracts_p)

        """
        test
        """
        extract_MX_GLO_df_t = pd.DataFrame(extracts_g_t)
        extract_MX_PER_df_t = pd.DataFrame(
            {key: extracts_p_t[key] for key in extracts_p_t.keys() if not ("Y" in key)})
        if graphWithClass == True:
            extract_MY_PER_df_t = pd.DataFrame(
                {key: extracts_p_t[key] for key in extracts_p_t.keys() if ("Y" in key)})
            extract_MXY_PER_df_t = pd.DataFrame(extracts_p_t)

        save_dataset(
            cwd=cwd + f'/mlna_{k}_b/global',
            dataframe=extract_MX_GLO_df,
            name=f'{domain}_extract_MX_GLO_df_for_{case_k}',
            prefix=domain,
            sep=','
        )
        save_dataset(
            cwd=cwd + f'/mlna_{k}_b/personalized',
            dataframe=extract_MX_PER_df,
            name=f'{domain}_extract_MX_PER_df_for_{case_k}',
            prefix=domain,
            sep=','
        )
        save_dataset(
            cwd=cwd + f'/mlna_{k}_b/global',
            dataframe=extract_MX_GLO_df_t,
            name=f'{domain}_extract_MX_GLO_df_t_for_{case_k}',
            prefix=domain,
            sep=','
        )
        save_dataset(
            cwd=cwd + f'/mlna_{k}_b/personalized',
            dataframe=extract_MX_PER_df_t,
            name=f'{domain}_extract_MX_PER_df_t_for_{case_k}',
            prefix=domain,
            sep=','
        )
        if graphWithClass == True:
            save_dataset(
                cwd=cwd + f'/mlna_{k}_b/personalized',
                dataframe=extract_MY_PER_df,
                name=f'{domain}_extract_MY_PER_df_for_{case_k}',
                prefix=domain,
                sep=','
            )
            save_dataset(
                cwd=cwd + f'/mlna_{k}_b/personalized',
                dataframe=extract_MXY_PER_df,
                name=f'{domain}_extract_MXY_PER_df_for_{case_k}',
                prefix=domain,
                sep=','
            )
            save_dataset(
                cwd=cwd + f'/mlna_{k}_b/personalized',
                dataframe=extract_MY_PER_df_t,
                name=f'{domain}_extract_MY_PER_df_t_for_{case_k}',
                prefix=domain,
                sep=','
            )
            save_dataset(
                cwd=cwd + f'/mlna_{k}_b/personalized',
                dataframe=extract_MXY_PER_df_t,
                name=f'{domain}_extract_MXY_PER_df_t_for_{case_k}',
                prefix=domain,
                sep=','
            )
        extract_g_df = None
        extract_p_df = None
        MLN = None
        bipart_combine = None
        bipart_intra_pagerank = None
        bipart_inter_pagerank = None

        # del extract_g_df
        # del extract_p_df
        # del MLN
        # del bipart_combine
        # del bipart_intra_pagerank
        # del bipart_inter_pagerank

        mlna = set()
        col_list = x_train.columns.to_list()
        for i in layer_config:
            for column in OHE[i]:
                if column in col_list:
                    mlna.add(column)
                else:
                    mlna.add(column.split("__")[-1])
        # classic
        # plot_features_importance_as_barh(
        # 	    data= default[1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= "Classic features importance",
        # 	    prefix= domain,
        # 	    cwd= cwd+f'/mlna_{k}_b',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)
        ## default + MLN
        # inject descriptors
        """
        train
        """
        MlC_GLO_MX = inject_features_extracted(x_train, extract_MX_GLO_df)

        MlC_PER_MX = inject_features_extracted(x_train, extract_MX_PER_df)
        if graphWithClass == True:
            MlC_PER_MY = inject_features_extracted(x_train, extract_MY_PER_df)
            MlC_PER_MXY = inject_features_extracted(x_train, extract_MXY_PER_df)

        MlC_GAP_MX = inject_features_extracted(MlC_GLO_MX, extract_MX_PER_df)
        if graphWithClass == True:
            MlC_GAP_MY = inject_features_extracted(MlC_GLO_MX, extract_MY_PER_df)
            MlC_GAP_MXY = inject_features_extracted(MlC_GLO_MX, extract_MXY_PER_df)

        """
        test
        """
        MlC_GLO_MX_T = inject_features_extracted(x_test, extract_MX_GLO_df_t)

        MlC_PER_MX_T = inject_features_extracted(x_test, extract_MX_PER_df_t)
        if graphWithClass == True:
            MlC_PER_MY_T = inject_features_extracted(x_test, extract_MY_PER_df_t)
            MlC_PER_MXY_T = inject_features_extracted(x_test, extract_MXY_PER_df_t)

        MlC_GAP_MX_T = inject_features_extracted(MlC_GLO_MX_T, extract_MX_PER_df_t)
        if graphWithClass == True:
            MlC_GAP_MY_T = inject_features_extracted(MlC_GLO_MX_T, extract_MY_PER_df_t)
            MlC_GAP_MXY_T = inject_features_extracted(MlC_GLO_MX_T, extract_MXY_PER_df_t)
        extracts_g = None
        extracts_p = None
        extracts_g_t = None
        extracts_p_t = None
        # del extracts_g
        # del extracts_p
        # print(MlC_PER_MX.isnull().sum())
        # print(MlC_PER_MX_T.isnull().sum())
        logic_i_p.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=MlC_PER_MX,
                x_testi=MlC_PER_MX_T,
                y_traini=y_train,
                y_testi=y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MlC_PER_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'/mlna_{k}_b/personalized',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        if graphWithClass == True:
            logic_i_p.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MlC_PER_MY,
                    x_testi=MlC_PER_MY_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MlC_PER_MY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}_b/personalized',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            logic_i_p.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MlC_PER_MXY,
                    x_testi=MlC_PER_MXY_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MlC_PER_MXY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}_b/personalized',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
        # print(f"""
        #     {MlC_GLO_MX.columns}
        #     {len(MlC_GLO_MX.columns)}
        #     {MlC_GLO_MX_T.columns}
        #     {len(MlC_GLO_MX_T.columns)}
        #     """)
        logic_i_g.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=MlC_GLO_MX,
                x_testi=MlC_GLO_MX_T,
                y_traini=y_train,
                y_testi=y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MlC_GLO_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'/mlna_{k}_b/global',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        logic_i_pg.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=MlC_GAP_MX,
                x_testi=MlC_GAP_MX_T,
                y_traini=y_train,
                y_testi=y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MlC_GAP_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'/mlna_{k}_b/mixed',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        if graphWithClass == True:
            logic_i_pg.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MlC_GAP_MY,
                    x_testi=MlC_GAP_MY_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MlC_GAP_MY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}_b/mixed',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            logic_i_pg.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MlC_GAP_MXY,
                    x_testi=MlC_GAP_MXY_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MlC_GAP_MXY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}_b/mixed',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
        # classic + mln
        # plot_features_importance_as_barh(
        # 	    data= logic_i_g[0][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+f'/mlna_{k}_b/global',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)
        # plot_features_importance_as_barh(
        # 	    data= logic_i_p[0][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+f'/mlna_{k}_b/personalized',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)
        # plot_features_importance_as_barh(
        # 	    data= logic_i_pg[0][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+f'/mlna_{k}_b/mixed',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)

        ## default - MLNa

        # is_full = list(set(col_list) - set(list(mlna)))
        # print(f"{col_list} + {mlna} + {is_full}")
        # if len(is_full) > 1:  # check if operation is possible
        # 	# print(typeof(mlna))
        #
        # 	VALUE_MINUS_MLNa = value_clfs.drop(list(mlna), axis=1)
        #
        # 	logic_i_g.append(
        # 		make_builder(
        # 			fix_imbalance=fix_imbalance,
        # 			DATA_OVER=VALUE_MINUS_MLNa,
        # 			target_variable=target_variable,
        # 			clfs=clfs,
        # 			domain=f'classic_-_mlna',
        # 			prefix=domain,
        # 			verbose=verbose,
        # 			cwd=cwd + f'/mlna_{k}_b'
        # 		)
        # 	)
        # 	logic_i_p.append(logic_i_g[-1])
        # 	logic_i_pg.append(logic_i_g[-1])
        # classic - mln attribut
        # plot_features_importance_as_barh(
        # 	    data= logic_i_g[-1][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic - mln attributs features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+f'/mlna_{k}_b',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)
        # print(f"{typeof(mlna)} {mmlna}")

        ## default + MLN - MLNa
        """
        train
        """
        MCA_GLO_MX = MlC_GLO_MX.drop(list(mlna), axis=1)

        MCA_PER_MX = MlC_PER_MX.drop(list(mlna), axis=1)
        if graphWithClass == True:
            MCA_PER_MY = MlC_PER_MY.drop(list(mlna), axis=1)
            MCA_PER_MXY = MlC_PER_MXY.drop(list(mlna), axis=1)

        MCA_GAP_MX = MlC_GAP_MX.drop(list(mlna), axis=1)
        if graphWithClass == True:
            MCA_GAP_MY = MlC_GAP_MY.drop(list(mlna), axis=1)
            MCA_GAP_MXY = MlC_GAP_MXY.drop(list(mlna), axis=1)

        """
        test
        """
        MCA_GLO_MX_T = MlC_GLO_MX_T.drop(list(mlna), axis=1)

        MCA_PER_MX_T = MlC_PER_MX_T.drop(list(mlna), axis=1)
        if graphWithClass == True:
            MCA_PER_MY_T = MlC_PER_MY_T.drop(list(mlna), axis=1)
            MCA_PER_MXY_T = MlC_PER_MXY_T.drop(list(mlna), axis=1)

        MCA_GAP_MX_T = MlC_GAP_MX_T.drop(list(mlna), axis=1)
        if graphWithClass == True:
            MCA_GAP_MY_T = MlC_GAP_MY_T.drop(list(mlna), axis=1)
            MCA_GAP_MXY_T = MlC_GAP_MXY_T.drop(list(mlna), axis=1)

        logic_i_p.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=MCA_PER_MX,
                x_testi=MCA_PER_MX_T,
                y_traini=y_train,
                y_testi=y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MCA_PER_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'/mlna_{k}_b/personalized',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        if graphWithClass == True:
            logic_i_p.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MCA_PER_MY,
                    x_testi=MCA_PER_MY_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MCA_PER_MY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}_b/personalized',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            logic_i_p.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MCA_PER_MXY,
                    x_testi=MCA_PER_MXY_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MCA_PER_MXY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}_b/personalized',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
        logic_i_g.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=MCA_GLO_MX,
                x_testi=MCA_GLO_MX_T,
                y_traini=y_train,
                y_testi=y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MCA_GLO_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'/mlna_{k}_b/global',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        logic_i_pg.append(
            make_builder(
                fix_imbalance=fix_imbalance,
                DATA_OVER=None,
                x_traini=MCA_GAP_MX,
                x_testi=MCA_GAP_MX_T,
                y_traini=y_train,
                y_testi=y_test,
                target_variable=target_variable,
                clfs=clfs,
                domain=f'MCA_GAP_MX',
                prefix=domain,
                verbose=verbose,
                cwd=cwd + f'/mlna_{k}_b/mixed',
                withCost=withCost,
                financialOption=financialOption
            )
        )
        if graphWithClass == True:
            logic_i_pg.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MCA_GAP_MY,
                    x_testi=MCA_GAP_MY_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MCA_GAP_MY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}_b/mixed',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
            logic_i_pg.append(
                make_builder(
                    fix_imbalance=fix_imbalance,
                    DATA_OVER=None,
                    x_traini=MCA_GAP_MXY,
                    x_testi=MCA_GAP_MXY_T,
                    y_traini=y_train,
                    y_testi=y_test,
                    target_variable=target_variable,
                    clfs=clfs,
                    domain=f'MCA_GAP_MXY',
                    prefix=domain,
                    verbose=verbose,
                    cwd=cwd + f'/mlna_{k}_b/mixed',
                    withCost=withCost,
                    financialOption=financialOption
                )
            )
        # classic + mln - mln attribut
        # plot_features_importance_as_barh(
        # 	    data= logic_i_p[-1][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln - mln attributs features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+f'/mlna_{k}_b/personalized',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)
        # plot_features_importance_as_barh(
        # 	    data= logic_i_g[-1][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln - mln attributs features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+f'/mlna_{k}_b/global',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)
        # plot_features_importance_as_barh(
        # 	    data= logic_i_pg[-1][1],
        # 	    getColor= custom_color,
        # 	    modelDictName= modelD,
        # 	    plotTitle= f"Classic + mln - mln attributs features importance for",
        # 	    prefix= domain,
        # 	    cwd= cwd+f'/mlna_{k}_b/mixed',
        # 	    graph_a= list(mlna),
        # 	    save= True
        # 	)

        # print html report for analysis of variable i
        # logic_i=[default,*logic_i]
        logic_p = [*logic_p, *logic_i_p]
        logic_g = [*logic_g, *logic_i_g]
        logic_pg = [*logic_pg, *logic_i_pg]
        table_g = print_summary([default, *logic_i_g], modelD)
        table_p = print_summary([default, *logic_i_p], modelD)
        table_pg = print_summary([default, *logic_i_pg], modelD)
        # del logic_p
        # del logic_g
        create_file(
            content=table_g[1],
            cwd=cwd + f'/mlna_{k}_b/global',
            prefix=domain,
            filename=f"mlna_for_{case_k}",
            extension=".html"
        )
        create_file(
            content=table_p[1],
            cwd=cwd + f'/mlna_{k}_b/personalized',
            prefix=domain,
            filename=f"mlna_for_{case_k}",
            extension=".html"
        )
        create_file(
            content=table_pg[1],
            cwd=cwd + f'/mlna_{k}_b/mixed',
            prefix=domain,
            filename=f"mlna_for_{case_k}",
            extension=".html"
        )
        table_g = None
        table_p = None
        logic_i_g = None
        logic_i_p = None
        VALUE_MLN_G = None
        VALUE_MLN_P = None
        VALUE_MLN_MINUS_MLN_P = None
        VALUE_MLN_MINUS_MLN_G = None
        VALUE_MINUS_MLNa = None
    # del table_g
    # del table_p
    # del logic_i_g
    # del logic_i_p
    # del VALUE_MLN_G
    # del VALUE_MLN_P
    # del VALUE_MLN_MINUS_MLN_P
    # del VALUE_MLN_MINUS_MLN_G
    # del VALUE_MINUS_MLNa
    table_p = print_summary([default, *logic_p], modelD)
    table_g = print_summary([default, *logic_g], modelD)
    table_pg = print_summary([default, *logic_pg], modelD)
    create_file(
        content=table_p[1],
        cwd=cwd + f'/mlna_{k}_b/personalized',
        prefix=domain,
        filename=f"mlna_for_all_{k}_combination_categorial_data",
        extension=".html"
    )
    create_file(
        content=table_g[1],
        cwd=cwd + f'/mlna_{k}_b/global',
        prefix=domain,
        filename=f"mlna_for_all_{k}_combination_categorial_data",
        extension=".html"
    )
    create_file(
        content=table_pg[1],
        cwd=cwd + f'/mlna_{k}_b/mixed',
        prefix=domain,
        filename=f"mlna_for_all_{k}_combination_categorial_data",
        extension=".html"
    )
    modelD = None
    PERSONS = None
    table_p = None
    table_g = None


# return logic
# @profile
def make_builder(
        fix_imbalance,
        target_variable,
        clfs,
        cwd,
        prefix,
        verbose,
        duration_divider,
        rate_divider,
        financialOption,
        original,
        domain='classic',
        DATA_OVER=None,
        withCost=True,
        x_traini=None,
        x_testi=None,
        y_traini=None,
        y_testi=None
):
    """
    build ML models
    Parameters
    ----------
    fix_imbalance
    target_variable
    clfs
    cwd
    prefix
    verbose
    financialOption
    domain
    DATA_OVER
    withCost
    x_traini
    x_testi
    y_traini
    y_testi

    Returns
    -------

    """

    # test train split
    if DATA_OVER is not None:
        x_train, x_test, y_train, y_test = test_train(DATA_OVER, target_variable)
    else:
        x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    # fix inbalance state of classes
    if fix_imbalance:
        x_train, y_train = get_SMOTHE_dataset(
            X=x_train,
            y=y_train,
            random_state=42,
            # sampling_strategy= "minority"
        )

    # get evaluation storage structure
    STORE_STD = init_training_store(x_train, withCost=withCost)
    # print(original)
    # run classic ML on default data
    store = train(
        clfs=clfs,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        store=STORE_STD,
        domain=domain,
        prefix=prefix,
        cwd=cwd,
        duration_divider=duration_divider,
        rate_divider=rate_divider,
        withCost=withCost,
        financialOption=financialOption,
        original = original
    )
    link_to_original = save_dataset(
        cwd=cwd,
        dataframe=store,
        name=f'{domain}_metric',
        prefix=prefix,
        sep=','
    )
    save_dataset(
        cwd=cwd,
        dataframe=x_train,
        name=f'{domain}_x_train',
        prefix=prefix,
        sep=','
    )
    save_dataset(
        cwd=cwd,
        dataframe=x_test,
        name=f'{domain}_x_test',
        prefix=prefix,
        sep=','
    )
    save_dataset(
        cwd=cwd,
        dataframe=y_train,
        name=f'{domain}_y_train',
        prefix=prefix,
        sep=','
    )
    save_dataset(
        cwd=cwd,
        dataframe=y_test,
        name=f'{domain}_y_test',
        prefix=prefix,
        sep=','
    )
    return (domain, store)


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

# @profile
def mlnaPipeline(
        cwd,
        domain,
        dataset_link,
        target_variable,
        duration_divider=None,
        rate_divider=None,
        all_nominal=True,
        all_numeric=False,
        verbose=True,
        fix_imbalance=True,
        levels=None,
        to_remove=None,
        encoding="utf-8",
        index_col=None,
        na_values=None,
        alphas=[0.85],
        portion=.65,
        graphWithClass=True,
        financialOption=None,
        dataset_delimiter=',',
        withCost=True
):
    """Run a sequence of action in goal to build, analyse, extract descriptors and evaluate with just one called

    Args:
        domain:
        dataset_link:
        dataset_delimiter:
        all_nominal:
        all_numeric:
        verbose:
        levels: allowed
            - 0 eda
            - 1 preprocessing
            - 2 mln 1 variable in nominal ones
            - 3 mln k variables in nominal ones
            - 4 mln 1 variable in numeric ones
            - 5 mln k variables in numeric ones

    Returns
        True if all instructions succeeds
    """

    print(f"Current working directory: {cwd}") if verbose else None

    # load the dedicated work dataset
    print(f"received path: {dataset_link}") if verbose else None
    dataset = load_data_set_from_url(path=dataset_link, sep=dataset_delimiter, encoding=encoding, index_col=index_col,
                                     na_values=na_values)
    dataset.reset_index(drop=True, inplace=True)
    print(f"loaded dataset dim: {dataset.shape}") if verbose else None

    # eda
    # print(dataset.shape)
    dataset = random_sample_merge_v2(df=dataset, target_column=target_variable, percentage=portion)
    dataset.reset_index(drop=True, inplace=True)

    print(f"loaded dataset dim: {dataset.shape}") if verbose else None
    # print(dataset.shape)
    # dataset = make_eda(dataframe=dataset, verbose=verbose)
    dataset = make_eda(dataframe=dataset, verbose=verbose)
    dataset_copy = dataset.copy(deep=True)
    # print(dataset.shape)
    # preprocessing
    # (
    # col_list,
    # numeric_col,
    # categorial_col,
    # ordinal_factor_colums,
    # nominal_factor_colums,
    # numeric_with_outliers_columns,
    # numeric_uniform_colums,
    # DATA_OHE_LB_LBU_STDU_STDWO,
    # OHE,
    # DATA_DISCRETIZE_OHE_2,
    # OHE_2
    # )
    if sum(['preprocessing' in file for _, _, files in
            os.walk(cwd + f'/outputs/{domain}/mlna_preprocessing/model_storage') for
            file in files]) == 0:
        promise = make_preprocessing(
            dataset=dataset,
            to_remove=to_remove,
            domain=domain,
            cwd=cwd + f"/outputs/{domain}",
            target_variable=target_variable,
            verbose=verbose,
            levels=levels
        )
        # split data in training and test lot
        x_train, x_test, y_train, y_test = test_train(dataframe=promise[7], target=target_variable, reset_index=False)
        x1_train = dataset_copy.loc[list(x_train.index)]
        x1_test = dataset_copy.loc[list(x_test.index)]
        save_model(
            cwd=cwd + f'/outputs/{domain}/mlna_preprocessing',
            clf=(promise, x_train, x_test, y_train, y_test,x1_train, x1_test),
            prefix="",
            clf_name="preprocessing"
        )
    else:
        name = \
            [file for _, _, files in
             os.walk(cwd + f'/outputs/{domain}/mlna_preprocessing/model_storage')
             for file in files if
             'preprocessing' in file][0]
        (promise, x_train, x_test, y_train, y_test,x1_train, x1_test) = read_model(
            cwd + f'/outputs/{domain}/mlna_preprocessing/model_storage/{name}')
        print('loaded preprocessing model', len(promise))
    # print(promise[7].shape)
    # get dict of models
    clfs = init_models()

    ## visualization of result
    modelD = model_desc()

    # fetch each type of pretreated data to apply MLNA and ML
    # for (key,value) in _ALL_PRETRAETED_DATA.items():
    # 	if 'Factors' in key:
    # value_1 = DATA_DISCRETIZE_OHE_2.sample(50)
    # value_2 = DATA_OHE_LB_LBU_STDU_STDWO.sample(50)
    # case where with triggers just categorials data to perform our MLNA

    # build divers logic of MLNA
    # 1) build an MLN on just one variable
    # global logic
    # global_logic = []
    # start with model building and importance analysis
    # exit()
    ## default
    if sum(['classic_metric' in file for _, _, files in os.walk(f"{cwd}/outputs/{domain}/data_selection_storage") for
            file in files]) == 0:
        default = make_builder(
            fix_imbalance=fix_imbalance,
            # DATA_OVER=promise[7],
            target_variable=target_variable,
            clfs=clfs,
            domain='classic',
            prefix=domain,
            verbose=verbose,
            duration_divider=duration_divider ,
            rate_divider=rate_divider,
            cwd=cwd + f"/outputs/{domain}",
            withCost=withCost,
            financialOption=financialOption,
            x_traini=x_train,
            x_testi=x_test,
            y_traini=y_train,
            y_testi=y_test,
            original = (x1_train, x1_test)
        )
    else:
        name = [file for _, _, files in os.walk(f"{cwd}/outputs/{domain}/data_selection_storage") for file in files if
                'classic_metric' in file][0]
        default = (
            'classic',
            load_data_set_from_url(
                path=f"{cwd}/outputs/{domain}/data_selection_storage/{name}"
                , sep=dataset_delimiter
                , encoding=encoding
                , index_col=0
                , na_values=na_values
            )
        )
        print(default)
        print("load classic training")

    for alpha in alphas:
        if (2 in levels) and (len(promise[8]) > 0):  # if this stage is allowed

            if sum(['_best_features' in file for _, _, files in
                    os.walk(
                        cwd + f'/outputs/{domain}{"/withClass" * int(graphWithClass)}/{alpha}/qualitative/model_storage')
                    for file in files]) == 0:
                (outperformers, NbGood) = make_mlna_1_variable_v2(
                    default=default,
                    x_traini=x_train,
                    x_testi=x_test,
                    y_traini=y_train,
                    y_testi=y_test,
                    OHE=promise[8],
                    duration_divider=duration_divider ,
                    rate_divider=rate_divider,
                    nominal_factor_colums=promise[2],
                    cwd=cwd + f'/outputs/{domain}{"/withClass" * int(graphWithClass)}/{alpha}/qualitative',
                    domain=domain,
                    fix_imbalance=fix_imbalance,
                    target_variable=target_variable,
                    custom_color=custom_color,
                    modelD=modelD,
                    verbose=verbose,
                    clfs=clfs,
                    alpha=alpha,
                    graphWithClass=graphWithClass,
                    withCost=withCost,
                    financialOption=financialOption,
                    original = (x1_train, x1_test)
                )
                outperformers = dict(sorted(outperformers.items(), key=lambda x: x[1], reverse=True))
                bestK = bestThreshold(list(outperformers)) + 1 if len(outperformers) > 2 else len(outperformers)
                print(f"{outperformers}, {NbGood} Goods and the best top k is {bestK}")
                save_model(
                    cwd=cwd + f'/outputs/{domain}{"/withClass" * int(graphWithClass)}/{alpha}/qualitative',
                    clf={
                        'model': outperformers,
                        'nbGood': NbGood,
                        'bestK': bestK,
                        "name": [promise[2][i] for i in list(outperformers.keys())],
                    },
                    prefix="feautures",
                    clf_name=f"{domain}_best_features"
                )
                contenu = f"""
                'model': {outperformers},
                'nbGood': {NbGood},
                'bestK': {bestK},
                "name":{[promise[2][i] for i in list(outperformers.keys())]},
                "BName":{[promise[2][i] for i in list(outperformers.keys())[:bestK]]},
                """
                with open(
                        cwd + f'/outputs/{domain}{"/withClass" * int(graphWithClass)}/{alpha}/qualitative/bestDetails',
                        "a") as fichier:
                    fichier.write(contenu)
                # exit()
            else:
                name = \
                    [file for _, _, files in os.walk(
                        cwd + f'/outputs/{domain}{"/withClass" * int(graphWithClass)}/{alpha}/qualitative/model_storage')
                     for file in files if
                     '_best_features' in file][0]
                backup = read_model(
                    cwd + f'/outputs/{domain}{"/withClass" * int(graphWithClass)}/{alpha}/qualitative/model_storage/{name}')
                outperformers = backup["model"]
                bestK = backup["bestK"]

            print(list(outperformers.keys())[:bestK])
            # exit()
            if len(list(outperformers.keys())) > 1:
                make_mlna_top_k_variable_v2(
                    default=default,
                    x_traini=x_train,
                    x_testi=x_test,
                    y_traini=y_train,
                    y_testi=y_test,
                    OHE=promise[8],
                    duration_divider=duration_divider,
                    rate_divider=rate_divider,
                    nominal_factor_colums=promise[2],
                    cwd=cwd + f'/outputs/{domain}{"/withClass" * int(graphWithClass)}/{alpha}/qualitative',
                    domain=domain,
                    fix_imbalance=fix_imbalance,
                    target_variable=target_variable,
                    custom_color=custom_color,
                    modelD=modelD,
                    verbose=verbose,
                    clfs=clfs,
                    alpha=alpha,
                    graphWithClass=graphWithClass,
                    topR=list(outperformers.keys()),
                    withCost=withCost,
                    financialOption=financialOption,
                    original = (x1_train, x1_test)
                )

        # 2) build an MLN on just k variable, 1 < k <= len of OHE
        if (3 in levels) and (len(promise[8]) > 0):
            # global_logic =[*global_logic, *
            make_mlna_k_variable_v2(
                default=default,
                x_traini=x_train,
                x_testi=x_test,
                y_traini=y_train,
                y_testi=y_test,
                duration_divider=duration_divider,
                rate_divider=rate_divider,
                OHE=promise[8],
                nominal_factor_colums=promise[2],
                cwd=cwd + f'/outputs/{domain}{"/withClass" * int(graphWithClass)}/{alpha}/qualitative',
                domain=domain,
                fix_imbalance=fix_imbalance,
                target_variable=target_variable,
                custom_color=custom_color,
                modelD=modelD,
                verbose=verbose,
                clfs=clfs,
                alpha=alpha,
                graphWithClass=graphWithClass,
                withCost=withCost,
                financialOption=financialOption,
                original = (x1_train, x1_test)
            )
        # ]

        # table = print_summary([default, *global_logic],modelD)
        # create_file(
        # 	content= table[1],
        # 	cwd= cwd+f"/outputs/{domain}/qualitative",
        # 	prefix= domain,
        # 	filename= f"mlna_for_all_categorial_data",
        # 	extension=".html"
        # 	)

        # elif 'Variables' in key:
        # case where with triggers just categorials data to perform our MLNA

        # build divers logic of MLNA
        # 1) build an MLN on just one variable
        # global logic
        # del global_logic
        # global_logic = []
        # start with model building and importance analysis

        ## default
        # default = make_builder(
        # 	fix_imbalance=fix_imbalance,
        # 	DATA_OVER=value,
        # 	target_variable=target_variable,
        # 	clfs=clfs,
        # 	domain= 'classic_all',
        # 	prefix=domain,
        # 	verbose=verbose,
        # 	cwd= cwd+'/outputs'
        # 	)

        if (4 in levels) and (len(promise[10]) > 0):  # if this stage is allowed

            # global_logic =[*global_logic, *
            make_mlna_1_variable(
                default=default,
                value_graph=promise[9],
                value_clfs=promise[7],
                OHE=promise[10],
                nominal_factor_colums=list(set([*promise[5], *promise[6]]) - set([target_variable])),
                cwd=cwd + f'/outputs/{domain}{"/withClass" * int(graphWithClass)}/{alpha}/quantitative',
                domain=domain,
                fix_imbalance=fix_imbalance,
                target_variable=target_variable,
                custom_color=custom_color,
                modelD=modelD,
                verbose=verbose,
                clfs=clfs,
                alpha=alpha,
                graphWithClass=graphWithClass,
                withCost=withCost,
                financialOption=financialOption
            )
        # ]

        # 2) build an MLN on just k variable, 1 < k <= len of OHE
        if (5 in levels) and (len(promise[10]) > 0):
            # global_logic =[*global_logic, *
            make_mlna_k_variable(
                default=default,
                value_graph=promise[9],
                value_clfs=promise[7],
                OHE=promise[10],
                nominal_factor_colums=sorted(list(set([*promise[5], *promise[6]]) - set([target_variable]))),
                cwd=cwd + f'/outputs/{domain}{"/withClass" * int(graphWithClass)}/{alpha}/quantitative',
                domain=domain,
                fix_imbalance=fix_imbalance,
                target_variable=target_variable,
                custom_color=custom_color,
                modelD=modelD,
                verbose=verbose,
                clfs=clfs,
                alpha=alpha,
                graphWithClass=graphWithClass,
                withCost=withCost,
                financialOption=financialOption
            )
        # ]
        # table = print_summary([default, *global_logic],modelD)
        # create_file(
        # 	content= table[1],
        # 	cwd= cwd+f"/outputs/{domain}/quantitative",
        # 	prefix= domain,
        # 	filename= f"mlna_for_all_data_quantitative",
        # 	extension=".html"
        # 	)

        if (6 in levels) and (len(promise[10]) > 0 and len(OHE) > 0):
            # del global_logic
            # global_logic = []
            # global_logic =[*global_logic, *
            make_mlna_k_variable(
                default=default,
                value_graph=promise[9],
                value_clfs=promise[7],
                OHE=[*promise[8], *promise[10]],
                nominal_factor_colums=sorted(
                    list(set([*promise[4], *promise[5], *promise[6]]) - set([target_variable]))),
                cwd=cwd + f'/outputs/{domain}{"/withClass" * int(graphWithClass)}/{alpha}/mixed',
                domain=domain,
                fix_imbalance=fix_imbalance,
                target_variable=target_variable,
                custom_color=custom_color,
                modelD=modelD,
                verbose=verbose,
                clfs=clfs,
                alpha=alpha,
                graphWithClass=graphWithClass,
                withCost=withCost,
                financialOption=financialOption
            )
        # ]

        # table = print_summary([default, *global_logic],modelD)
        # create_file(
        # 	content= table[1],
        # 	cwd= cwd+f"/outputs/{domain}/mixed",
        # 	prefix= domain,
        # 	filename= f"mlna_for_all_mixed",
        # 	extension=".html"
        # 	)
    return True
