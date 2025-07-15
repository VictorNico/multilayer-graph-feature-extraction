# 03_graph_construction.py

import sys
# Ajoutez le répertoire parent pour pouvoir importer les modules
sys.path.append('..')  # Ajouter le répertoire parent au chemin de recherche des modules
from modules.preprocessing import get_combinations  # Preprocessing functions
from modules.file import *  # File manipulation functions
from modules.graph import *  # Modeling functions
import statistics

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

def generate_config_df(
    graphWithClass=False,
    mlnL='/mlna_1',
    cwd=None,
    root=None,
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
        cwd=cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_GLO_df_for_{name}',
        prefix=domain,
        sep=',',
        sub='/descriptors'
    )
    save_dataset(
        cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_PER_df_for_{name}',
        prefix=domain,
        sep=',',
        sub='/descriptors'
    )
    save_dataset(
        cwd=cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_GLO_df_t'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_GLO_df_t_for_{name}',
        prefix=domain,
        sep=',',
        sub='/descriptors'
    )
    save_dataset(
        cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
        dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}X_PER_df_t'],
        name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}X_PER_df_t_for_{name}',
        prefix=domain,
        sep=',',
        sub='/descriptors'
    )
    if graphWithClass is True:
        save_dataset(
            cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}Y_PER_df_for_{name}',
            prefix=domain,
            sep=',',
            sub='/descriptors'
        )
        save_dataset(
            cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}XY_PER_df_for_{name}',
            prefix=domain,
            sep=',',
            sub='/descriptors'
        )
        save_dataset(
            cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}Y_PER_df_t'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}Y_PER_df_t_for_{name}',
            prefix=domain,
            sep=',',
            sub='/descriptors'
        )
        save_dataset(
            cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
            dataframe=config_df[f'extract_{"C" if graphWithClass is True else "M"}XY_PER_df_t'],
            name=f'{domain}_extract_{"C" if graphWithClass is True else "M"}XY_PER_df_t_for_{name}',
            prefix=domain,
            sep=',',
            sub='/descriptors'
        )

    save_model(
        cwd=cwd + f'{mlnL}/{name}',
        clf=config_df,
        prefix="",
        clf_name=f'config_df_for_{name}_{"withClass" if graphWithClass else "withoutClass"}',
        ext=".conf",
        sub=""
    )

def make_mlna_1_variable_v2(
        x_traini,
        x_testi,
        y_traini,
        y_testi,
        OHE,
        nominal_factor_colums,
        cwd,
        root,
        domain,
        target_variable,
        alpha,
        graphWithClass=False
):
    ## visualization of result
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)
    PERSONS_T = get_persons(x_test)
    for i in range(len(OHE)):
        # build the MLN for the variable i on training dataset
        if sum([f'config_df_for_{nominal_factor_colums[i]}_{"withClass" if graphWithClass else "withoutClass"}' in file for _, _, files in
             os.walk(cwd + f'/mlna_1/{nominal_factor_colums[i]}/') for
             file in files]) > 0 :
            continue
        copT = x_train.copy(deep=True)
        copT[target_variable] = y_train.copy(deep=True)
        MLN = build_mlg_with_class(
            copT, [OHE[i]], target_variable) \
            if (graphWithClass is True) \
            else build_mlg(copT, [OHE[i]])
        # save the graph
        save_graph(
            cwd=cwd + f'/mlna_1/{nominal_factor_colums[i]}',
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
            root=root,
            graphWithClass=graphWithClass,
            mlnL='/mlna_1',
            domain=domain,
            extracts_g=extracts_g,
            extracts_p=extracts_p,
            extracts_g_t=extracts_g_t,
            extracts_p_t=extracts_p_t,
            name=nominal_factor_colums[i]
        )

def make_mlna_k_variable_v2(
        x_traini,
        x_testi,
        y_traini,
        y_testi,
        OHE,
        nominal_factor_colums,
        cwd,
        root,
        domain,
        alpha,
        target_variable,
        graphWithClass=True
):
    ## visualization of result
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)
    PERSONS_T = get_persons(x_test)

    for k in list([2]):  # for 1<k<|OHE[i]|+2
        # for k in [2]: # for 1<k<|OHE[i]|+2
        # for k in range(2:len(OHE)+1: # for 1<k<|OHE[i]|+2
        for layer_config in get_combinations(range(len(OHE)), k):  # create subsets of k index of OHE and fetch it
            # for layer_config in [[1,2]]: # create subsets of k index of OHE and fetch it
            copT = x_train.copy(deep=True)
            copT[target_variable] = y_train.copy(deep=True)
            col_targeted = [f'{nominal_factor_colums[i]}' for i in layer_config]
            case_k = '_'.join(col_targeted)
            if sum([
                       f'config_df_for_{case_k}_{"withClass" if graphWithClass else "withoutClass"}' in file
                       for _, _, files in
                       os.walk(cwd + f'/mlna_{k}/{case_k}/') for
                       file in files]) > 0:
                continue
            # build the MLN for the variable i
            MLN = build_mlg_with_class(
                    copT, [OHE[i] for i in layer_config],
                    target_variable
                ) \
                if (graphWithClass is True) \
                else build_mlg(copT,[OHE[i] for i in layer_config]
                               )
            # save the graph
            save_graph(
                cwd=cwd + f'/mlna_{k}/{case_k}',
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
                root=root,
                graphWithClass=graphWithClass,
                mlnL=f'/mlna_{k}',
                domain=domain,
                extracts_g=extracts_g,
                extracts_p=extracts_p,
                extracts_g_t=extracts_g_t,
                extracts_p_t=extracts_p_t,
                name=case_k
            )

def make_mlna_top_k_variable_v2(
        x_traini,
        x_testi,
        y_traini,
        y_testi,
        OHE,
        nominal_factor_colums,
        cwd,
        root,
        domain,
        target_variable,
        alpha,
        graphWithClass=False,
        topR=[]
):
    ## visualization of result
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)
    PERSONS_T = get_persons(x_test)

    for k in range(2, len(topR) + 1):  # for 1<k<|OHE[i]|+2
        # for k in [2]: # for 1<k<|OHE[i]|+2
        # for k in range(2:len(OHE)+1: # for 1<k<|OHE[i]|+2
        layer_config = topR[:k]  # create subsets of k index of OHE and fetch it
        copT = x_train.copy(deep=True)
        copT[target_variable] = y_train.copy(deep=True)
        col_targeted = [f'{nominal_factor_colums[i]}' for i in layer_config]
        case_k = '_'.join(col_targeted)
        if sum([
            f'config_df_for_{case_k}_{"withClass" if graphWithClass else "withoutClass"}' in file
            for _, _, files in
            os.walk(cwd + f'/mlna_{k}_b/{case_k}/') for
            file in files]) > 0:
            continue
        # build the MLN for the variable i
        MLN = build_mlg_with_class(copT, [OHE[i] for i in layer_config],
                                   target_variable) if (graphWithClass is True) else build_mlg(
            copT, [OHE[i] for i in layer_config])
        # save the graph
        save_graph(
            cwd=cwd + f'/mlna_{k}_b/{case_k}',
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
            root=root,
            graphWithClass=graphWithClass,
            mlnL=f'/mlna_{k}_b',
            domain=domain,
            extracts_g=extracts_g,
            extracts_p=extracts_p,
            extracts_g_t=extracts_g_t,
            extracts_p_t=extracts_p_t,
            name=case_k
        )

def main():
    # Définition des arguments
    import argparse

    parser = argparse.ArgumentParser(description='Exécution du pipeline MLNA - SPLIT DATASET')
    parser.add_argument('--cwd', type=str, required=True, help='Répertoire de travail courant')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Nom du dataset')
    parser.add_argument('--alpha', type=float, required=True, help='Valeur d\'alpha')
    parser.add_argument('--turn', type=int, required=True, help='Valeur du tour')
    parser.add_argument('--graph_with_class', action="store_true", required=False, help='integrant les classes?')

    # Récupération des arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    config = load_config(f"{args.cwd}/configs/{args.dataset_folder}/config.ini")

    domain = config["DATA"]["domain"]

    encoding = config["PREPROCESSING"]["encoding"]
    dataset_delimiter = config["SPLIT"]["dataset_delimiter"]
    target_variable = config["DATA"]["target"]

    processed_dir = config["GENERAL"]["processed_dir"]
    split_dir = config["GENERAL"]["split_dir"]
    results_dir = config["GENERAL"]["results_dir"]
    target_columns_type = config["GENERAL"]["target_columns_type"]
    verbose = config.getboolean("GENERAL", "verbose")

    index_col = None if config["SPLIT"]["index_col"] in ["None", ""] else config.getint("SPLIT", "index_col")

    # ------------------------------------------------------------------------------------------------------------------
    # lookup existing train test file
    if sum([f'{domain}_train' in file for _, _, files in
            os.walk(args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}') for
            file in files]) == 0:
        print("❌ Unable to access splits data")
        exit(1)

    # get path
    xtrain_path = args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}/' + [file for _, _, files in
         os.walk(
             args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}')
         for
         file in files][
        [f'{domain}_train' in file for _, _, files in
         os.walk(args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}') for
         file in files].index(True)
    ]
    xtest_path = args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}/' + [file for _, _, files in
        os.walk(
            args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}')
        for
        file in files][
        [f'{domain}_test' in file for _, _, files in
         os.walk(args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}') for
         file in files].index(True)
    ]

    # load xtrain and xtest files
    print("the xtrain path is ", xtrain_path) if verbose else None
    print("the xtest path is ", xtest_path) if verbose else None
    X_train = load_data_set_from_url(path=xtrain_path, sep=dataset_delimiter, encoding=encoding,
                                     index_col=index_col,
                                     na_values=None)

    y_traini = X_train[target_variable]
    x_traini = X_train.drop(columns=[target_variable])

    X_test = load_data_set_from_url(path=xtest_path, sep=dataset_delimiter, encoding=encoding,
                                     index_col=index_col,
                                     na_values=None)

    y_testi = X_test[target_variable]
    x_testi = X_test.drop(columns=[target_variable])

    print("the x_train shape is ", X_train.shape) if verbose else None
    print("the x_test shape is ", X_test.shape) if verbose else None

    # ------------------------------------------------------------------------------------------------------------------
    if sum([f'preprocessing' in file for _, _, files in
            os.walk(args.cwd + f'/{processed_dir}{args.dataset_folder}') for
            file in files]) == 0:
        print("❌ Unable to access preprocessing config")
        exit(1)

    # load preprocessing configs
    prepro_path = args.cwd + f'/{processed_dir}{args.dataset_folder}/' + [file for _, _, files in
       os.walk(
           args.cwd + f'/{processed_dir}{args.dataset_folder}/')
       for
       file in files][
        [f'preprocessing' in file for _, _, files in
         os.walk(args.cwd + f'/{processed_dir}{args.dataset_folder}/') for
         file in files].index(True)
    ]
    prepro_config = read_model(path=prepro_path)

    # ------------------------------------------------------------------------------------------------------------------
    if target_columns_type == "cat":
        OHE = prepro_config["OHE"]
        columns = prepro_config["categorial_col"]
    elif target_columns_type == "num":
        OHE = [*prepro_config["OHE_2"]]
        columns = list(
            {*prepro_config["numeric_with_outliers_columns"],
             *prepro_config["numeric_uniform_colums"]} - {
                target_variable})
    else:
        OHE = [*prepro_config["OHE"], *prepro_config["OHE_2"]]
        columns = list(
            {*prepro_config["categorial_col"], *prepro_config["numeric_with_outliers_columns"], *prepro_config["numeric_uniform_colums"]} - {
                target_variable})

    # ------------------------------------------------------------------------------------------------------------------
    if args.turn == 1: # check if we are onto the first turn
        if sum(['graph_turn_1_completed.dtvni' == file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_1') for
                file in files]) > 0:
            print("✅ MLNA 1 Graph already completed")
        else:
            make_mlna_1_variable_v2(
                x_traini=x_traini,
                x_testi=x_testi,
                y_traini=y_traini,
                y_testi=y_testi,
                OHE=OHE,
                nominal_factor_colums=columns,
                cwd = args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}',
                root= args.cwd,
                domain= domain,
                target_variable= target_variable,
                alpha= args.alpha,
                graphWithClass=False
            )
            with open(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_1/graph_turn_1_completed.dtvni', "a") as fichier:
                fichier.write("")
    if args.turn == 2:  # check if we are onto the first turn
        if sum([f"MNIFS_{domain}_best_features" in file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}') for
                file in files]) == 0:
            print("❌ Unable to access selection protocol results")
            exit(1)

        if sum(['graph_turn_2_completed.dtvni' == file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/select') for
                file in files]) > 0:
            print("✅ MLNA 1 Graph already completed")
        else:

            # load mnifs configs
            mnifs_path = args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/' + [file for _, _, files in
                os.walk(
                  args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}')
                for
                file in files][
                [f"MNIFS_{domain}_best_features" in file for _, _, files in
                 os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}') for
                 file in files].index(True)
            ]
            mnifs_config = read_model(path=mnifs_path)

            make_mlna_top_k_variable_v2(
                x_traini=x_traini,
                x_testi=x_testi,
                y_traini=y_traini,
                y_testi=y_testi,
                OHE=OHE,
                nominal_factor_colums=columns,
                cwd=args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/select',
                root=args.cwd,
                domain=domain,
                target_variable=target_variable,
                alpha=args.alpha,
                graphWithClass=args.graph_with_class,
                topR=list(mnifs_config['model'].keys())
            )
            with open(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/select/graph_turn_2_completed.dtvni', "a") as fichier:
                fichier.write("")
    if args.turn == 3:  # check if we are onto the first turn
        if sum(['graph_turn_3_completed.dtvni' == file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_2/') for
                file in files]) > 0:
            print("✅ COMBINATORY MLNA 2 Graph  already completed")
        else:
            make_mlna_k_variable_v2(
                x_traini=x_traini,
                x_testi=x_testi,
                y_traini=y_traini,
                y_testi=y_testi,
                OHE=OHE,
                nominal_factor_colums=columns,
                cwd = args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}',
                root= args.cwd,
                domain= domain,
                target_variable= target_variable,
                alpha= args.alpha,
                graphWithClass=False
            )
            with open(
                    args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_2/graph_turn_3_completed.dtvni',
                    "a") as fichier:
                fichier.write("")

    print("Descripteurs extraits et sauvegardés.")

if __name__ == "__main__":
    main()

