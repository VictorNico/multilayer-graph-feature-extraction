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
#################################################
##          Libraries importation
#################################################

###### Begin
from tqdm import tqdm
import importlib
# List des modules to load
modules = [
	'modules.file', 
	'modules.eda', 
	'modules.graph',
	'modules.modeling',
	'modules.preprocessing',
	'modules.report',
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
		isNAColumns= get_na_columns(dataframe)
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
	dataset_original = dataset.copy(deep=True) # savea backup of our dataset

	# binarise nominals factors
	DATA_OHE, OHE = nominal_factor_encoding(
		dataset, 
		categorial_col if not(target_variable in categorial_col) else list(set(categorial_col)-set([target_variable]))
		) if len(categorial_col) > 0 else (dataset, [])
	DATA_OHE = make_eda(dataframe=DATA_OHE, verbose=verbose)
	print(f"OHE <----> {OHE}")
	# # label encoding of ordinal data
	DATA_OHE_LB = ordinal_factor_encoding(
		DATA_OHE, 
		[target_variable]
		) if len([target_variable]) > 0 else DATA_OHE

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
		numeric_uniform_colums if not(target_variable in numeric_uniform_colums) else list(set(numeric_uniform_colums)-set([target_variable]))
		) if len(numeric_uniform_colums) > 0 else DATA_OHE_LB
	print(f"{get_na_columns(DATA_OHE_LB_LBU_STDU)}")
	DATA_OHE_LB_LBU_STDU = make_eda(dataframe=DATA_OHE_LB_LBU_STDU, verbose=verbose)
	# normalisation of numeric data with outliers to deeve it into interval 0,1
	DATA_OHE_LB_LBU_STDU_STDWO = numeric_standardization_with_outliers(
		DATA_OHE_LB_LBU_STDU,
		numeric_with_outliers_columns if not(target_variable in numeric_with_outliers_columns) else list(set(numeric_with_outliers_columns)-set([target_variable]))
		) if len(numeric_with_outliers_columns) > 0 else DATA_OHE_LB_LBU_STDU
	print(f"{get_na_columns(DATA_OHE_LB_LBU_STDU_STDWO)}")
	DATA_OHE_LB_LBU_STDU_STDWO = make_eda(dataframe=DATA_OHE_LB_LBU_STDU_STDWO, verbose=verbose)
	# save the first one where just all factor are binarized
	# _ALL_PRETRAETED_DATA['withAllFactorsVBinarized']= DATA_OHE_LB_LBU_STDU_STDWO.copy(deep=True)
	print(f"{get_na_columns(DATA_OHE_LB_LBU_STDU_STDWO)}")
	# save preprocessed 1st one data
	link_to_preprocessed_factor_data = save_dataset(
		cwd= cwd+'/mlna_preprocessing', 
		dataframe= DATA_OHE_LB_LBU_STDU_STDWO, 
		name= f'{domain}_preprocessed', 
		# prefix= domain, 
		sep= ','
		)

	if (4 in levels) or (5 in levels):
		# discretize nuneric value
		DATA_DISCRETIZE = discretise_numeric_dimension(
			columns= list(set([*numeric_with_outliers_columns, *numeric_uniform_colums])-set([target_variable])), 
			dataframe= DATA_OHE_LB_LBU_STDU_STDWO, 
			inplace=False,
			verbose= verbose
			) if len(list(set([*numeric_with_outliers_columns, *numeric_uniform_colums])-set([target_variable]))) > 0 else DATA_OHE_LB_LBU_STDU_STDWO
		link_to_preprocessed_disc_data = save_dataset(
			cwd= cwd+'/mlna_preprocessing', 
			dataframe= DATA_DISCRETIZE, 
			name= f'{domain}_preprocessed_discretize', 
			# prefix= domain, 
			sep= ','
			)
		print(f"{get_na_columns(DATA_DISCRETIZE)}")
		DATA_DISCRETIZE = make_eda(dataframe=DATA_DISCRETIZE, verbose=verbose)
		# binarise nominals factors
		DATA_DISCRETIZE_OHE_2, OHE_2 = nominal_factor_encoding(
			DATA_DISCRETIZE, 
			list(set([*numeric_with_outliers_columns, *numeric_uniform_colums])-set([target_variable]))
			)  if len(list(set([*numeric_with_outliers_columns, *numeric_uniform_colums])-set([target_variable]))) > 0 else (DATA_DISCRETIZE, [])
		print(f"{get_na_columns(DATA_DISCRETIZE_OHE_2)}")
		DATA_DISCRETIZE_OHE_2 = make_eda(dataframe=DATA_DISCRETIZE_OHE_2, verbose=verbose)
	
		link_to_preprocessed_all_data = save_dataset(
			cwd= cwd+'/mlna_preprocessing', 
			dataframe= DATA_DISCRETIZE_OHE_2, 
			name= f'{domain}_preprocessed_all', 
			# prefix= domain, 
			sep= ','
			)
		print(f"discretize data shape: {DATA_DISCRETIZE_OHE_2.shape}") if verbose else None
		print(f"{get_na_columns(DATA_DISCRETIZE_OHE_2)}")
		# save the second one where all variables are binarized
		# _ALL_PRETRAETED_DATA['withAllVariablesBinarized']= DATA_DISCRETIZE_OHE_2.copy(deep=True)
		return [ 
			col_list,
			numeric_col,
			categorial_col if not(target_variable in categorial_col) else list(set(categorial_col)-set([target_variable])),
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
		categorial_col if not(target_variable in categorial_col) else list(set(categorial_col)-set([target_variable])),
		ordinal_factor_colums,
		nominal_factor_colums,
		numeric_with_outliers_columns,
		numeric_uniform_colums,
		DATA_OHE_LB_LBU_STDU_STDWO,
		OHE
		]

# @profile
def make_mlna_1_variable(default, value_graph, value_clfs, OHE, nominal_factor_colums, cwd, domain, fix_imbalance, target_variable, custom_color, modelD, verbose, clfs, alpha):
	"""
	"""
	## visualization of result
	modelD = model_desc()
	PERSONS = get_persons(value_graph)
	#print(PERSONS)
	# local eval storage
	logic_g = []
	logic_p = []
	for i in range(len(OHE)):
		# logic storage
		logic_i_g = []
		logic_i_p = []
		# build the MLN for the variable i
		MLN = build_mlg(value_graph, [OHE[i]])
		#print(value_clfs.iloc[:,])
		# save the graph
		save_graph(
			cwd= cwd+'/mlna_1', 
			graph= MLN, 
			name= f'{nominal_factor_colums[i]}_mln', 
			rows_len= value_clfs.shape[0], 
			prefix= domain, 
			cols_len= len(OHE[i])
			)

		# Binome page rank
		bipart_combine = nx.pagerank(MLN, alpha=alpha)

		# get intra page rank
		bipart_intra_pagerank = nx.pagerank(
			MLN,
			personalization=compute_personlization(get_intra_node_label(MLN),MLN), 
			alpha=alpha
			)

		# get inter page rank
		bipart_inter_pagerank = nx.pagerank(
			MLN,
			personalization=compute_personlization(get_inter_node_label(MLN),MLN), 
			alpha=alpha
			)

		# extract centrality degree from MLN
		extracts_g = {}
		extracts_p = {}
		
		extracts_g[f'MLN_{nominal_factor_colums[i]}_degree'] = [
			get_number_of_borrowers_with_same_n_layer_value(borrower= index, graph= MLN, layer_nber=0)[1] for index in PERSONS
			]
		extracts_p[f'MLN_{nominal_factor_colums[i]}_degree'] = extracts_g[f'MLN_{nominal_factor_colums[i]}_degree']
		extracts_g[f'MLN_bipart_intra_{nominal_factor_colums[i]}'] = get_max_borrower_pr(bipart_intra_pagerank)
		extracts_g[f'MLN_bipart_inter_{nominal_factor_colums[i]}'] = get_max_borrower_pr(bipart_inter_pagerank)
		extracts_g[f'MLN_bipart_combine_{nominal_factor_colums[i]}'] = get_max_borrower_pr(bipart_combine)
		extracts_p[f'MLN_bipart_combine_perso_{nominal_factor_colums[i]}'] = [get_max_borrower_pr(nx.pagerank(MLN,personalization=compute_personlization(get_combine_perso_nodes_label(MLN, [val], 1)[0][0],MLN), alpha=alpha))[index] for index, val in enumerate(PERSONS)]
		extracts_p[f'MLN_bipart_inter_perso_{nominal_factor_colums[i]}'] = [get_max_borrower_pr(nx.pagerank(MLN,personalization=compute_personlization(get_inter_perso_nodes_label(MLN, [val], 1)[0][0],MLN), alpha=alpha))[index] for index, val in enumerate(PERSONS)]
		extracts_p[f'MLN_bipart_intra_perso_{nominal_factor_colums[i]}'] = [get_max_borrower_pr(nx.pagerank(MLN,personalization=compute_personlization(get_intra_perso_nodes_label(MLN, [val], 1)[0][0],MLN), alpha=alpha))[index] for index, val in enumerate(PERSONS)]
		extracts_g[f'MLN_bipart_intra_max_{nominal_factor_colums[i]}'] = [get_max_modality_pagerank_score(index, MLN, 1, bipart_intra_pagerank) for index in PERSONS]
		extracts_g[f'MLN_bipart_inter_max_{nominal_factor_colums[i]}'] = [get_max_modality_pagerank_score(index, MLN, 1, bipart_inter_pagerank) for index in PERSONS]
		extracts_g[f'MLN_bipart_combine_max_{nominal_factor_colums[i]}'] = [get_max_modality_pagerank_score(index, MLN, 1, bipart_combine) for index in PERSONS]
		extracts_p[f'MLN_bipart_combine_perso_max_{nominal_factor_colums[i]}'] = [get_max_modality_pagerank_score(val, MLN, 1, nx.pagerank(MLN,personalization=compute_personlization(get_combine_perso_nodes_label(MLN, [val], 1)[0][0],MLN), alpha=alpha)) for index, val in enumerate(PERSONS)]
		extracts_p[f'MLN_bipart_inter_perso_max_{nominal_factor_colums[i]}'] = [get_max_modality_pagerank_score(val, MLN, 1, nx.pagerank(MLN,personalization=compute_personlization(get_inter_perso_nodes_label(MLN, [val], 1)[0][0],MLN), alpha=alpha)) for index, val in enumerate(PERSONS)]
		extracts_p[f'MLN_bipart_intra_perso_max_{nominal_factor_colums[i]}'] = [get_max_modality_pagerank_score(val, MLN, 1, nx.pagerank(MLN,personalization=compute_personlization(get_intra_perso_nodes_label(MLN, [val], 1)[0][0],MLN), alpha=alpha)) for index, val in enumerate(PERSONS)]
		# ultra personalisation of pagerank
		#(nodes_of_borrowers, nodes) = get_user_nodes_label(MLN, PERSONS, 1) # get all posible relate borrower's nodes
		# get ultra personalized pagerank of distincts values of nodes
		# build a dict base on values of nodes as key
		#print("edges of borrowers") if verbose else None
		#plural_form = {"".join(list(k)): nx.pagerank(MLN,personalization=compute_personlization(list(k)), alpha=alpha) for k in nodes} 
		#print("ultra PR") if verbose else None
		#extracts_p[f'MLN_bipart_ultra_{nominal_factor_colums[i]}'] = [get_max_borrower_pr(plural_form["".join(nodes_of_borrowers[index])])[index] for index, val in enumerate(PERSONS)] 
		#extracts_p[f'MLN_bipart_ultra_max_{nominal_factor_colums[i]}'] = [get_max_modality_pagerank_score(val, MLN, 1,plural_form["".join(nodes_of_borrowers[index])]) for index, val in enumerate(PERSONS)]
		#print("descriptor compute") if verbose else None
		# standardization
		standard_extraction(extracts_g, extracts_g.keys())
		standard_extraction(extracts_p, extracts_p.keys())

		## default - MLNa
		mlna = set()
		col_list = value_clfs.columns.to_list()
		for column in OHE[i]:
			if column in col_list:
				mlna.add(column)
			else:
				mlna.add(nominal_factor_colums[i])

		# classic
		plot_features_importance_as_barh(
			    data= default[1],
			    getColor= custom_color,
			    modelDictName= modelD,
			    plotTitle= "Classic features importance",
			    prefix= domain, 
			    cwd= cwd+'/mlna_1',
			    graph_a= list(mlna),
			    save= True
			)

		# save descriptors
		extract_g_df = pd.DataFrame(extracts_g)
		extract_p_df = pd.DataFrame(extracts_p)
		link_to_preprocessed_all_data = save_dataset(
			cwd= cwd+'/mlna_1/global', 
			dataframe= extract_g_df, 
			name= f'{domain}_extracted_features_mln_for_{nominal_factor_colums[i]}', 
			prefix= domain, 
			sep= ','
			)
		link_to_preprocessed_all_data = save_dataset(
			cwd= cwd+'/mlna_1/personalized', 
			dataframe= extract_p_df, 
			name= f'{domain}_extracted_features_mln_for_{nominal_factor_colums[i]}', 
			prefix= domain, 
			sep= ','
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
		## default + MLN
		# inject descriptors
		VALUE_MLN_G = inject_features_extracted(value_clfs, extracts_g)
		VALUE_MLN_P = inject_features_extracted(value_clfs, extracts_p)
		extracts_g = None
		extracts_p = None
		# del extracts_g
		# del extracts_p

		logic_i_p.append(
			make_builder(
				fix_imbalance=fix_imbalance, 
				DATA_OVER=VALUE_MLN_G, 
				target_variable=target_variable, 
				clfs=clfs, 
				domain= f'classic_mln_{nominal_factor_colums[i]}', 
				prefix=domain,
				verbose=verbose,
				cwd= cwd+'/mlna_1/personalized'
				)
			)
		logic_i_g.append(
			make_builder(
				fix_imbalance=fix_imbalance, 
				DATA_OVER=VALUE_MLN_P, 
				target_variable=target_variable, 
				clfs=clfs, 
				domain= f'classic_mln_{nominal_factor_colums[i]}', 
				prefix=domain,
				verbose=verbose,
				cwd= cwd+'/mlna_1/global'
				)
			)

		# classic + mln
		plot_features_importance_as_barh(
			    data= logic_i_p[-1][1],
			    getColor= custom_color,
			    modelDictName= modelD,
			    plotTitle= f"Classic + mln features importance for_{nominal_factor_colums[i]}",
			    prefix= domain, 
			    cwd= cwd+'/mlna_1/personalized',
			    graph_a= list(mlna),
			    save= True
			)
		plot_features_importance_as_barh(
			    data= logic_i_p[-1][1],
			    getColor= custom_color,
			    modelDictName= modelD,
			    plotTitle= f"Classic + mln features importance for_{nominal_factor_colums[i]}",
			    prefix= domain, 
			    cwd= cwd+'/mlna_1/global',
			    graph_a= list(mlna),
			    save= True
			)

		VALUE_MINUS_MLNa = value_clfs.drop(list(mlna), axis=1)

		logic_i_g.append(
			make_builder(
				fix_imbalance=fix_imbalance, 
				DATA_OVER=VALUE_MINUS_MLNa, 
				target_variable=target_variable, 
				clfs=clfs, 
				domain= f'classic_-_mlna_{nominal_factor_colums[i]}', 
				prefix=domain,
				verbose=verbose,
				cwd= cwd+'/mlna_1'
				)
			)
		logic_i_p.append(logic_i_g[-1])
		# classic - mln attribut
		plot_features_importance_as_barh(
			    data= logic_i_g[-1][1],
			    getColor= custom_color,
			    modelDictName= modelD,
			    plotTitle= f"Classic - mln attributs features importance for_{nominal_factor_colums[i]}",
			    prefix= domain, 
			    cwd= cwd+'/mlna_1',
			    graph_a= list(mlna),
			    save= True
			)

		## default + MLN - MLNa
		VALUE_MLN_MINUS_MLN_P = VALUE_MLN_P.drop(list(mlna), axis=1)
		VALUE_MLN_MINUS_MLN_G = VALUE_MLN_G.drop(list(mlna), axis=1)

		logic_i_p.append(
			make_builder(
				fix_imbalance=fix_imbalance, 
				DATA_OVER=VALUE_MLN_MINUS_MLN_P, 
				target_variable=target_variable, 
				clfs=clfs, 
				domain= f'classic_mln_-_mlna_{nominal_factor_colums[i]}', 
				prefix=domain,
				verbose=verbose,
				cwd= cwd+'/mlna_1/personalized'
				)
			)
		logic_i_g.append(
			make_builder(
				fix_imbalance=fix_imbalance, 
				DATA_OVER=VALUE_MLN_MINUS_MLN_G, 
				target_variable=target_variable, 
				clfs=clfs, 
				domain= f'classic_mln_-_mlna_{nominal_factor_colums[i]}', 
				prefix=domain,
				verbose=verbose,
				cwd= cwd+'/mlna_1/global'
				)
			)
		# classic + mln - mln attribut
		plot_features_importance_as_barh(
			    data= logic_i_p[-1][1],
			    getColor= custom_color,
			    modelDictName= modelD,
			    plotTitle= f"Classic + mln - mln attributs features importance for_{nominal_factor_colums[i]}",
			    prefix= domain, 
			    cwd= cwd+'/mlna_1/personalized',
			    graph_a= list(mlna),
			    save= True
			)
		plot_features_importance_as_barh(
			    data= logic_i_g[-1][1],
			    getColor= custom_color,
			    modelDictName= modelD,
			    plotTitle= f"Classic + mln - mln attributs features importance for_{nominal_factor_colums[i]}",
			    prefix= domain, 
			    cwd= cwd+'/mlna_1/global',
			    graph_a= list(mlna),
			    save= True
			)

		# print html report for analysis of variable i
		#logic_i=[default,*logic_i]
		logic_p = [*logic_p, *logic_i_p]
		logic_g = [*logic_g, *logic_i_g]
		table_p = print_summary([default,*logic_i_p],modelD)
		table_g = print_summary([default,*logic_i_g],modelD)
		create_file(
			content= table_p[1], 
			cwd= cwd+'/mlna_1/personalized', 
			prefix= domain, 
			filename= f"mlna_for_{nominal_factor_colums[i]}", 
			extension=".html"
			)
		create_file(
			content= table_g[1], 
			cwd= cwd+'/mlna_1/global', 
			prefix= domain, 
			filename= f"mlna_for_{nominal_factor_colums[i]}", 
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
	table_p = print_summary([default,*logic_p],modelD)
	table_g = print_summary([default,*logic_g],modelD)
	create_file(
		content= table_p[1], 
		cwd= cwd+'/mlna_1/personalized', 
		prefix= domain, 
		filename= f"mlna_for_all_categorial data", 
		extension=".html"
		)
	create_file(
		content= table_g[1], 
		cwd= cwd+'/mlna_1/global', 
		prefix= domain, 
		filename= f"mlna_for_all_categorial data", 
		extension=".html"
		)
	modelD = None
	PERSONS = None
	table_p = None
	table_g = None
	# del modelD
	# del PERSONS
	# return logic

# @profile
def make_mlna_k_variable(default, value_graph, value_clfs, OHE, nominal_factor_colums, cwd, domain, fix_imbalance, target_variable, custom_color, modelD, verbose, clfs, alpha):
	"""
	"""

	# local eval storage
	logic_g = []
	logic_p = []
	## visualization of result
	modelD = model_desc()
	PERSONS = get_persons(value_clfs)


	for k in list(set([len(OHE)])): # for 1<k<|OHE[i]|+2
	# for k in [2]: # for 1<k<|OHE[i]|+2
	# for k in range(2:len(OHE)+1: # for 1<k<|OHE[i]|+2
		for layer_config in get_combinations(range(len(OHE)),k): # create subsets of k index of OHE and fetch it
		#for layer_config in [[1,2]]: # create subsets of k index of OHE and fetch it
			# logic storage
			logic_i_g = []
			logic_i_p = []
			# build the MLN for the variable i
			MLN = build_mlg(value_graph, [OHE[i] for i in layer_config])
			col_targeted= [f'{nominal_factor_colums[i]}' for i in layer_config]
			case_k= '_'.join(col_targeted)
			# save the graph
			save_graph(
				cwd= cwd+f'/mlna_{k}', 
				graph= MLN, 
				name= f'{case_k}_mln', 
				rows_len= value_graph.shape[0], 
				prefix= domain, 
				cols_len= len(OHE)
				)

			# Binome page rank
			bipart_combine = nx.pagerank(MLN, alpha=alpha)

			# get intra page rank
			bipart_intra_pagerank = nx.pagerank(
				MLN,
				personalization=compute_personlization(get_intra_node_label(MLN),MLN), 
				alpha=alpha
				)


			# get inter page rank
			bipart_inter_pagerank = nx.pagerank(
				MLN,
				personalization=compute_personlization(get_inter_node_label(MLN),MLN), 
				alpha=alpha
				)

			# extract centrality degree from MLN
			extracts_g = {}
			extracts_p = {}
			
			extracts_g[f'MLN_{case_k}_degree'] = [
				get_number_of_borrowers_with_same_custom_layer_value(borrower= index, graph= MLN, custom_layer= range(k))[1] for index in PERSONS
				]
			extracts_p[f'MLN_{case_k}_degree'] = extracts_g[f'MLN_{case_k}_degree']
			extracts_g[f'MLN_bipart_intra_{case_k}'] = get_max_borrower_pr(bipart_intra_pagerank)
			extracts_g[f'MLN_bipart_inter_{case_k}'] = get_max_borrower_pr(bipart_inter_pagerank)
			extracts_g[f'MLN_bipart_combine_{case_k}'] = get_max_borrower_pr(bipart_combine)
			extracts_p[f'MLN_bipart_combine_perso_{case_k}'] = [get_max_borrower_pr(nx.pagerank(MLN,personalization=compute_personlization(get_combine_perso_nodes_label(MLN, [val], k)[0][0],MLN), alpha=alpha))[index] for index, val in enumerate(PERSONS)]
			extracts_p[f'MLN_bipart_inter_perso_{case_k}'] = [get_max_borrower_pr(nx.pagerank(MLN,personalization=compute_personlization(get_inter_perso_nodes_label(MLN, [val], k)[0][0],MLN), alpha=alpha))[index] for index, val in enumerate(PERSONS)]
			extracts_p[f'MLN_bipart_intra_perso_{case_k}'] = [get_max_borrower_pr(nx.pagerank(MLN,personalization=compute_personlization(get_intra_perso_nodes_label(MLN, [val], k)[0][0],MLN), alpha=alpha))[index] for index, val in enumerate(PERSONS)]
			extracts_g[f'MLN_bipart_intra_max_{case_k}'] = [get_max_modality_pagerank_score(index, MLN, k, bipart_intra_pagerank) for index in PERSONS]
			extracts_g[f'MLN_bipart_inter_max_{case_k}'] = [get_max_modality_pagerank_score(index, MLN, k, bipart_inter_pagerank) for index in PERSONS]
			extracts_g[f'MLN_bipart_combine_max_{case_k}'] = [get_max_modality_pagerank_score(index, MLN, k, bipart_combine) for index in PERSONS]
			extracts_p[f'MLN_bipart_combine_perso_max_{case_k}'] = [get_max_modality_pagerank_score(val, MLN, k, nx.pagerank(MLN,personalization=compute_personlization(get_combine_perso_nodes_label(MLN, [val], k)[0][0],MLN), alpha=alpha)) for index, val in enumerate(PERSONS)]
			extracts_p[f'MLN_bipart_inter_perso_max_{case_k}'] = [get_max_modality_pagerank_score(val, MLN, k, nx.pagerank(MLN,personalization=compute_personlization(get_inter_perso_nodes_label(MLN, [val], k)[0][0],MLN), alpha=alpha)) for index, val in enumerate(PERSONS)]
			extracts_p[f'MLN_bipart_intra_perso_max_{case_k}'] = [get_max_modality_pagerank_score(val, MLN, k, nx.pagerank(MLN,personalization=compute_personlization(get_intra_perso_nodes_label(MLN, [val], k)[0][0],MLN), alpha=alpha)) for index, val in enumerate(PERSONS)]
			# ultra personalisation of pagerank
			#(nodes_of_borrowers, nodes) = get_user_nodes_label(MLN, PERSONS, k) # get all posible relate borrower's nodes
			# get ultra personalized pagerank of distincts values of nodes
			# build a dict base on values of nodes as key
			#plural_form = {"".join(list(b)): nx.pagerank(MLN,personalization=compute_personlization(list(b)), alpha=alpha) for b in nodes} 
			
			#extracts_p[f'MLN_bipart_ultra_{case_k}'] = [get_max_borrower_pr(plural_form["".join(nodes_of_borrowers[index])])[index] for index, val in enumerate(PERSONS)] 
			#extracts_p[f'MLN_bipart_ultra_max_{case_k}'] = [get_max_modality_pagerank_score(val, MLN, k,plural_form["".join(nodes_of_borrowers[index])]) for index, val in enumerate(PERSONS)]
			
			# standardization
			standard_extraction(extracts_p, extracts_p.keys())
			standard_extraction(extracts_g, extracts_g.keys())
			
			# save descriptors
			extract_g_df = pd.DataFrame(extracts_g)
			extract_p_df = pd.DataFrame(extracts_p)
			save_dataset(
				cwd= cwd+f'/mlna_{k}/global', 
				dataframe= extract_g_df, 
				name= f'{domain}_extracted_g_features_mln_for_{case_k}', 
				prefix= domain, 
				sep= ','
				)
			save_dataset(
				cwd= cwd+f'/mlna_{k}/personalized', 
				dataframe= extract_p_df, 
				name= f'{domain}_extracted_p_features_mln_for_{case_k}', 
				prefix= domain, 
				sep= ','
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
			col_list = value_clfs.columns.to_list()
			for i in layer_config:
				for column in OHE[i]:
					if column in col_list:
						mlna.add(column)
					else:
						mlna.add(column.split("__")[-1])
			# classic
			plot_features_importance_as_barh(
				    data= default[1],
				    getColor= custom_color,
				    modelDictName= modelD,
				    plotTitle= "Classic features importance",
				    prefix= domain, 
				    cwd= cwd+f'/mlna_{k}',
				    graph_a= list(mlna),
				    save= True
				)
			## default + MLN
			# inject descriptors
			VALUE_MLN_G = inject_features_extracted(value_clfs, extracts_g)
			VALUE_MLN_P = inject_features_extracted(value_clfs, extracts_p)
			extracts_g = None
			extracts_p = None
			# del extracts_g
			# del extracts_p

			logic_i_g.append(
			        make_builder(
				    fix_imbalance=fix_imbalance, 
				    DATA_OVER=VALUE_MLN_G, 
				    target_variable=target_variable, 
				    clfs=clfs, 
				    domain= f'classic_mln_{case_k}', 
				    prefix=domain,
				    verbose=verbose,
				    cwd= cwd+f'/mlna_{k}/global'
				    )
				)
			logic_i_p.append(
				make_builder(
					fix_imbalance=fix_imbalance, 
					DATA_OVER=VALUE_MLN_P, 
					target_variable=target_variable, 
					clfs=clfs, 
					domain= f'classic_mln_{case_k}', 
					prefix=domain,
					verbose=verbose,
					cwd= cwd+f'/mlna_{k}/personalized'
					)
				)
			# classic + mln
			plot_features_importance_as_barh(
				    data= logic_i_g[0][1],
				    getColor= custom_color,
				    modelDictName= modelD,
				    plotTitle= f"Classic + mln features importance for_{case_k}",
				    prefix= domain, 
				    cwd= cwd+f'/mlna_{k}/global',
				    graph_a= list(mlna),
				    save= True
				)
			plot_features_importance_as_barh(
				    data= logic_i_p[0][1],
				    getColor= custom_color,
				    modelDictName= modelD,
				    plotTitle= f"Classic + mln features importance for_{case_k}",
				    prefix= domain, 
				    cwd= cwd+f'/mlna_{k}/personalized',
				    graph_a= list(mlna),
				    save= True
				)
			
			## default - MLNa
			
			is_full = list(set(col_list) - set(list(mlna)))
			print(f"{col_list} + {mlna} + {is_full}")
			if len(is_full) > 1: #  check if operation is possible
				#print(typeof(mlna))

				VALUE_MINUS_MLNa = value_clfs.drop(list(mlna), axis=1)

				logic_i_g.append(
					make_builder(
						fix_imbalance=fix_imbalance, 
						DATA_OVER=VALUE_MINUS_MLNa, 
						target_variable=target_variable, 
						clfs=clfs, 
						domain= f'classic_-_mlna_{case_k}', 
						prefix=domain,
						verbose=verbose,
						cwd= cwd+f'/mlna_{k}'
						)
					)
				logic_i_p.append(logic_i_g[-1])
				# classic - mln attribut
				plot_features_importance_as_barh(
					    data= logic_i_g[-1][1],
					    getColor= custom_color,
					    modelDictName= modelD,
					    plotTitle= f"Classic - mln attributs features importance for_{case_k}",
					    prefix= domain, 
					    cwd= cwd+f'/mlna_{k}',
					    graph_a= list(mlna),
					    save= True
					)
				#print(f"{typeof(mlna)} {mmlna}")

			## default + MLN - MLNa
			VALUE_MLN_MINUS_MLN_P = VALUE_MLN_P.drop(list(mlna), axis=1)
			VALUE_MLN_MINUS_MLN_G = VALUE_MLN_G.drop(list(mlna), axis=1)

			logic_i_p.append(
				make_builder(
					fix_imbalance=fix_imbalance, 
					DATA_OVER=VALUE_MLN_MINUS_MLN_P, 
					target_variable=target_variable, 
					clfs=clfs, 
					domain= f'classic_mln_-_mlna_{case_k}', 
					prefix=domain,
					verbose=verbose,
					cwd= cwd+f'/mlna_{k}/personalized'
					)
				)
			logic_i_g.append(
				make_builder(
					fix_imbalance=fix_imbalance, 
					DATA_OVER=VALUE_MLN_MINUS_MLN_G, 
					target_variable=target_variable, 
					clfs=clfs, 
					domain= f'classic_mln_-_mlna_{case_k}', 
					prefix=domain,
					verbose=verbose,
					cwd= cwd+f'/mlna_{k}/global'
					)
				)
			# classic + mln - mln attribut
			plot_features_importance_as_barh(
				    data= logic_i_p[-1][1],
				    getColor= custom_color,
				    modelDictName= modelD,
				    plotTitle= f"Classic + mln - mln attributs features importance for_{case_k}",
				    prefix= domain, 
				    cwd= cwd+f'/mlna_{k}/personalized',
				    graph_a= list(mlna),
				    save= True
				)
			plot_features_importance_as_barh(
				    data= logic_i_g[-1][1],
				    getColor= custom_color,
				    modelDictName= modelD,
				    plotTitle= f"Classic + mln - mln attributs features importance for_{case_k}",
				    prefix= domain, 
				    cwd= cwd+f'/mlna_{k}/global',
				    graph_a= list(mlna),
				    save= True
				)
			
			
			
			
			

			# print html report for analysis of variable i
			# logic_i=[default,*logic_i]
			logic_p = [*logic_p, *logic_i_p]
			logic_g = [*logic_g, *logic_i_g]
			table_g = print_summary([default,*logic_i_g],modelD)
			table_p = print_summary([default,*logic_i_p],modelD)
			# del logic_p
			# del logic_g
			create_file(
				content= table_g[1], 
				cwd= cwd+f'/mlna_{k}/global', 
				prefix= domain, 
				filename= f"mlna_for_{case_k}", 
				extension=".html"
				)
			create_file(
				content= table_p[1], 
				cwd= cwd+f'/mlna_{k}/personalized', 
				prefix= domain, 
				filename= f"mlna_for_{case_k}", 
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
		table_p = print_summary([default,*logic_p],modelD)
		table_g = print_summary([default,*logic_g],modelD)
		create_file(
			content= table_p[1], 
			cwd= cwd+f'/mlna_{k}/personalized', 
			prefix= domain, 
			filename= f"mlna_for_all_{k}_combination_categorial_data", 
			extension=".html"
			)
		create_file(
			content= table_g[1], 
			cwd= cwd+f'/mlna_{k}/global', 
			prefix= domain, 
			filename= f"mlna_for_all_{k}_combination_categorial_data", 
			extension=".html"
			)
		modelD = None
		PERSONS = None
		table_p = None
		table_g = None
	# return logic

# @profile
def make_builder(fix_imbalance, DATA_OVER, target_variable, clfs, cwd, prefix, verbose, domain= 'classic'):
	"""
	"""

	# test train split
	x_train, x_test, y_train, y_test = test_train(DATA_OVER,target_variable)
	# fix inbalance state of classes
	if fix_imbalance:
		x_train, y_train= get_SMOTHE_dataset(
			X= x_train,
			y= y_train,
			random_state= 42, 
			# sampling_strategy= "minority"
			)
	else:
		DATA_OVER= DATA_OVER.copy(deep=True)

	# get evaluation storage structure
	STORE_STD = init_training_store(x_train)

	# run classic ML on default data
	store=train(
		clfs= clfs,
		x_train=x_train,
		y_train=y_train,
		x_test=x_test,
		y_test=y_test, 
		store= STORE_STD, 
		domain= domain, 
		prefix= prefix, 
		cwd= cwd
		)
	link_to_original = save_dataset(
		cwd= cwd, 
		dataframe= store, 
		name= f'{domain}_metric', 
		prefix= prefix, 
		sep= ','
		)
	save_dataset(
		cwd= cwd, 
		dataframe= x_train, 
		name= f'{domain}_x_train', 
		prefix= prefix, 
		sep= ','
		)
	save_dataset(
		cwd= cwd, 
		dataframe= x_test, 
		name= f'{domain}_x_test', 
		prefix= prefix, 
		sep= ','
		)
	save_dataset(
		cwd= cwd, 
		dataframe= y_train, 
		name= f'{domain}_y_train', 
		prefix= prefix, 
		sep= ','
		)
	save_dataset(
		cwd= cwd, 
		dataframe= y_test, 
		name= f'{domain}_y_test', 
		prefix= prefix, 
		sep= ','
		)
	return (domain, store)

# @profile
def mlnaPipeline(cwd, domain, dataset_link, target_variable, dataset_delimiter=',', all_nominal=True, all_numeric=False, verbose=True, fix_imbalance=True, levels=None, to_remove=None, encoding="utf-8",index_col=None, na_values=None, alphas=[0.85]):
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
	dataset = load_data_set_from_url(path=dataset_link,sep=dataset_delimiter, encoding=encoding,index_col=index_col, na_values=na_values)
	dataset.reset_index(drop=True, inplace=True)
	print(f"loaded dataset dim: {dataset.shape}") if verbose else None


	# eda
	#print(dataset.shape)
	dataset = dataset.sample(int(dataset.shape[0]*.25))
	dataset.reset_index(drop=True, inplace=True)
	#print(dataset.shape)
	# dataset = make_eda(dataframe=dataset, verbose=verbose)
	dataset = make_eda(dataframe=dataset, verbose=verbose)
	#print(dataset.shape)
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
	promise = make_preprocessing(
		dataset=dataset, 
		to_remove=to_remove, 
		domain=domain, 
		cwd=cwd+f"/outputs/{domain}", 
		target_variable=target_variable, 
		verbose=verbose,
		levels=levels 
		)
	#print(promise[7].shape)
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
	if sum(['classic_metric' in file for _,_,files in os.walk(f"{cwd}/outputs/{domain}/data_selection_storage") for file in files]) == 0:
		default = make_builder(
			fix_imbalance=fix_imbalance, 
			DATA_OVER=promise[7], 
			target_variable=target_variable, 
			clfs=clfs, 
			domain= 'classic', 
			prefix=domain, 
			verbose=verbose,
			cwd= cwd+f"/outputs/{domain}"
			)
	else:
		name = [file for _,_,files in os.walk(f"{cwd}/outputs/{domain}/data_selection_storage") for file in files if 'classic_metric' in file][0]
		default = (
			'classic', 
			load_data_set_from_url(
				path=f"{cwd}/outputs/{domain}/data_selection_storage/{name}"
				,sep=dataset_delimiter
				,encoding=encoding
				,index_col=0
				,na_values=na_values
				)
			)
		print(default)
		print("load classic training")


	for alpha in alphas:
		if (2 in levels) and (len(promise[8]) > 0): # if this stage is allowed
			
			# global_logic =[*
			make_mlna_1_variable(
				default=default, 
				value_graph=promise[7], 
				value_clfs=promise[7],
				OHE=promise[8], 
				nominal_factor_colums=promise[2], 
				cwd=cwd+f"/outputs/{domain}/{alpha}/qualitative", 
				domain=domain, 
				fix_imbalance=fix_imbalance, 
				target_variable=target_variable, 
				custom_color=custom_color, 
				modelD=modelD, 
				verbose=verbose, 
				clfs=clfs,
				alpha=alpha
				)
			# ]

		# 2) build an MLN on just k variable, 1 < k <= len of OHE
		if (3 in levels)  and (len(promise[8]) > 0):
			# global_logic =[*global_logic, *
			make_mlna_k_variable(
				default=default, 
				value_graph=promise[7], 
				value_clfs=promise[7],
				OHE=promise[8], 
				nominal_factor_colums=sorted(promise[2]), 
				cwd=cwd+f"/outputs/{domain}/{alpha}/qualitative", 
				domain=domain, 
				fix_imbalance=fix_imbalance, 
				target_variable=target_variable, 
				custom_color=custom_color, 
				modelD=modelD, 
				verbose=verbose, 
				clfs=clfs,
				alpha=alpha
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
		
		if (4 in levels)  and (len(promise[10]) > 0): # if this stage is allowed
			
			# global_logic =[*global_logic, *
			make_mlna_1_variable(
				default=default, 
				value_graph=promise[9], 
				value_clfs=promise[7], 
				OHE=promise[10], 
				nominal_factor_colums=list(set([*promise[5], *promise[6]])-set([target_variable])), 
				cwd=cwd+f"/outputs/{domain}/{alpha}/quantitative", 
				domain=domain, 
				fix_imbalance=fix_imbalance, 
				target_variable=target_variable, 
				custom_color=custom_color, 
				modelD=modelD, 
				verbose=verbose, 
				clfs=clfs,
				alpha=alpha
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
				nominal_factor_colums=sorted(list(set([*promise[5], *promise[6]])-set([target_variable]))), 
				cwd=cwd+f"/outputs/{domain}/{alpha}/quantitative", 
				domain=domain, 
				fix_imbalance=fix_imbalance, 
				target_variable=target_variable, 
				custom_color=custom_color, 
				modelD=modelD, 
				verbose=verbose, 
				clfs=clfs,
				alpha=alpha
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
				OHE=[*promise[8],*promise[10]], 
				nominal_factor_colums=sorted(list(set([*promise[4],*promise[5], *promise[6]])-set([target_variable]))), 
				cwd=cwd+f"/outputs/{domain}/{alpha}/mixed", 
				domain=domain, 
				fix_imbalance=fix_imbalance, 
				target_variable=target_variable, 
				custom_color=custom_color, 
				modelD=modelD, 
				verbose=verbose, 
				clfs=clfs,
				alpha=alpha
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
	
