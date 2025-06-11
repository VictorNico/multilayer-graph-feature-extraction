# 01_data_preprocessing.py


import sys
# Ajoutez le répertoire parent pour pouvoir importer les modules
sys.path.append('..')  # Ajoute le répertoire parent au chemin de recherche des modules
from modules.preprocessing import *  # Preprocessing functions
from modules.file import *  # File manipulation functions
from modules.eda import *  # Exploratory Data Analysis (EDA) functions


#################################################
##          Methods definition
#################################################
def remove_rare_classes(df, target_column, min_count=10):
    class_counts = df[target_column].value_counts()
    rare_classes = class_counts[class_counts < min_count].index
    print(f"[INFO] Classes supprimées (trop rares) : {list(rare_classes)}")

    return df[~df[target_column].isin(rare_classes)].reset_index(drop=True)

# @profile
def make_eda(dataframe, cwd, dir_name, verbose,processed_dir):
    """Lookup NAs values in the dataset and apply a preprocessing on it

    Args:
        dataframe:
        cwd:
        dir_name:
        verbose:

    Returns:
        A new dataframe without NA values
    """
    df = dataframe.copy(deep=True)
    if isinstance(dataframe, pd.DataFrame):
        is_na_columns = get_na_columns(dataframe)
        print(f"{is_na_columns}") if verbose else None
        # save the list of initial NA columns
        save_model(
            cwd=cwd + f'/{processed_dir}{dir_name}',
            clf={
                "is_na_columns": is_na_columns
            },
            prefix="",
            clf_name="na_columns",
            ext=".conf",
            sub=""
        ) if len(is_na_columns) > 0 else None
        # treat them
        if len(is_na_columns) > 0:
            df = impute_nan_values(dataframe, is_na_columns)
            print(f"{get_na_columns(df)}") if verbose else None

    return df


# @profile
def make_preprocessing(dataset, to_remove, domain, cwd, dir_name, target_variable, verbose,processed_dir):
    """Apply some framework preprocessing like, OHE, LableEncoding, normalisation, discretization

    Args:
        dataset:
        to_remove:
        domain:
        cwd:
        dir_name:
        target_variable:
        verbose
        levels

    Returns:
        A tuple of parameters which are need to nexted framework execution
    """

    # delete attribut which are not helpfull
    dataset.drop(to_remove, axis=1, inplace=True)
    before = dataset[target_variable].value_counts()
    dataset = remove_rare_classes(dataset, target_variable, min_count=5)
    after = dataset[target_variable].value_counts()
    save_model(
        cwd=cwd + f'/{processed_dir}{dir_name}',
        clf={
            "before": before,
            "after": after
        },
        prefix="",
        clf_name="remove_rare_classes",
        ext=".conf",
        sub=""
    )

    # get list of numeric, ordinal and nominal dimension in dataset

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
	numeric_uniform_columns:{numeric_uniform_colums}
		""") if verbose else None

    # save the list of column group
    save_model(
        cwd=cwd + f'/{processed_dir}{dir_name}',
        clf={
            "col_list": col_list,
            "numeric_col": numeric_col,
            "categorial_col": categorial_col,
            "ordinal_factor_colums": ordinal_factor_colums,
            "nominal_factor_colums": nominal_factor_colums,
            "numeric_with_outliers_columns": numeric_with_outliers_columns,
            "numeric_uniform_columns": numeric_uniform_colums,
        },
        prefix="",
        clf_name="column_group",
        ext=".conf",
        sub=""
    )

    ## apply preprocessing treatments
    _ALL_PRETRAETED_DATA = {}
    dataset_original = dataset.copy(deep=True)  # save a backup of our dataset

    # binarise nominals factors
    DATA_OHE, OHE = nominal_factor_encoding(
        dataset,
        categorial_col if not (target_variable in categorial_col) else list(
            set(categorial_col) - {target_variable})
    ) if len(categorial_col) > 0 else (dataset, [])
    DATA_OHE = make_eda(dataframe=DATA_OHE, cwd=cwd, dir_name=dir_name, verbose=verbose,processed_dir=processed_dir)
    print(f"OHE <----> {OHE}") if verbose else None

    # label encoding of ordinal data
    # if (target_variable in categorial_col):
    DATA_OHE = remove_rare_classes(DATA_OHE, target_column=target_variable, min_count=5)
    DATA_OHE_LB, label_enc = (ordinal_factor_encoding(
        DATA_OHE,
        [target_variable]
    ) if (target_variable in categorial_col) else (DATA_OHE, None))
    save_model(
        cwd=cwd + f'/{processed_dir}{dir_name}',
        clf={
            "label_encoded": label_enc
        },
        prefix="",
        clf_name="label_encoded",
        ext=".conf",
        sub=""
    ) if (target_variable in categorial_col) else None


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
            set(numeric_uniform_colums) - {target_variable})
    ) if len(numeric_uniform_colums) > 0 else DATA_OHE_LB

    print(f"{get_na_columns(DATA_OHE_LB_LBU_STDU)}") if verbose else None

    DATA_OHE_LB_LBU_STDU = make_eda(dataframe=DATA_OHE_LB_LBU_STDU, cwd=cwd, dir_name=dir_name, verbose=verbose,processed_dir=processed_dir)


    # normalisation of numeric data with outliers to deeve it into interval 0,1
    DATA_OHE_LB_LBU_STDU_STDWO = numeric_standardization_with_outliers(
        DATA_OHE_LB_LBU_STDU,
        numeric_with_outliers_columns if not (target_variable in numeric_with_outliers_columns) else list(
            set(numeric_with_outliers_columns) - {target_variable})
    ) if len(numeric_with_outliers_columns) > 0 else DATA_OHE_LB_LBU_STDU

    print(f"{get_na_columns(DATA_OHE_LB_LBU_STDU_STDWO)}") if verbose else None

    DATA_OHE_LB_LBU_STDU_STDWO = make_eda(dataframe=DATA_OHE_LB_LBU_STDU_STDWO, cwd=cwd, dir_name=dir_name, verbose=verbose,processed_dir=processed_dir)

    # save the first one where just all factor are binarized
    # _ALL_PRETRAETED_DATA['withAllFactorsVBinarized']= DATA_OHE_LB_LBU_STDU_STDWO.copy(deep=True)
    print(f"{get_na_columns(DATA_OHE_LB_LBU_STDU_STDWO)}") if verbose else None
    # save preprocessed 1st one data
    link_to_preprocessed_factor_data = save_dataset(
        cwd=cwd + f'/{processed_dir}{dir_name}',
        dataframe=DATA_OHE_LB_LBU_STDU_STDWO,
        name=f'{domain}_preprocessed',
        # prefix= domain,
        sep=',',
        sub="/cat"
    )
    #
    # if (4 in levels) or (5 in levels):
    #     # discretize nuneric value
    DATA_DISCRETIZE = discretise_numeric_dimension_by_entropy(
        bins=3,
        threshold=.00000001,
        target=target_variable,
        columns=list({*numeric_with_outliers_columns, *numeric_uniform_colums} - {target_variable}),
        dataframe=DATA_OHE_LB_LBU_STDU_STDWO,
        inplace=False,
        verbose=verbose
    ) if len(list({*numeric_with_outliers_columns, *numeric_uniform_colums} - {
        target_variable})) > 0 else DATA_OHE_LB_LBU_STDU_STDWO

    link_to_preprocessed_disc_data = save_dataset(
        cwd=cwd + f'/{processed_dir}{dir_name}',
        dataframe=DATA_DISCRETIZE,
        name=f'{domain}_preprocessed_discretize',
        # prefix= domain,
        sep=',',
        sub="/num"
    )

    print(f"{get_na_columns(DATA_DISCRETIZE)}") if verbose else None

    DATA_DISCRETIZE = make_eda(dataframe=DATA_DISCRETIZE, cwd=cwd, dir_name=dir_name, verbose=verbose,processed_dir=processed_dir)

    # binarise nominals factors
    DATA_DISCRETIZE_OHE_2, OHE_2 = nominal_factor_encoding(
        DATA_DISCRETIZE,
        list({*numeric_with_outliers_columns, *numeric_uniform_colums} - {target_variable})
    ) if len(
        list({*numeric_with_outliers_columns, *numeric_uniform_colums} - {target_variable})) > 0 else (
        DATA_DISCRETIZE, [])

    print(f"{get_na_columns(DATA_DISCRETIZE_OHE_2)}") if verbose else None

    DATA_DISCRETIZE_OHE_2 = make_eda(dataframe=DATA_DISCRETIZE_OHE_2, cwd=cwd, dir_name=dir_name, verbose=verbose,processed_dir=processed_dir)

    link_to_preprocessed_all_data = save_dataset(
        cwd=cwd + f'/{processed_dir}{dir_name}',
        dataframe=DATA_DISCRETIZE_OHE_2,
        name=f'{domain}_preprocessed',
        # prefix= domain,
        sep=',',
        sub="/num"
    )
    print(f"discretize data shape: {DATA_DISCRETIZE_OHE_2.shape}") if verbose else None
    print(f"{get_na_columns(DATA_DISCRETIZE_OHE_2)}") if verbose else None
    # save the second one where all variables are binarized
    # _ALL_PRETRAETED_DATA['withAllVariablesBinarized']= DATA_DISCRETIZE_OHE_2.copy(deep=True)

    save_model(
        cwd=cwd + f'/{processed_dir}{dir_name}',
        clf={
            "col_list": col_list,
            "numeric_col":numeric_col,
            "categorial_col":categorial_col if not (target_variable in categorial_col) else list(
                set(categorial_col) - {target_variable}),
            "ordinal_factor_colums":ordinal_factor_colums,
            "nominal_factor_colums":nominal_factor_colums,
            "numeric_with_outliers_columns":numeric_with_outliers_columns,
            "numeric_uniform_colums":numeric_uniform_colums,
            "DATA_OHE_LB_LBU_STDU_STDWO":DATA_OHE_LB_LBU_STDU_STDWO,
            "OHE":OHE,
            "DATA_DISCRETIZE_OHE_2":DATA_DISCRETIZE_OHE_2,
            "OHE_2":OHE_2
        },
        prefix="",
        clf_name="preprocessing",
        ext=".conf",
        sub=""
    )



def main():
    # Définition des arguments
    import argparse

    parser = argparse.ArgumentParser(description='Exécution du pipeline MLNA - PREPROCESSING')
    parser.add_argument('--cwd', type=str, required=True, help='Répertoire de travail courant')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Nom du dataset')

    # Récupération des arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    config = load_config(f"{args.cwd}/configs/{args.dataset_folder}/config.ini")

    raw_path = config["DATA"]["raw_path"]
    target_variable = config["DATA"]["target"]
    processed_dir = config["GENERAL"]["processed_dir"]
    domain = config["DATA"]["domain"]
    dataset_delimiter = config["DATA"]["dataset_delimiter"]
    index_col = None if config["DATA"]["index_col"] in ["None",""] else config.getint("DATA","index_col")
    na_values = config["DATA"]["na_values"]

    target_columns_type = config["GENERAL"]["target_columns_type"]
    encoding = config["PREPROCESSING"]["encoding"]
    to_remove = config["PREPROCESSING"]["to_remove"].split(',') if config["PREPROCESSING"]["to_remove"].split(',')[0] != '' else []
    portion = config.getfloat("PREPROCESSING","portion")

    verbose = config.getboolean("GENERAL", "verbose")

    # ------------------------------------------------------------------------------------------------------------------
    if sum(['preprocessing' in file for _, _, files in
            os.walk(args.cwd + f'/{processed_dir}{args.dataset_folder}') for
            file in files]) > 0:
        print("✅ Stage already completed")
        exit(0)

    print(f"Current working directory: {args.cwd}") if verbose else None

    # load the dedicated work dataset
    print(f"received path: {raw_path}") if verbose else None
    dataset = load_data_set_from_url(path=f"{args.cwd}/{raw_path}", sep=dataset_delimiter, encoding=encoding, index_col=index_col,
                                     na_values=na_values)
    print(dataset.columns)
    print(dataset[target_variable].value_counts())
    dataset.reset_index(drop=True, inplace=True)
    print(f"loaded dataset dim: {dataset.shape}") if verbose else None

    # eda
    # print(dataset.shape)
    dataset = random_sample_merge_v2(df=dataset, target_column=target_variable, percentage=portion)
    dataset.reset_index(drop=True, inplace=True)

    print(f"loaded dataset dim: {dataset.shape}") if verbose else None
    dataset = make_eda(dataframe=dataset, cwd=args.cwd, dir_name=args.dataset_folder, verbose=verbose,processed_dir=processed_dir)
    dataset_copy = dataset.copy(deep=True)

    save_model(
        cwd=args.cwd + f'/{processed_dir}{args.dataset_folder}',
        clf=dataset_copy,
        prefix="",
        clf_name="original",
        sub=f""
    )
    # identify columns with unique value to remove
    unique_cols = dataset.columns[dataset.nunique() <= 1].tolist()

    # Application de la préparation

    make_preprocessing(
        dataset=dataset,
        to_remove=[*unique_cols,*to_remove],
        domain=domain,
        cwd=args.cwd,
        dir_name=args.dataset_folder,
        target_variable=target_variable,
        verbose=verbose,
        processed_dir=processed_dir
    )
    print("✅ Preprocessing completed.") if verbose else None


if __name__ == "__main__":
    main()
