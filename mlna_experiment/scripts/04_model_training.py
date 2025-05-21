# 04_model_training.py

import sys
# Ajoutez le répertoire parent pour pouvoir importer les modules
sys.path.append('..')  # Ajouter le répertoire parent au chemin de recherche des modules

from modules.modeling import *  # Fonctions d'entraînement
from modules.preprocessing import *  # Preprocessing functions
from modules.file import *  # File manipulation functions
from modules.graph import *  # Modeling functions
import statistics
import ast


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
    save_dataset(
        cwd=cwd,
        dataframe=store,
        name=f'{domain}_metric',
        prefix=prefix,
        sep=',',
        sub='evaluation'
    )
    save_dataset(
        cwd=cwd,
        dataframe=x_train,
        name=f'{domain}_x_train',
        prefix=prefix,
        sep=',',
        sub='evaluation'
    )
    save_dataset(
        cwd=cwd,
        dataframe=x_test,
        name=f'{domain}_x_test',
        prefix=prefix,
        sep=',',
        sub='evaluation'
    )
    save_dataset(
        cwd=cwd,
        dataframe=y_train,
        name=f'{domain}_y_train',
        prefix=prefix,
        sep=',',
        sub='evaluation'
    )
    save_dataset(
        cwd=cwd,
        dataframe=y_test,
        name=f'{domain}_y_test',
        prefix=prefix,
        sep=',',
        sub='evaluation'
    )
    return (domain, store)

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
            cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
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
                cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
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
                cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
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
            cwd=cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}',
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
            cwd=cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}',
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
                cwd=cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}',
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
                cwd=cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}',
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
            cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
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
                cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
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
                cwd=cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}',
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
            cwd=cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}',
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
            cwd=cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}',
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
                cwd=cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}',
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
                cwd=cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}',
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
        verbose,
        clfs,
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
        # name=f'{nominal_factor_colums[i]}_mln'
        # load config df file
        config_df_path = args.cwd + f'/mlna_1/{nominal_factor_colums[i]}/' + [file for _, _, files in
              os.walk(
                  args.cwd + f'/mlna_1/{nominal_factor_colums[i]}/')
              for
              file in files][
            [f'config_df_for_{name}_{"withClass" if graphWithClass else "withoutClass"}' in file for _, _, files in
             os.walk(args.cwd + f'/mlna_1/{nominal_factor_colums[i]}/') for
             file in files].index(True)
        ]
        config_df = read_model(path=config_df_path)
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
        table_g = None
        table_p = None
        logic_i_g = None
        logic_i_p = None
        mlc_cf = None

    logic_g = None
    logic_p = None
    logic_pg = None
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

def main():
    # Définition des arguments
    import argparse

    parser = argparse.ArgumentParser(description='Exécution du pipeline MLNA - SPLIT DATASET')
    parser.add_argument('--cwd', type=str, required=True, help='Répertoire de travail courant')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Nom du dataset')
    parser.add_argument('--alpha', type=float, required=True, help='Valeur d\'alpha')
    parser.add_argument('--turn', type=int, required=True, help='Valeur du tour')

    # Récupération des arguments
    args = parser.parse_args()




    config = load_config(f"{args.cwd}/configs/{args.dataset_folder}/config.ini")

    domain = config["DATA"]["domain"]

    encoding = config["PREPROCESSING"]["encoding"]
    dataset_delimiter = config["DATA"]["dataset_delimiter"]
    target_variable = config["DATA"]["target"]

    cost = config.getboolean("TRAINING", "cost")
    duration_divider = config.getint("TRAINING", "duration_divider")
    rate_divider = config.getint("TRAINING", "rate_divider")
    financialOption = config["TRAINING"]["financialOption"]

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
    if sum([f'original' in file for _, _, files in
         os.walk(args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}') for
         file in files]) == 0 and cost is True:
        print("❌ Unable to access original form of dataset")
        exit(1)
    # load the dedicated work orginal dataset
    original_path = args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}' + [file for _, _, files in
       os.walk(
           args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}')
       for
       file in files][
        [f'original' in file for _, _, files in
         os.walk(args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}') for
         file in files].index(True)
    ]
    original = read_model(path=original_path)

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
    # default training

    # ------------------------------------------------------------------------------------------------------------------
    if args.turn == 1: # check if we are onto the first turn
        if sum(['model_turn_1_completed' in file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_1') for
                file in files]) > 0:
            print("✅ MLNA 1 Modeling already completed")
        else:
            (outperformers, NbGood) = make_mlna_1_variable_v2(
                x_traini=x_traini,
                x_testi=x_testi,
                y_traini=y_traini,
                y_testi=y_testi,
                OHE=OHE,
                nominal_factor_colums=columns,
                cwd = args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}',
                domain= domain,
                target_variable= target_variable,
                graphWithClass=False,
                fix_imbalance=False,
                withCost=cost,
                financialOption = ast.literal_eval(financialOption) if cost else None,
                duration_divider = duration_divider,
                rate_divider= rate_divider,
                original=original,
                default=None,
            )
            outperformers = dict(sorted(outperformers.items(), key=lambda x: x[1], reverse=True))
            bestK = bestThreshold(list(outperformers)) + 1 if len(outperformers) > 2 else len(outperformers)
            print(f"{outperformers}, {NbGood} Goods and the best top k is {bestK}")
            save_model(
                cwd=args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}',
                clf={
                    'model': outperformers,
                    'nbGood': NbGood,
                    'bestK': bestK,
                    "name": [prepro_config["categorial_col"][i] for i in list(outperformers.keys())],
                },
                prefix="MNIFS",
                clf_name=f"{domain}_best_features",
                sub=""
            )
            contenu = f"""
                            'model': {outperformers},
                            'nbGood': {NbGood},
                            'bestK': {bestK},
                            "name":{[prepro_config[2][i] for i in list(outperformers.keys())]},
                            "BName":{[prepro_config[2][i] for i in list(outperformers.keys())[:bestK]]},
                            """
            with open(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_1/model_turn_1_completed.dtvni', "a") as fichier:
                fichier.write(contenu)

        if sum(['model_turn_1_completed' in file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_2/') for
                file in files]) > 0:
            print("✅ COMBINATORY MLNA 2 Model  already completed")
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
                    args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_2/graph_turn_1_completed.dtvni',
                    "a") as fichier:
                fichier.write("")
    if args.turn == 2:  # check if we are onto the first turn
        make_mlna_top_k_variable_v2(
            x_traini=x_traini,
            x_testi=x_testi,
            y_traini=y_traini,
            y_testi=y_testi,
            OHE=OHE,
            nominal_factor_colums=columns,
            cwd=args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}',
            root=args.cwd,
            domain=domain,
            target_variable=target_variable,
            alpha=args.alpha,
            graphWithClass=False,
            topR=[]
        )


    print("Descripteurs extraits et sauvegardés.")

if __name__ == "__main__":
    main()