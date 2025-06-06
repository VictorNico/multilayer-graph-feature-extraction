# 04_model_training.py

import sys
# Ajoutez le répertoire parent pour pouvoir importer les modules
sys.path.append('..')  # Ajouter le répertoire parent au chemin de recherche des modules

from modules.modeling import *  # Fonctions d'entraînement
from modules.preprocessing import *  # Preprocessing functions
from modules.file import *  # File manipulation functions
from modules.graph import *  # Modeling functions
from modules.report import *  # Report functions
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
        sub='/evaluation'
    )
    save_dataset(
        cwd=cwd,
        dataframe=x_train,
        name=f'{domain}_x_train',
        prefix=prefix,
        sep=',',
        sub='/evaluation'
    )
    save_dataset(
        cwd=cwd,
        dataframe=x_test,
        name=f'{domain}_x_test',
        prefix=prefix,
        sep=',',
        sub='/evaluation'
    )
    save_dataset(
        cwd=cwd,
        dataframe=y_train,
        name=f'{domain}_y_train',
        prefix=prefix,
        sep=',',
        sub='/evaluation'
    )
    save_dataset(
        cwd=cwd,
        dataframe=y_test,
        name=f'{domain}_y_test',
        prefix=prefix,
        sep=',',
        sub='/evaluation'
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

    if sum([f'MlC_PER_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
            os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}') for
            file in files]) == 0:
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
    else:
        res_path = cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
            os.walk(
                cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}')
            for
            file in files][
            [f'MlC_PER_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
             os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}') for
             file in files].index(True)
        ]
        res = (f'MlC_PER_{"C" if graphWithClass is True else "M"}X_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                     index_col=0,
                                     na_values=None))
        logic_i_p.append(res)
    if graphWithClass is True:
        if sum([f'MlC_PER_{"C" if graphWithClass is True else "M"}Y_for_{name}_metric' in file for _, _, files in
                os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}') for
                file in files]) == 0:
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
        else:
            res_path = cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
                os.walk(
                    cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files][
                [f'MlC_PER_{"C" if graphWithClass is True else "M"}Y_for_{name}_metric' in file for _, _, files in
                 os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}') for
                 file in files].index(True)
            ]
            res = (f'MlC_PER_{"C" if graphWithClass is True else "M"}Y_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                         index_col=0,
                                         na_values=None))
            logic_i_p.append(res)
        if sum([f'MlC_PER_{"C" if graphWithClass is True else "M"}XY_for_{name}_metric' in file for _, _, files in
                os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files]) == 0:
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
        else:
            res_path = cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
                os.walk(
                    cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files][
                [f'MlC_PER_{"C" if graphWithClass is True else "M"}XY_for_{name}_metric' in file for _, _, files in
                 os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}') for
                 file in files].index(True)
            ]
            res = (f'MlC_PER_{"C" if graphWithClass is True else "M"}XY_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                         index_col=0,
                                         na_values=None))
            logic_i_p.append(res)
    if sum([f'MlC_GLO_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
            os.walk(cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}')
            for
            file in files]) == 0:
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
    else:
        res_path = cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
            os.walk(
                cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}')
            for
            file in files][
            [f'MlC_GLO_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
             os.walk(cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}') for
             file in files].index(True)
        ]
        res = (f'MlC_GLO_{"C" if graphWithClass is True else "M"}X_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                     index_col=0,
                                     na_values=None))
        logic_i_p.append(res)
    if sum([f'MlC_GAP_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
            os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
            for
            file in files]) == 0:
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
    else:
        res_path = cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
            os.walk(
                cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
            for
            file in files][
            [f'MlC_GAP_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
             os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}') for
             file in files].index(True)
        ]
        res = (f'MlC_GAP_{"C" if graphWithClass is True else "M"}X_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                     index_col=0,
                                     na_values=None))
        logic_i_p.append(res)
    if graphWithClass is True:
        if sum([f'MlC_GAP_{"C" if graphWithClass is True else "M"}Y_for_{name}_metric' in file for _, _, files in
                os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files]) == 0:
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
        else:
            res_path = cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
                os.walk(
                    cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files][
                [f'MlC_GAP_{"C" if graphWithClass is True else "M"}Y_for_{name}_metric' in file for _, _, files in
                 os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}') for
                 file in files].index(True)
            ]
            res = (f'MlC_GAP_{"C" if graphWithClass is True else "M"}Y_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                         index_col=0,
                                         na_values=None))
            logic_i_p.append(res)
        if sum([f'MlC_GAP_{"C" if graphWithClass is True else "M"}XY_for_{name}_metric' in file for _, _, files in
                os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files]) == 0:
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
        else:
            res_path = cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
                os.walk(
                    cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files][
                [f'MlC_GAP_{"C" if graphWithClass is True else "M"}XY_for_{name}_metric' in file for _, _, files in
                 os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}') for
                 file in files].index(True)
            ]
            res = (f'MlC_GAP_{"C" if graphWithClass is True else "M"}XY_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                         index_col=0,
                                         na_values=None))
            logic_i_p.append(res)

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

    if sum([f'MCA_PER_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
            os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}') for
            file in files]) == 0:
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
    else:
        res_path = cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
            os.walk(
                cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}')
            for
            file in files][
            [f'MCA_PER_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
             os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}') for
             file in files].index(True)
        ]
        res = (f'MCA_PER_{"C" if graphWithClass is True else "M"}X_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                     index_col=0,
                                     na_values=None))
        logic_i_p.append(res)
    if graphWithClass is True:
        if sum([f'MCA_PER_{"C" if graphWithClass is True else "M"}Y_for_{name}_metric' in file for _, _, files in
                os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}') for
                file in files]) == 0:
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
        else:
            res_path = cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
                os.walk(
                    cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files][
                [f'MCA_PER_{"C" if graphWithClass is True else "M"}Y_for_{name}_metric' in file for _, _, files in
                 os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}') for
                 file in files].index(True)
            ]
            res = (f'MCA_PER_{"C" if graphWithClass is True else "M"}Y_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                         index_col=0,
                                         na_values=None))
            logic_i_p.append(res)
        if sum([f'MCA_PER_{"C" if graphWithClass is True else "M"}XY_for_{name}_metric' in file for _, _, files in
                os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files]) == 0:
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
        else:
            res_path = cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
                os.walk(
                    cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files][
                [f'MCA_PER_{"C" if graphWithClass is True else "M"}XY_for_{name}_metric' in file for _, _, files in
                 os.walk(cwd + f'{mlnL}/{name}/personalized{"/withClass" if graphWithClass else "/withoutClass"}') for
                 file in files].index(True)
            ]
            res = (f'MCA_PER_{"C" if graphWithClass is True else "M"}XY_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                         index_col=0,
                                         na_values=None))
            logic_i_p.append(res)
    if sum([f'MCA_GLO_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
            os.walk(cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}')
            for
            file in files]) == 0:
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
    else:
        res_path = cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
            os.walk(
                cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}')
            for
            file in files][
            [f'MCA_GLO_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
             os.walk(cwd + f'{mlnL}/{name}/global{"/withClass" if graphWithClass else "/withoutClass"}') for
             file in files].index(True)
        ]
        res = (f'MCA_GLO_{"C" if graphWithClass is True else "M"}X_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                     index_col=0,
                                     na_values=None))
        logic_i_p.append(res)
    if sum([f'MCA_GAP_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
            os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
            for
            file in files]) == 0:
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
    else:
        res_path = cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
            os.walk(
                cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
            for
            file in files][
            [f'MCA_GAP_{"C" if graphWithClass is True else "M"}X_for_{name}_metric' in file for _, _, files in
             os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}') for
             file in files].index(True)
        ]
        res = (f'MCA_GAP_{"C" if graphWithClass is True else "M"}X_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                     index_col=0,
                                     na_values=None))
        logic_i_p.append(res)
    if graphWithClass is True:
        if sum([f'MCA_GAP_{"C" if graphWithClass is True else "M"}Y_for_{name}_metric' in file for _, _, files in
                os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files]) == 0:
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
        else:
            res_path = cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
                os.walk(
                    cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files][
                [f'MCA_GAP_{"C" if graphWithClass is True else "M"}Y_for_{name}_metric' in file for _, _, files in
                 os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}') for
                 file in files].index(True)
            ]
            res = (f'MCA_GAP_{"C" if graphWithClass is True else "M"}Y_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                         index_col=0,
                                         na_values=None))
            logic_i_p.append(res)
        if sum([f'MCA_GAP_{"C" if graphWithClass is True else "M"}XY_for_{name}_metric' in file for _, _, files in
                os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files]) == 0:
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
        else:
            res_path = cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}/evaluation/' + [file for _, _, files in
                os.walk(
                    cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}')
                for
                file in files][
                [f'MCA_GAP_{"C" if graphWithClass is True else "M"}XY_for_{name}_metric' in file for _, _, files in
                 os.walk(cwd + f'{mlnL}/{name}/mixed{"/withClass" if graphWithClass else "/withoutClass"}') for
                 file in files].index(True)
            ]
            res = (f'MCA_GAP_{"C" if graphWithClass is True else "M"}XY_for_{name}',load_data_set_from_url(path=res_path, sep=',', encoding='utf-8',
                                         index_col=0,
                                         na_values=None))
            logic_i_p.append(res)

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
        if sum([f'{nominal_factor_colums[i]}_{"withClass" if graphWithClass else "withoutClass"}_model_turn_1_completed' in file for _, _, files in
             os.walk(cwd + f'/mlna_1/{nominal_factor_colums[i]}/') for
             file in files]) > 0 :
            continue
        # name=f'{nominal_factor_colums[i]}_mln'
        # load config df file
        config_df_path = cwd + f'/mlna_1/{nominal_factor_colums[i]}/' + [file for _, _, files in
              os.walk(
                  cwd + f'/mlna_1/{nominal_factor_colums[i]}/')
              for
              file in files][
            [f'config_df_for_{nominal_factor_colums[i]}_{"withClass" if graphWithClass else "withoutClass"}' in file for _, _, files in
             os.walk(cwd + f'/mlna_1/{nominal_factor_colums[i]}/') for
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
        # print(logic_i_p)
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
        with open(
                cwd + f'/mlna_1/{nominal_factor_colums[i]}/{nominal_factor_colums[i]}_{"withClass" if graphWithClass else "withoutClass"}_model_turn_1_completed.dtvni',
                "a") as fichier:
            fichier.write("")

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
            # build the MLN for the variable i
            col_targeted = [f'{nominal_factor_colums[i]}' for i in layer_config]
            case_k = '_'.join(col_targeted)
            print(case_k)
            if sum([
                       f'{case_k}_{"withClass" if graphWithClass else "withoutClass"}_model_turn_1_completed' in file
                       for _, _, files in
                       os.walk(cwd + f'/mlna_{k}/{case_k}/') for
                       file in files]) > 0:
                continue
            # load config df file
            config_df_path = cwd + f'/mlna_{k}/{case_k}/' + [file for _, _, files in
                 os.walk(
                     cwd + f'/mlna_{k}/{case_k}/')
                 for
                 file in files][
                [(f'config_df_for_{case_k}_{"withClass" if graphWithClass else "withoutClass"}' in file)
                 for _, _, files in
                 os.walk(cwd + f'/mlna_{k}/{case_k}/') for
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
            table_g = None
            table_p = None
            logic_i_g = None
            logic_i_p = None
            mlc_cf = None
            with open(
                    cwd + f'/mlna_{k}/{case_k}/{case_k}_{"withClass" if graphWithClass else "withoutClass"}_model_turn_1_completed.dtvni',
                    "a") as fichier:
                fichier.write("")

        modelD = None
        PERSONS = None
        table_p = None
        table_g = None


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
    x_train, x_test, y_train, y_test = x_traini, x_testi, y_traini, y_testi
    PERSONS = get_persons(x_train)
    PERSONS_T = get_persons(x_test)
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
        col_targeted = [f'{nominal_factor_colums[i]}' for i in layer_config]
        case_k = '_'.join(col_targeted)
        if sum([
            f'{case_k}_{"withClass" if graphWithClass else "withoutClass"}_model_turn_1_completed' in file
            for _, _, files in
            os.walk(cwd + f'/mlna_{k}/{case_k}/') for
            file in files]) > 0:
            continue
        # load config df file
        config_df_path = cwd + f'/mlna_{k}_b/{case_k}/' + [file for _, _, files in
                                                         os.walk(
                                                             cwd + f'/mlna_{k}_b/{case_k}/')
                                                         for
                                                         file in files][
            [(f'config_df_for_{case_k}_{"withClass" if graphWithClass else "withoutClass"}' in file)
             for _, _, files in
             os.walk(cwd + f'/mlna_{k}_b/{case_k}/') for
             file in files].index(True)
        ]
        config_df = read_model(path=config_df_path)
        ##########################################
        ####### Descriptors Config Generator ######

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
        table_g = None
        table_p = None
        logic_i_g = None
        logic_i_p = None
        mlc_cf = None
        with open(
                cwd + f'/mlna_{k}_b/{case_k}/{case_k}_{"withClass" if graphWithClass else "withoutClass"}_model_turn_2_completed.dtvni',
                "a") as fichier:
            fichier.write("")
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
    parser.add_argument('--baseline', action="store_true", required=False, help='Entrainement sur les données de base ?')
    parser.add_argument('--graph_with_class', action="store_true", required=False, help='integrant les classes?')

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
    original_path = args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}/' + [file for _, _, files in
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
    clfs = init_models()
    if args.baseline is True: # Baseline execution
        if sum([f'classic_metric' in file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/evaluation') for
                file in files]) > 0:
            print("✅ Stage already completed")
            exit(0)


        default = make_builder(
            fix_imbalance=False,
            # DATA_OVER=promise[7],
            target_variable=target_variable,
            clfs=clfs,
            domain='classic',
            prefix=domain,
            verbose=verbose,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            cwd=args.cwd + f'/{results_dir}{domain}',
            withCost=cost,
            financialOption=ast.literal_eval(financialOption) if cost else None,
            x_traini=x_traini,
            x_testi=x_testi,
            y_traini=y_traini,
            y_testi=y_testi,
            original=original
        )
        print("✅ Baseline models training completed")
        exit(0)
    else:
        if sum([f'classic_metric' in file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/evaluation') for
                file in files]) == 0:
            print("❌ Unable to access baseline models evaluation")
            exit(0)
        default_path = args.cwd + f'/{results_dir}{domain}/evaluation/' + \
                        [file for _, _, files in
                         os.walk(
                             args.cwd + f'/{results_dir}{domain}/evaluation')
                         for
                         file in files][
                            [f'classic_metric' in file for _, _, files in
                             os.walk(args.cwd + f'/{results_dir}{domain}/evaluation') for
                             file in files].index(True)
                        ]
        default = (
            'classic',load_data_set_from_url(path=default_path, sep=dataset_delimiter, encoding=encoding,
                                     index_col=index_col,
                                     na_values=None))
    # ------------------------------------------------------------------------------------------------------------------
    if args.turn == 1: # check if we are onto the first turn
        if sum(['model_turn_1_completed.dtvni' == file for _, _, files in
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
                default=default,
                clfs=clfs,
                verbose=verbose
            )
            outperformers = dict(sorted(outperformers.items(), key=lambda x: x[1], reverse=True))
            bestK = bestThreshold(list(outperformers)) + 1 if len(outperformers) > 2 else len(outperformers)
            # elbow = elbow_method(list(outperformers))
            # cusum = cumulative_difference_threshold(list(outperformers))
            print(f"{outperformers}, {NbGood} Goods and the best top k is {bestK}")
            save_model(
                cwd=args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}',
                clf={
                    'model': outperformers,
                    'nbGood': NbGood,
                    'bestK': bestK,
                    "name": [prepro_config["categorial_col"][i] for i in list(outperformers.keys())],
                },
                prefix="",
                clf_name=f"MNIFS_{domain}_best_features",
                sub="",
                times=False
            )
            contenu = f"""
                'model': {outperformers},
                'nbGood': {NbGood},
                'bestK': {bestK},
                'name':{[prepro_config['categorial_col'][i] for i in list(outperformers.keys())]},
                'BName':{[prepro_config['categorial_col'][i] for i in list(outperformers.keys())[:bestK]]},
                """
            with open(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_1/model_turn_1_completed.dtvni', "a") as fichier:
                fichier.write(contenu)


        if sum(['model_turn_1_completed.dtvni' == file for _, _, files in
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
                domain= domain,
                target_variable= target_variable,
                graphWithClass=False,
                fix_imbalance=False,
                withCost=cost,
                financialOption = ast.literal_eval(financialOption) if cost else None,
                duration_divider = duration_divider,
                rate_divider= rate_divider,
                original=original,
                default=default,
                clfs=clfs,
                verbose=verbose
            )
            with open(
                    args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/mlna_2/model_turn_1_completed.dtvni',
                    "a") as fichier:
                fichier.write("")
    if args.turn == 2:  # check if we are onto the first turn
        if sum([f"MNIFS_{domain}_best_features" in file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}') for
                file in files]) == 0:
            print("❌ Unable to access selection protocol results")
            exit(1)

        if sum(['model_turn_2_completed.dtvni' == file for _, _, files in
                os.walk(args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/select') for
                file in files]) > 0:
            print("✅ MLNA k Best model already completed")
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
                domain=domain,
                target_variable=target_variable,
                alpha=args.alpha,
                graphWithClass=args.graph_with_class,
                topR=list(mnifs_config['model'].keys()),
                fix_imbalance=False,
                withCost=cost,
                financialOption = ast.literal_eval(financialOption) if cost else None,
                duration_divider = duration_divider,
                rate_divider= rate_divider,
                original=original,
                default=default,
                clfs=clfs,
                verbose=verbose
            )
            with open(
                    args.cwd + f'/{results_dir}{domain}/{args.alpha}/{target_columns_type}/select/model_turn_2_completed.dtvni',
                    "a") as fichier:
                fichier.write("")


    print("Descripteurs extraits et sauvegardés.")

if __name__ == "__main__":
    main()