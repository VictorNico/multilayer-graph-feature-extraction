# 05_report_generation.py
import os
import sys

# Ajoutez le répertoire parent pour pouvoir importer les modules
sys.path.append('..')  # Ajouter le répertoire parent au chemin de recherche des modules

from modules.statistical import *  # Report functions
from modules.mailing import *  # Report functions
from modules.env import *  # Env functions



def check_completed_folder(
        results_path="",
        alphas=None,
        target_columns_type='',
        motif='model_turn_2_completed.dtvni',
        fold='select'
):
    """ @methods check_completed_folder
            Look for all completed pipeline folder
            Parameters
            ----------
            results_path: str - the path to the results
            alphas: list - experimental used alpha values
            target_columns_type: str - the type of feature analyse focus
            motif: str - the name of flag result file showing end of the required stage
            fold: str - the name of the folder where to find out

            Returns
            -------
            List of all completed pipeline folder names

        """
    # get children folders of the results folder
    if alphas is None:
        alphas = [round(i, 1) for i in np.arange(0.1, 1.0, 0.1)]
        alphas.append(0.85)

    children_folder = [dirnames for _, dirnames, _ in os.walk(f'{results_path}')][0]
    print(alphas,children_folder)
    # look for a flag file for each child
    completed = []  # where we store completed children
    for child in children_folder:
        counter = 0  # nb of alpha completed in that alpha case
        for alpha in alphas:
            if sum([str(motif).casefold().strip() == str(file).casefold().strip() for _, _, files in
                    os.walk(f'{results_path}{child}/{alpha}/{target_columns_type}/{fold}') for
                    file in files]) > 0:
                counter += 1
        if counter == len(alphas):
            completed.append(child)

    return completed


def build_macro_store(approach, logics, configs, metrics, datasets):
    results = {}
    results['classic'] = {}
    results['real'] = {}
    results['gain'] = {}
    for app in approach:
        for logic in logics:
            for config in configs:
                if ('Y' in config and logic in ['GAP', 'PER']) or not ('Y' in config):
                    key = f"{app}_{logic}_{config}"
                    results["gain"][key] = {}
                    results["real"][key] = {}
                    for metric in metrics:
                        results["real"][key][metric] = {ds: [] for ds in datasets}
                        results["gain"][key][metric] = {ds: [] for ds in datasets}
                        results['classic'][metric] = {ds: [] for ds in datasets}

    return results


def metric_extraction(
        completed_folder=None,
        alphas=None,
        configs=None,
        logics=None,
        approach=None,
        metrics=None,
        glo=True,
        per=False,
        gap=False,
        cwd=None,
        target_columns_type=None,
        dataset_delimiter=None,
        encoding=None,
        index_col=None,
        models=['LDA','LR','SVM','DT','RF','XGB'],
):
    """ @methods metric_extraction
            Look inside completed folder to extract and record saved models performance over some metrics
            Parameters
            ----------
            completed_folder: list - List of all completed pipeline folder names
            alphas: list - experimental used alpha values
            configs: list - pipeline possible configurations
            logics: list - pipeline ways to personalize the pagerank
            approach: list - pipeline ways to embed new description in model input
            metrics: list - pipeline evaluation metrics to report
            glo: bool - if looking for glo results
            per: bool - if looking for per results
            gap: bool - if looking for gap results
            cwd: str - the current working directory
            target_columns_type: str - the type of feature analyse focus
            dataset_delimiter: str - result delimiter
            encoding: str - result encoding file
            index_col: int - id of index col

            Returns
            -------
            List of all completed pipeline folder names

        """

    if metrics is None:
        metrics = ['accuracy', 'f1-score', 'precision', 'recall', 'financial-cost']
    if approach is None:
        approach = ['MlC', 'MCA']
    if logics is None:
        logics = ['GLO', 'PER', 'GAP']
    if configs is None:
        configs = ['MX', 'CX', 'CY', 'CXY']
    if completed_folder is None:
        completed_folder = []
    if target_columns_type is None:
        target_columns_type = 'cat'
    if alphas is None:
        alphas = [round(i, 1) for i in np.arange(0.1, 1.0, 0.1)]
        alphas.append(0.85)

    # define stores
    macro_store = build_macro_store(approach, logics, configs, metrics, completed_folder)
    selection_store = {
        folder: {key: {'real_best_k': [], 'predicted_best_k': [], 'value': [], 'model': []} for key in alphas} for
        folder in completed_folder
    }

    # fetch completed folders

    for index, alpha in enumerate(alphas):  # on alpha
        for index2, result_folder in enumerate(completed_folder):  # on result folder name
            if sum([f"MNIFS_{result_folder}_best_features" in file for _, _, files in
                    os.walk(cwd + f'{result_folder}/{alpha}/{target_columns_type}') for
                    file in files]) == 0:
                print("❌ Unable to access selection protocol results")
                pass

            mnifs_path = cwd + f'{result_folder}/{alpha}/{target_columns_type}/' + \
                         [file for _, _, files in
                          os.walk(
                              cwd + f'{result_folder}/{alpha}/{target_columns_type}')
                          for
                          file in files][
                             [f"MNIFS_{result_folder}_best_features" in file for _, _, files in
                              os.walk(cwd + f'{result_folder}/{alpha}/{target_columns_type}') for
                              file in files].index(True)
                         ]
            mnifs_config = read_model(path=mnifs_path)
            selection_store[result_folder][alpha]['accuracies'] = list(mnifs_config['model'].keys())
            selection_store[result_folder][alpha]['predicted_best_k'] = mnifs_config['bestK']

            if sum([f'classic_metric' in file for _, _, files in
                    os.walk(cwd + f'{result_folder}/evaluation') for
                    file in files]) == 0:
                print("❌ Unable to access baseline models evaluation")
                pass
            default_path = cwd + f'/{result_folder}/evaluation/' + \
                           [file for _, _, files in
                            os.walk(
                                cwd + f'/{result_folder}/evaluation')
                            for
                            file in files][
                               [f'classic_metric' in file for _, _, files in
                                os.walk(cwd + f'/{result_folder}/evaluation') for
                                file in files].index(True)
                           ]
            classic_f = load_data_set_from_url(path=default_path, sep=dataset_delimiter, encoding=encoding,
                                               index_col=index_col,
                                               na_values=None)

            # global list of accuracy
            list_of_accuracy = []
            # get model list on classic results
            models_list = classic_f.index.values.tolist()
            ## save only the ones use during the classic learning
            models_name = {key:key for key in models if key in models_list}
            # print(models_name)
            # print(load_results(
            #         outputs_path=f'{cwd}/{result_folder}',
            #         _type=target_columns_type,
            #         k=1,
            #         alpha=alpha,
            #         per=per,
            #         glo=glo,
            #         mix=gap,
            #         isRand=False,
            #         match=lambda x: True,
            #         attributs=[dirnames for _, dirnames, _ in
            #                    os.walk(f'{cwd}/{result_folder}/{alpha}/{target_columns_type}/mlna_1')][0],
            #         isBest=False,
            #         dataset_delimiter=dataset_delimiter,
            #         encoding=encoding,
            #         index_col=index_col
            #     ))
            # exit(0)
            list_of_accuracy, macro_store = analyse_files(
                models_name,
                metrics,
                load_results(
                    outputs_path=f'{cwd}/{result_folder}',
                    _type=target_columns_type,
                    k=1,
                    alpha=alpha,
                    per=per,
                    glo=glo,
                    mix=gap,
                    isRand=False,
                    match=lambda x: True,
                    attributs=[dirnames for _, dirnames, _ in
                               os.walk(f'{cwd}/{result_folder}/{alpha}/{target_columns_type}/mlna_1')][0],
                    isBest=False,
                    dataset_delimiter=dataset_delimiter,
                    encoding=encoding,
                    index_col=index_col
                ),
                classic_f,
                macro_store,
                result_folder,
                list_of_accuracy,
                1
            )
            print(list_of_accuracy)
            if os.path.isdir(f'{cwd}/{result_folder}/{alpha}/{target_columns_type}/mlna_2'):
                list_of_accuracy, macro_store = analyse_files(
                    models_name,
                    metrics,
                    load_results(
                        outputs_path=f'{cwd}/{result_folder}',
                        _type=target_columns_type,
                        k=2,
                        alpha=alpha,
                        per=per,
                        glo=glo,
                        mix=gap,
                        isRand=True,
                        match=lambda x: True,
                        attributs=[dirnames for _, dirnames, _ in
                                   os.walk(f'{cwd}/{result_folder}/{alpha}/{target_columns_type}/mlna_2')][0],
                        isBest=False,
                        dataset_delimiter=dataset_delimiter,
                        encoding=encoding,
                        index_col=index_col
                    ),
                    classic_f,
                    macro_store,
                    result_folder,
                    list_of_accuracy,
                    2
                )
            ## identify the number of existing layer storage in best k
            mlna_folders_names = \
                [dirnames for _, dirnames, _ in os.walk(f'{cwd}/{result_folder}/{alpha}/{target_columns_type}/select')][
                    0]
            mlna_folders_names = sorted([int(el.split("_")[1]) for el in mlna_folders_names if "mlna" in el])
            # print(result_folder, mlna_folders_names, "//", alpha, mlna_folders_names)
            for index3, layer in enumerate(mlna_folders_names):  # on layer
                att = [dirnames for _, dirnames, _ in
                       os.walk(f'{cwd}/{result_folder}/{alpha}/{target_columns_type}/select/mlna_{layer}_b')][0]

                list_of_accuracy, macro_store = analyse_files(
                    models_name,
                    metrics,
                    load_results(
                        f'{cwd}/{result_folder}',
                        target_columns_type,
                        layer,
                        alpha,
                        per=per,
                        glo=glo,
                        mix=gap,
                        isRand=False,
                        match=lambda x: True,
                        attributs=att,
                        isBest=True,
                        dataset_delimiter=dataset_delimiter,
                        encoding=encoding,
                        index_col=index_col
                    ),
                    classic_f,
                    macro_store,
                    result_folder,
                    list_of_accuracy,
                    layer
                )

            # analyse impact of layers and identify the best mlna as k
            selection_store[result_folder][alpha]['list'] = list_of_accuracy
            list_of_accuracy = sorted(list_of_accuracy, key=lambda x: abs(x[2]),
                                      reverse=False)  # best will be at position 0
            selection_store[result_folder][alpha]['real_best_k'].append(list_of_accuracy[0][0])
            selection_store[result_folder][alpha]['model'].append(list_of_accuracy[0][1])
            selection_store[result_folder][alpha]['value'].append(list_of_accuracy[0][2])

    return macro_store, selection_store


def shap_extraction(
        completed_folder=None,
        alphas=None,
        cwd=None,
        target_columns_type=None,
        dataset_delimiter=None,
        encoding=None,
        index_col=None,
        top=10,
        models=['LDA','LR','SVM','DT','RF','XGB'],
        n=2,
):
    """ @methods metric_extraction
            Look inside completed folder to extract and record saved models performance over some metrics
            Parameters
            ----------
            completed_folder: list - List of all completed pipeline folder names
            alphas: list - experimental used alpha values
            configs: list - pipeline possible configurations
            logics: list - pipeline ways to personalize the pagerank
            approach: list - pipeline ways to embed new description in model input
            metrics: list - pipeline evaluation metrics to report
            glo: bool - if looking for glo results
            per: bool - if looking for per results
            gap: bool - if looking for gap results
            cwd: str - the current working directory
            target_columns_type: str - the type of feature analyse focus
            dataset_delimiter: str - result delimiter
            encoding: str - result encoding file
            index_col: int - id of index col

            Returns
            -------
            List of all completed pipeline folder names

        """
    if completed_folder is None:
        completed_folder = []
    if target_columns_type is None:
        target_columns_type = 'cat'
    if alphas is None:
        alphas = [round(i, 1) for i in np.arange(0.1, 1.0, 0.1)]
        alphas.append(0.85)

    # fetch completed folders
    print('coml',completed_folder)
    shapStore = {
        k: {
            'MlC': {
                'BOT': {
                    'CXY': []
                }
            },
            'MCA': {
                'BOT': {
                    'CXY': []
                }
            }
        } for k in completed_folder
    }
    for index, alpha in enumerate(alphas):  # on alpha
        for index2, result_folder in enumerate(completed_folder):  # on result folder name
            if sum([f"MNIFS_{result_folder}_best_features" in file for _, _, files in
                    os.walk(cwd + f'{result_folder}/{alpha}/{target_columns_type}') for
                    file in files]) == 0:
                print("❌ Unable to access selection protocol results")
                pass

            mnifs_path = cwd + f'{result_folder}/{alpha}/{target_columns_type}/' + \
                         [file for _, _, files in
                          os.walk(
                              cwd + f'{result_folder}/{alpha}/{target_columns_type}')
                          for
                          file in files][
                             [f"MNIFS_{result_folder}_best_features" in file for _, _, files in
                              os.walk(cwd + f'{result_folder}/{alpha}/{target_columns_type}') for
                              file in files].index(True)
                         ]
            mnifs_config = read_model(path=mnifs_path)
            elb = elbow_method(list(mnifs_config['model'].keys()))
            ## identify the number of existing layer storage in best k
            att = [dirnames for _, dirnames, _ in
                   os.walk(f'{cwd}/{result_folder}/{alpha}/{target_columns_type}/select/mlna_{elb}_b')][0]

            temp = load_results(
                f'{cwd}{result_folder}',
                target_columns_type,
                elb,
                alpha,
                per=False,
                glo=False,
                mix=False,
                bot=True,
                isRand=False,
                match=lambda x: True,
                attributs=att,
                isBest=True,
                dataset_delimiter=dataset_delimiter,
                encoding=encoding,
                index_col=index_col
            )
            print(f'{cwd}{result_folder}/{alpha}/{target_columns_type}/select/mlna_{elb}_b/{att}/mixed/both/evaluation')
            # pretty_print(temp)
            shapStore[result_folder]['MlC']['BOT']['CXY'] = shapStore[result_folder]['MlC']['BOT']['CXY'] + \
                                                            temp['MlC']['BOT']['CXY']
            shapStore[result_folder]['MCA']['BOT']['CXY'] = shapStore[result_folder]['MCA']['BOT']['CXY'] + \
                                                            temp['MCA']['BOT']['CXY']

    if sum([f'classic_metric' in file for _, _, files in
            os.walk(cwd + f'/{result_folder}/evaluation') for
            file in files]) == 0:
        print("❌ Unable to access baseline models evaluation")
        pass
    default_path = cwd + f'/{result_folder}/evaluation/' + \
                   [file for _, _, files in
                    os.walk(
                        cwd + f'/{result_folder}/evaluation')
                    for
                    file in files][
                       [f'classic_metric' in file for _, _, files in
                        os.walk(cwd + f'/{result_folder}/evaluation') for
                        file in files].index(True)
                   ]
    classic_f = load_data_set_from_url(path=default_path, sep=dataset_delimiter, encoding=encoding,
                                       index_col=index_col,
                                       na_values=None)

    # global list of accuracy
    list_of_accuracy = []
    # get model list on classic results
    models_list = classic_f.index.values.tolist()
    ## save only the ones use during the classic learning
    models_name = {key:key for key in models if key in models_list}
    res = analyse_files_for_shap_value(
        models_name,
        shapStore,
        completed_folder,
        top=top,
        n=n,
    )

    return res


def main():
    # Définition des arguments
    import argparse

    parser = argparse.ArgumentParser(description='Exécution du pipeline MLNA - SPLIT DATASET')
    parser.add_argument('--cwd', type=str, required=True, help='Répertoire de travail courant')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Nom du dataset')

    # Récupération des arguments
    args = parser.parse_args()
    print(args)
    config = load_config(f"{args.cwd}/configs/{args.dataset_folder}/config.ini")

    encoding = config["PREPROCESSING"]["encoding"]
    dataset_delimiter = config["SPLIT"]["dataset_delimiter"]
    domain = config["DATA"]["domain"]
    results_dir = config["GENERAL"]["results_dir"]
    report_dir = config["GENERAL"]["report_dir"]
    target_columns_type = config["GENERAL"]["target_columns_type"]
    top = config.getint("REPORT", "shap_top")

    index_col = None if config["SPLIT"]["index_col"] in ["None", ""] else config.getint("SPLIT", "index_col")
    metrics = ['accuracy', 'f1-score']
    met_t_name = ['Acc', 'F1']
    environment = load_env_with_path()

    print(args.cwd,environment['alphas'])

    if sum([f'reporting_completed.dtvni' in file for _, _, files in
            os.walk(args.cwd + f'/{results_dir}{domain}/') for
            file in files]) != 0:
        print("✅ Stage already completed")
        exit(0)

    folders = check_completed_folder(
        results_path=f"{args.cwd}/{results_dir}",
        target_columns_type=target_columns_type,
        alphas=environment['alphas'],
        motif='model_turn_2_completed.dtvni'
    )
    folders = ['ADU', 'GER', 'BAN', 'NUR']
    print(folders)
    macro_store, selection_store = metric_extraction(
        completed_folder=folders,
        alphas=environment['alphas'],
        configs=None,
        logics=None,
        approach=None,
        metrics=metrics,
        glo=True,
        per=True,
        gap=True,
        cwd=f"{args.cwd}/{results_dir}",
        target_columns_type=target_columns_type,
        dataset_delimiter=dataset_delimiter,
        encoding=encoding,
        index_col=index_col
    )

    shapPlots = shap_extraction(
        completed_folder=folders,
        alphas=environment['alphas'],
        cwd=f"{args.cwd}/{results_dir}",
        target_columns_type=target_columns_type,
        dataset_delimiter=dataset_delimiter,
        encoding=encoding,
        index_col=index_col,
        top=top,
        n=2
    )

    # pretty_print(macro_store)
    dat, table, (real_values, elbow_values, cusum_values) = selection_proto(selection_store, f"{args.cwd}/{report_dir}")

    # Tolérances à tester
    tolerances = [0.01, 0.02, 0.03, 0.04, 0.05]

    # Fonction pour calculer la précision
    def compute_precision(estimated_values, real_values, tolerance):
        precisions = {}
        for dataset in real_values:
            real = np.array(real_values[dataset])
            estimated = np.array(estimated_values[dataset])
            precision = np.mean(np.abs(real - estimated) <= tolerance) * 100
            precisions[dataset] = precision
        return precisions

    # Calcul des précisions pour chaque tolérance
    elbow_results = {tol: compute_precision(elbow_values, real_values, tol) for tol in tolerances}
    cusum_results = {tol: compute_precision(cusum_values, real_values, tol) for tol in tolerances}

    precision_tab = proto_precision_tikz(
                tolerances,
                elbow_results,
                cusum_results,
                folders,
                layout_config=None
            )

    # Sélectionner les meilleures valeurs pour chaque catégorie
    # for config, metric_data in macro_store['real'].items():
    #     for metric, cat_data in metric_data.items():
    #         for cat, values in cat_data.items():
    #             if values:
    #                 macro_store['real'][config][metric][cat] = max(values, key=lambda x: x[0])
    #
    # for config, metric_data in macro_store['gain'].items():
    #     for metric, cat_data in metric_data.items():
    #         for cat, values in cat_data.items():
    #             if values:
    #                 macro_store['gain'][config][metric][cat] = max(values, key=lambda x: x[0])
    #
    # for metric, cat_data in macro_store['classic'].items():
    #     for cat, values in cat_data.items():
    #         if values:
    #             macro_store['classic'][metric][cat] = max(values, key=lambda x: x[0])

    best_classic_models = {}
    for metric, cat_data in macro_store['classic'].items():
        for cat, values in cat_data.items():
            if values:
                best_classic_models[(metric, cat)] = max(values, key=lambda x: x[0])[1]  # récupère le nom du modèle

    # Puis filtrer real et gain pour ce modèle spécifique
    # créer d'abord un backup de la macro brute
    macro_store_bak = copy.deepcopy(macro_store)
    # pretty_print(macro_store_bak)
    for config, metric_data in macro_store['real'].items():
        for metric, cat_data in metric_data.items():
            for cat, values in cat_data.items():
                if values and (metric, cat) in best_classic_models:
                    target_model = best_classic_models[(metric, cat)]

                    # Trouver les résultats pour ce modèle spécifique
                    model_real_values = [v for v in values if v[1] == target_model]
                    model_gain_values = [v for v in macro_store['gain'][config][metric][cat] if v[1] == target_model]

                    if model_real_values and model_gain_values:
                        macro_store['real'][config][metric][cat] = max(model_real_values, key=lambda x: x[0])
                        macro_store['gain'][config][metric][cat] = max(model_gain_values, key=lambda x: x[0])

    for metric, cat_data in macro_store['classic'].items():
        for cat, values in cat_data.items():
            if values:
                macro_store['classic'][metric][cat] = max(values, key=lambda x: x[0])

    # Initialiser un compteur pour le total des meilleures valeurs par configuration
    total_best_counts_real = {config: 0 for config in macro_store['real'].keys()}
    total_best_counts_gain = {config: 0 for config in macro_store['real'].keys()}
    # Trouver les meilleures améliorations par colonne
    best_values_real = {metric: {cat: (-np.inf, "") for cat in folders} for metric in metrics}
    best_values_gain = {metric: {cat: (-np.inf, "") for cat in folders} for metric in metrics}
    # print(macro_store)
    for config, metric_data in macro_store['real'].items():
        for metric, cat_data in metric_data.items():
            # print(config, metric, cat_data)
            for cat, (value, algo) in cat_data.items():
                if value > best_values_real[metric][cat][0] and not("classic" in config):
                    best_values_real[metric][cat] = (value, algo)

    for config, metric_data in macro_store['gain'].items():
        for metric, cat_data in metric_data.items():
            # print(config, metric, cat_data)
            for cat, (value, algo) in cat_data.items():
                if value > best_values_gain[metric][cat][0] and not("classic" in config):
                    best_values_gain[metric][cat] = (value, algo)

    # Compter le nombre de fois qu'une ligne détient les meilleurs résultats
    # pretty_print(total_best_counts_real)
    for config, metric_data in macro_store['real'].items():
        for metric in metrics:
            # pretty_print(macro_store['real'][config])
            for cat in folders:
                if round(macro_store['real'][config][metric][cat][0], 1) == round(best_values_real[metric][cat][0], 1) and not("classic" in config):
                    total_best_counts_real[config] += 1
    total_best_counts_real['classic'] = 0
    # pretty_print(macro_store['gain'])
    for config, metric_data in macro_store['gain'].items():
        for metric in metrics:
            # pretty_print(macro_store['gain'][config])
            for cat in folders:
                if round(macro_store['gain'][config][metric][cat][0], 1) == round(best_values_gain[metric][cat][0], 1) and not("classic" in config):
                    total_best_counts_gain[config] += 1
    total_best_counts_gain['classic'] = 0

    # Trier les configurations par total décroissant
    sorted_totals_gain = sorted(total_best_counts_gain.items(), key=lambda x: x[1], reverse=True)
    sorted_totals_real = sorted(total_best_counts_real.items(), key=lambda x: x[1], reverse=True)
    best_total_gain, second_best_total_gain, third_best_total_gain = sorted_totals_gain[0][1], sorted_totals_gain[1][1], sorted_totals_gain[2][1]
    best_total_real, second_best_total_real, third_best_total_real = sorted_totals_real[0][1], sorted_totals_real[1][1], sorted_totals_real[2][1]
    # Génération du tableau LaTeX
    def compute_table(
            total_best_counts,
            macro_store,
            metrics,
            best_values,
            best_total,
            second_best_total,
            third_best_total,
            folders,
            real=False
    ):
        latex_table = """
        \\resizebox{\\textwidth}{!}{%
        \\begin{tabular}{|l""" + "|l" * (len(folders) * len(metrics)) + """|c|}
            \\hline
            \\multirow{3}{*}{}""" + " ".join(
            ["& \\multicolumn{" + str(len(folders)) + "}{|c|}{" + met + "}" for met in metrics]) + """& \\multirow{3}{*}{Total}\\\\
            \\cline{2-""" + str(len(folders) * len(metrics) + 1) + """}
            & """ + " & ".join(folders * len(metrics)) + """& \\\\
            \\hline
        """

        for config in macro_store.keys():
            row = ""
            total_value = total_best_counts[config]
            row += config.replace('_', '\\_')
            for metric in metrics:
                for cat in folders:
                    value, algo = macro_store[config][metric][cat]
                    best_value, _ = best_values[metric][cat]
                    if not real is True:
                        formatted_value = f"\\textbf{{{value:.1f}}}" if round(value, 1) == round(best_value,
                                                                                             1) else f"{value:.1f}"
                    else:
                        formatted_value = f"\\textbf{{{value:.3f}}}" if round(value, 3) == round(best_value,
                                                                                                 3) else f"{value:.3f}"
                    row += f" & {formatted_value} ({algo})"
            formatted_total = f"{total_value}"
            if total_value == best_total:
                formatted_total = f"\\textbf{{{total_value}}}"
            elif total_value == second_best_total:
                formatted_total = f"\\underline{{{total_value}}}"
            elif total_value == third_best_total:
                formatted_total = f"\\textit{{{total_value}}}"

            row += f" & {formatted_total if not('classic' in config) else ''} \\\\ \\hline\n"
            latex_table += row

        latex_table += "\\end{tabular}}"
        return latex_table

    def compute_table_for_config_analysis_per_model_per_folder(
            macro_store,
            metrics,
            folders,
            models=['LDA','LR','SVM','DT','RF','XGB'],
    ):
        # variable de stockage des tabulaires
        fold_tab = {}
        local_macro = {'classic':macro_store['classic'],**macro_store['real']}
        dictio_prior_conf = {
            'MlC_GLO_MX':19,
            'MlC_GLO_CX':17,
            'MlC_PER_MX':15,
            'MlC_PER_CX':13,
            'MlC_PER_CY':11,
            'MlC_PER_CXY':9,
            'MlC_GAP_MX':7,
            'MlC_GAP_CX':5,
            'MlC_GAP_CY':3,
            'MlC_GAP_CXY':1,
            'MCA_GLO_MX':20,
            'MCA_GLO_CX':18,
            'MCA_PER_MX':16,
            'MCA_PER_CX':14,
            'MCA_PER_CY':12,
            'MCA_PER_CXY':10,
            'MCA_GAP_MX':8,
            'MCA_GAP_CX':6,
            'MCA_GAP_CY':4,
            'MCA_GAP_CXY':2
        }
        # parcours de datasets
        for folder in folders:
            latex_table = ("""
            \\resizebox{\\textwidth}{!}{%
            \\begin{tabular}{|l""" + "|l" * (len(models) * len(metrics)) + """|}
                \\hline 
                \\multirow{3}{*}{} & \\multicolumn{""" + str(len(models) * len(metrics)) + "}{|c|}{" + folder + """} \\\\ """
                + """\\cline{2-""" + str(len(models) * len(metrics)+1) + """}
                """ + " ".join(
                ["& \\multicolumn{" + str(len(metrics)) + "}{|c|}{" + met + "}" for met in models]) + """\\\\
                \\cline{2-""" + str(len(models) * len(metrics)+1) + """}
                & """ + " & ".join(met_t_name * len(models)) + """ \\\\
                \\hline
            """)


            for config in local_macro.keys():
                row = ""
                row += config.replace('_', '\\_')
                for model in models:
                    for metric in metrics:
                        # chercher les valeurs metrique et les classer
                        # Trouver les résultats pour ce modèle spécifique
                        model_real_values = sorted(list({(v[0], dictio_prior_conf[conf]  if (not conf in "classic") else 0, conf, macro_store['gain'][conf][metric][folder][index] if (not conf in "classic") else 0) for conf in local_macro.keys() for index, v in enumerate(local_macro[conf][metric][folder]) if v[1] == model}), reverse=True, key= lambda x: (x[0], x[1]))

                        # pretty_print(model_real_values[0])
                        value = max(list({v[0] for v in local_macro[config][metric][folder] if v[1] == model}))
                        best_value, _, conf, gain = model_real_values[0]
                        formatted_value = f"\\textbf{{{value:.3f}}}" if (round(value, 3) == round(best_value,
                                                                                                     3)) and conf == config else f"{value:.3f}"
                        row += f" & {formatted_value}"

                row += f" \\\\ \\hline\n"
                latex_table += row

            latex_table += "\\end{tabular}}"
            fold_tab[folder] = latex_table
        return fold_tab

    def compute_table_for_config_analysis_per_model_per_metric(
            macro_store,
            metrics,
            folders,
            models=['LDA','LR','SVM','DT','RF','XGB'],
    ):
        # variable de stockage des tabulaires
        fold_tab = {}
        local_macro = {**macro_store['gain']}
        dictio_prior_conf = {
            'MlC_GLO_MX':19,
            'MlC_GLO_CX':17,
            'MlC_PER_MX':15,
            'MlC_PER_CX':13,
            'MlC_PER_CY':11,
            'MlC_PER_CXY':9,
            'MlC_GAP_MX':7,
            'MlC_GAP_CX':5,
            'MlC_GAP_CY':3,
            'MlC_GAP_CXY':1,
            'MCA_GLO_MX':20,
            'MCA_GLO_CX':18,
            'MCA_PER_MX':16,
            'MCA_PER_CX':14,
            'MCA_PER_CY':12,
            'MCA_PER_CXY':10,
            'MCA_GAP_MX':8,
            'MCA_GAP_CX':6,
            'MCA_GAP_CY':4,
            'MCA_GAP_CXY':2
        }
        # parcours de datasets
        latex_table = ("""
        \\resizebox{\\textwidth}{!}{%
        \\begin{tabular}{|l""" + "|l" * (len(folders) * len(metrics)) + """|}
            \\hline 
            \\multirow{2}{*}{} """ + " ".join(
            ["& \\multicolumn{" + str(len(folders)) + "}{|c|}{" + met + "}" for met in metrics]) + """\\\\
            \\cline{2-""" + str(len(folders) * len(metrics)+1) + """}
            & """ + " & ".join(folders * len(metrics)) + """ \\\\
            \\hline
        """)


        for model in models:
            row = ""
            row += model
            for metric in metrics:
                for folder in folders:
                    # chercher les valeurs metrique et les classer
                    # Trouver les résultats pour ce modèle spécifique
                    model_gain_values = sorted(list({(v[0], dictio_prior_conf[conf]  if (not conf in "classic") else 0, conf, v[1]) for conf in local_macro.keys() for index, v in enumerate(local_macro[conf][metric][folder])}), reverse=True, key= lambda x: (x[0], x[1]))

                    # pretty_print(model_gain_values[0])
                    value = max(list({v[0] for conf in local_macro.keys() for v in local_macro[conf][metric][folder] if v[1] == model}))
                    best_value, _, conf, algo = model_gain_values[0]
                    formatted_value = "\\textbf{"+conf.replace('_', '\\_')+f"({value:.1f})"+"}" if (round(value, 1) == round(best_value,
                                                                                                 1)) and algo == model else conf.replace('_', '\\_')+f"({value:.1f})"
                    row += f" & {formatted_value}"

            row += f" \\\\ \\hline\n"
            latex_table += row

        latex_table += "\\end{tabular}}"
        return latex_table

    latex_table_real = compute_table(
            total_best_counts_real,
        {'classic':macro_store['classic'],**macro_store['real']},
            metrics,
            best_values_real,
            best_total_real,
            second_best_total_real,
            third_best_total_real,
            folders=['ADU', 'GER', 'BAN', 'NUR'],
            real=True
    )
    latex_table_gain = compute_table(
        total_best_counts_gain,
        {'classic':macro_store['classic'],**macro_store['gain']},
        metrics,
        best_values_gain,
        best_total_gain,
        second_best_total_gain,
        third_best_total_gain,
        folders
    )

    table_for_config_analysis_per_model_per_folder = compute_table_for_config_analysis_per_model_per_folder(
        macro_store_bak,
        metrics,
        folders
    )

    table_for_config_analysis_per_model_per_metric = compute_table_for_config_analysis_per_model_per_metric(
        macro_store_bak,
        metrics=['accuracy'],
        folders=folders
    )
    table_for_config_analysis_per_model_per_metric_1 = compute_table_for_config_analysis_per_model_per_metric(
        macro_store_bak,
        metrics=['f1-score'],
        folders=folders
    )
    # saving
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
    create_domain(f"{args.cwd}/{report_dir}{timestr}")
    filename1 = f"{args.cwd}/{report_dir}{timestr}/report_{timestr}.tex"
    _file = open(filename1, "w", encoding='utf-8')
    _file.write(header + """
                    \\begin{table}[H]
                    \\centering
                    """ + table + """
                    \\caption{Table of best performance according to alpha for each thresholding method and dataset}
                    \\label{valeur:protocole}
                    \\end{table}
                    \\newpage""" + """
                    \\begin{table}[H]
                    \\centering
                    
                    """ + precision_tab + """
                    \\caption{Precise selection protocol}
                    \\label{precision:protocole}
                    \\end{table}
                    \\newpage""" + """
                    \\begin{table}[H]
                    \\centering
                    
                    \\resizebox{\\textwidth}{!}{
                    """ + latex_table_gain + """}
                    \\caption{Impact gain table}
                    \\label{precision:protocole}
                    \\end{table}
                    \\newpage
                    """ +"""
                    \\begin{table}[H]
                    \\centering
                    """ + latex_table_real + """
                    \\caption{Impact value table}
                    \\label{precision:protocole}
                    \\end{table}
                    \\newpage
                    """ +
                    """
                    """.join(["""
                    \\begin{table}[H]
                    \\centering
                    """ + tab + """
                    \\end{table}
                    \\newpage""" for tab in table_for_config_analysis_per_model_per_folder.values()])
                + """
                \\begin{table}[H]
                \\centering
                """ + table_for_config_analysis_per_model_per_metric + """
                \\end{table}
                """
                + """
                \\begin{table}[H]
                \\centering
                """ + table_for_config_analysis_per_model_per_metric_1 + """
                \\end{table}
                \\newpage
                """

                + shapPlots + footer)
    _file.close()


#     SendReport(
#         GMAIL_USER_=environment['gmail_user'],
#         GMAIL_APP_PASSWORD_=environment['gmail_password'],
#         LATEX_FILE_=filename1,
#         TO_EMAILS_=environment['recipients'],
#         CC_EMAILS_=environment['email_cc'],
#         SUBJECT_=f"Pipeline Report - {timestr}",
#         EMAIL_BODY_=f"""
# Hello,
#
# Please find attached the automatic report of our machine learning pipeline.
#
# This report contains:
# - Performance analysis of the attribute selection protocol (MNIFS).
# - The best performance configurations of the experimenetations environment (20 configurations)
# - SHAPLey values showing attribute contributions for a top k ({top})
#
# Report automatically generated on {timestr}.
#
# Yours faithfully
# Pipeline ML System
# MSc. DJIEMBOU Victor in DS & AI @UY1
# Junior Data Scientist @freelance
# Senior FullStack Developer @freelance
# Community Lead @World
#     """
#     )
    with open(
            args.cwd + f'/{results_dir}{domain}/reporting_completed.dtvni',
            "a") as fichier:
        fichier.write("")

if __name__ == "__main__":
    main()
