# 05_report_generation.py
import sys

# Ajoutez le répertoire parent pour pouvoir importer les modules
sys.path.append('..')  # Ajouter le répertoire parent au chemin de recherche des modules

from modules.statistical import *  # Report functions
from modules.mailing import *  # Report functions

from dotenv import load_dotenv


def load_env_with_path():
    """Chargement avec chemin spécifique vers le fichier .env"""
    # Chemin vers le fichier .env
    env_path = Path('.') / '.env'

    # Chargement avec chemin explicite
    load_dotenv(dotenv_path=env_path)

    # Ou depuis un répertoire parent
    # load_dotenv(dotenv_path=Path('..') / '.env')

    return {
        'gmail_user': os.getenv('GMAIL_USER'),
        'gmail_password': os.getenv('GMAIL_APP_PASSWORD'),
        'recipients': os.getenv('EMAIL_RECIPIENTS', '').split(','),
        'email_cc': os.getenv('EMAIL_CC', '').split(',')
    }

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
    # look for a flag file for each child
    completed = []  # where we store completed children
    for child in children_folder:
        counter = 0  # nb of alpha completed in that alpha case
        for alpha in alphas:
            if sum([motif == file for _, _, files in
                    os.walk(f'{results_path}{child}/{alpha}/{target_columns_type}/{fold}') for
                    file in files]) > 0:
                counter += 1
        if counter == len(alphas):
            completed.append(child)

    return completed


def build_macro_store(approach, logics, configs, metrics, datasets):
    results = {}
    for app in approach:
        for logic in logics:
            for config in configs:
                if ('Y' in config and logic in ['GAP', 'PER']) or not ('Y' in config):
                    key = f"{app}_{logic}_{config}"
                    results[key] = {}
                    for metric in metrics:
                        results[key][metric] = {ds: [] for ds in datasets}

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
        index_col=None
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
            ## get model dictionary
            models = model_desc()
            ## save only the ones use during the classic learning
            models_name = {key: models[key] for key in models.keys() if key in models_list}
            # print(models_name)

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
                    isBest=True,
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
                    isBest=True,
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
        top=10
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
                f'{cwd}/{result_folder}',
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
            shapStore[result_folder]['MlC']['BOT']['CXY'] = shapStore[result_folder]['MlC']['BOT']['CXY'] + \
                                                            temp['MlC']['BOT']['CXY']
            shapStore[result_folder]['MCA']['BOT']['CXY'] = shapStore[result_folder]['MCA']['BOT']['CXY'] + \
                                                            temp['MCA']['BOT']['CXY']

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
    ## get model dictionary
    models = model_desc()
    ## save only the ones use during the classic learning
    models_name = {key: models[key] for key in models.keys() if key in models_list}
    res = analyse_files_for_shap_value(
        models_name,
        shapStore,
        completed_folder,
        top=top
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

    if sum([f'reporting_completed.dtvni' in file for _, _, files in
            os.walk(args.cwd + f'/{results_dir}{domain}/') for
            file in files]) == 0:
        print("✅ Stage already completed")
        exit(0)

    folders = check_completed_folder(
        results_path=f"{args.cwd}/{results_dir}",
        target_columns_type=target_columns_type
    )
    macro_store, selection_store = metric_extraction(
        completed_folder=folders,
        alphas=None,
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
        alphas=None,
        cwd=f"{args.cwd}/{results_dir}",
        target_columns_type=target_columns_type,
        dataset_delimiter=dataset_delimiter,
        encoding=encoding,
        index_col=index_col,
        top=top
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
    for config, metric_data in macro_store.items():
        for metric, cat_data in metric_data.items():
            for cat, values in cat_data.items():
                if values:
                    macro_store[config][metric][cat] = max(values, key=lambda x: x[0])

    # Initialiser un compteur pour le total des meilleures valeurs par configuration
    total_best_counts = {config: 0 for config in macro_store.keys()}
    # Trouver les meilleures améliorations par colonne
    best_values = {metric: {cat: (-np.inf, "") for cat in folders} for metric in metrics}
    # print(macro_store)
    for config, metric_data in macro_store.items():
        for metric, cat_data in metric_data.items():
            # print(config, metric, cat_data)
            for cat, (value, algo) in cat_data.items():
                if value > best_values[metric][cat][0]:
                    best_values[metric][cat] = (value, algo)

    # Compter le nombre de fois qu'une ligne détient les meilleurs résultats
    for config, metric_data in macro_store.items():
        for metric in metrics:
            for cat in folders:
                if round(macro_store[config][metric][cat][0], 1) == round(best_values[metric][cat][0], 1):
                    total_best_counts[config] += 1

    # Trier les configurations par total décroissant
    sorted_totals = sorted(total_best_counts.items(), key=lambda x: x[1], reverse=True)
    best_total, second_best_total, third_best_total = sorted_totals[0][1], sorted_totals[1][1], sorted_totals[2][1]
    # Génération du tableau LaTeX
    latex_table = """
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
                formatted_value = f"\\textbf{{{value:.1f}}}" if round(value, 1) == round(best_value,
                                                                                         1) else f"{value:.1f}"
                row += f" & {formatted_value} ({algo})"
        # Formater la colonne Total

        # if total_value == best_total:
        #     formatted_total = "\\textbf{\\color{blue}"+str(total_value)+"}"
        # elif total_value == second_best_total:
        #     formatted_total = "\\underline{\\color{green}"+str(total_value)+"}"
        # elif total_value == third_best_total:
        #     formatted_total = "\\textit{\\color{red}"+str(total_value)+"}"
        # else:
        formatted_total = f"{total_value}"
        if total_value == best_total:
            formatted_total = f"\\textbf{{{total_value}}}"
        elif total_value == second_best_total:
            formatted_total = f"\\underline{{{total_value}}}"
        elif total_value == third_best_total:
            formatted_total = f"\\textit{{{total_value}}}"

        row += f" & {formatted_total} \\\\ \\hline\n"
        latex_table += row

    latex_table += "\\end{tabular}"

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
                    """ + latex_table + """
                    \\caption{Precise selection protocol}
                    \\label{precision:protocole}
                    \\end{table}
                    \\newpage
                    """ + shapPlots + footer)
    _file.close()

    environment = load_env_with_path()
    SendReport(
        GMAIL_USER_=environment['gmail_user'],
        GMAIL_APP_PASSWORD_=environment['gmail_password'],
        LATEX_FILE_=filename1,
        TO_EMAILS_=environment['recipients'],
        CC_EMAILS_=environment['email_cc'],
        SUBJECT_=f"Pipeline Report - {timestr}",
        EMAIL_BODY_=f"""
Hello,

Please find attached the automatic report of our machine learning pipeline.

This report contains:
- Performance analysis of the attribute selection protocol (MNIFS).
- The best performance configurations of the experimenetations environment (20 configurations)
- SHAPLey values showing attribute contributions for a top k ({top})

Report automatically generated on {timestr}.

Yours faithfully
Pipeline ML System
MSc. DJIEMBOU Victor in DS & AI @UY1
Junior Data Scientist @freelance
Senior FullStack Developer @freelance
Community Lead @World
    """
    )
    with open(
            args.cwd + f'/{results_dir}{domain}/reporting_completed.dtvni',
            "a") as fichier:
        fichier.write("")

if __name__ == "__main__":
    main()
