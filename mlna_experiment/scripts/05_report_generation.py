# 05_report_generation.py
import sys

# Ajoutez le répertoire parent pour pouvoir importer les modules
sys.path.append('..')  # Ajouter le répertoire parent au chemin de recherche des modules

from modules.statistical import *  # Report functions


def check_completed_folder(results_path, alphas, target_columns_type):
    # get children folders of the results folder
    children_folder = [dirnames for _, dirnames, _ in os.walk(f'{results_path}')][0]
    # look for a flag file for each child
    completed = []  # where we store completed children
    for child in children_folder:
        counter = 0  # nb of alpha completed in that alpha case
        for alpha in alphas:
            if sum(['model_turn_2_completed.dtvni' == file for _, _, files in
                    os.walk(f'{results_path}{child}/{alpha}/{target_columns_type}/select') for
                    file in files]) > 0:
                counter += 1
        if counter == len(alphas):
            completed.append(child)

    return completed


def main():
    # Définition des arguments
    import argparse

    parser = argparse.ArgumentParser(description='Exécution du pipeline MLNA - SPLIT DATASET')
    parser.add_argument('--cwd', type=str, required=True, help='Répertoire de travail courant')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Nom du dataset')

    # Récupération des arguments
    args = parser.parse_args()

    config = load_config(f"{args.cwd}/configs/{args.dataset_folder}/config.ini")

    domain = config["DATA"]["domain"]

    encoding = config["PREPROCESSING"]["encoding"]
    dataset_delimiter = config["DATA"]["dataset_delimiter"]
    target_variable = config["DATA"]["target"]

    processed_dir = config["GENERAL"]["processed_dir"]
    split_dir = config["GENERAL"]["split_dir"]
    results_dir = config["GENERAL"]["results_dir"]
    target_columns_type = config["GENERAL"]["target_columns_type"]
    verbose = config.getboolean("GENERAL", "verbose")

    index_col = None if config["SPLIT"]["index_col"] in ["None", ""] else config.getint("SPLIT", "index_col")

    alphas = [round(i, 1) for i in np.arange(0.1, 1.0, 0.1)]
    alphas.append(0.85)
    print(check_completed_folder(f"{args.cwd}/{results_dir}", alphas, target_columns_type))


if __name__ == "__main__":
    main()
