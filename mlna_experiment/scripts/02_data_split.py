# 02_data_split.py
from .cpu_limitation_usage import *
import sys
# Ajoutez le répertoire parent pour pouvoir importer les modules
sys.path.append('..')  # Ajoute le répertoire parent au chemin de recherche des modules
from modules.file import *  # File manipulation functions
from modules.modeling import *  # Modeling functions
import statistics
from modules.env import *  # Env functions



def try_split_all_models(df, target, test_size, max_perf, max_tries=100, reset_index=False, verbose=True):
    models = init_models()
    print(df[target].value_counts())
    for attempt in range(max_tries):
        seed = random.randint(0, 10000)
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        classes_train = set(y_train.unique())
        classes_test = set(y_test.unique())
        classes_all = set(y.unique())
        print(classes_train, classes_test, classes_all)
        if classes_train != classes_all or classes_test != classes_all:
            print(f"[WARN] Certaines classes manquent dans le train/test.") if verbose else None
            print(f"y_train classes: {sorted(classes_train)}") if verbose else None
            print(f"y_test  classes: {sorted(classes_test)}") if verbose else None
            print(f"y total classes: {sorted(classes_all)}") if verbose else None
            continue

        all_below_threshold = True
        print(f"🔍 Essai {attempt+1} avec seed {seed}")

        perfs = []
        for name, clf in models.items():
            try:
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                cfm = compute_confusion_matrix(y_test, preds, list(np.unique(y_test)))

                f1_score_r = f1_macro(cfm)
                f1 = f1_score(y_test, preds, average='macro')  # ← ici
                print(f"  → {name}: f1_macro = {f1:.4f}") if verbose else None
                perfs.append(f1)
            except Exception as e:
                print(f"[ERROR] {name}: {e}") if verbose else None
                all_below_threshold = False
                break

        if statistics.mean(perfs) >= max_perf:
            all_below_threshold = False

        if all_below_threshold:
            if reset_index is True:
                print(f"[WARN] Resetting the index to {seed}") if verbose else None
                X_train.reset_index(drop=True,inplace = True)
                X_test.reset_index(drop=True,inplace = True)
                y_train.reset_index(drop=True,inplace = True)
                y_test.reset_index(drop=True,inplace = True)
            return X_train, X_test, y_train, y_test, seed

    raise Exception(f"❌ Aucun split trouvé avec tous les modèles < {max_perf} (f1_macro) après {max_tries} essais")


def main():
    # Définition des arguments
    import argparse

    parser = argparse.ArgumentParser(description='Exécution du pipeline MLNA - SPLIT DATASET')
    parser.add_argument('--cwd', type=str, required=True, help='Répertoire de travail courant')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Nom du dataset')

    # Récupération des arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    config = load_config(f"{args.cwd}/configs/{args.dataset_folder}/config.ini")

    processed_dir = config["GENERAL"]["processed_dir"]
    split_dir = config["GENERAL"]["split_dir"]
    target_columns_type = config["GENERAL"]["target_columns_type"]
    verbose = config.getboolean("GENERAL", "verbose")

    test_size = float(config["SPLIT"]["test_size"])
    max_perf = float(config["SPLIT"].get("max_perf", 0.95))
    index_col = None if config["SPLIT"]["index_col"] in ["None", ""] else config.getint("SPLIT", "index_col")

    domain = config["DATA"]["domain"]
    target = config["DATA"]["target"]

    encoding = config["PREPROCESSING"]["encoding"]
    dataset_delimiter = config["SPLIT"]["dataset_delimiter"]

    # ------------------------------------------------------------------------------------------------------------------
    # lookup existing train test file
    if sum(['train' in file for _, _, files in
            os.walk(args.cwd + f'/{split_dir}{args.dataset_folder}/{target_columns_type}') for
            file in files]) > 0:
        print("✅ Stage already completed")
        exit(0)

    # ------------------------------------------------------------------------------------------------------------------
    # lookup processed file
    if sum([f'{domain}_preprocessed' in file for _, _, files in
            os.walk(args.cwd + f'/{processed_dir}{args.dataset_folder}/{target_columns_type}') for
            file in files]) == 0:
        print("❌ Unable to access preprocessed data")
        exit(1)

    dataset_path = args.cwd + f'/{processed_dir}{args.dataset_folder}/{target_columns_type}/'+[file for _, _, files in
            os.walk(args.cwd + f'/{processed_dir}{args.dataset_folder}/{target_columns_type}') for
            file in files][
                [f'{domain}_preprocessed' in file for _, _, files in
                os.walk(args.cwd + f'/{processed_dir}{args.dataset_folder}/{target_columns_type}') for
                file in files].index(True)
            ]
    print("the dataset path is ", dataset_path) if verbose else None
    dataset = load_data_set_from_url(path=dataset_path, sep=dataset_delimiter, encoding=encoding,
                                     index_col=index_col,
                                     na_values=None)
    X_train, X_test, y_train, y_test, used_seed = try_split_all_models(dataset, target, test_size, max_perf, reset_index=False, verbose=verbose)

    # Fusion pour sauvegarde
    train_df = X_train.copy()
    train_df[target] = y_train
    test_df = X_test.copy()
    test_df[target] = y_test
    save_dataset(
        cwd=args.cwd + f'/{split_dir}{args.dataset_folder}',
        dataframe=train_df,
        name=f'{domain}_train',
        sep=',',
        sub=f"/{target_columns_type}"
    )
    save_dataset(
        cwd=args.cwd + f'/{split_dir}{args.dataset_folder}',
        dataframe=test_df,
        name=f'{domain}_test',
        sep=',',
        sub=f"/{target_columns_type}"
    )

    # ------------------------------------------------------------------------------------------------------------------
    # load the dedicated work orginal dataset
    dataset_copy_path = args.cwd + f'/{processed_dir}{args.dataset_folder}/' + [file for _, _, files in
          os.walk(
              args.cwd + f'/{processed_dir}{args.dataset_folder}')
          for
          file in files][
        [f'original' in file for _, _, files in
         os.walk(args.cwd + f'/{processed_dir}{args.dataset_folder}') for
         file in files].index(True)
    ]
    dataset_copy = read_model(path=dataset_copy_path)

    x1_train = dataset_copy.loc[list(X_train.index)]
    x1_test = dataset_copy.loc[list(X_test.index)]
    save_model(
        cwd=args.cwd + f'/{split_dir}{args.dataset_folder}',
        clf=(x1_train, x1_test),
        prefix="",
        clf_name="original",
        sub=f"/{target_columns_type}"
    )

    print(f"✅ Preprocessing completed with seed {used_seed} et avec la moyenne des modèles < {max_perf}") if verbose else None

if __name__ == "__main__":
    main()
