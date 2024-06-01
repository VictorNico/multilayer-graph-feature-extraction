if __name__ == '__main__':
    import os
    from utils.file import *
    from utils.decision_tree import *
    from utils.metrics import *
    import time
    import sys
    import argparse
    from sklearn.tree import DecisionTreeClassifier
    from pandas import DataFrame, read_csv

    # print(help(DecisionTreeClassifier))
    def pack(data, y):
        dtc = DecisionTreeClassifier(min_impurity_decrease=0.00000000000000000001, max_features=20, min_samples_split=2)
        X_train = data.drop(y, axis=1)
        X_train.reset_index(drop=True, inplace=True)
        y_train = data[y]
        # print(X_train.columns,y_train)
        dtc.fit(X_train,y_train)
        return dtc

    parser = argparse.ArgumentParser(description='Exécution du pipeline MLNA')
    parser.add_argument('--input_file', type=str, required=True, help='path to input file')
    parser.add_argument('--delimiter', type=str, required=True, help='CSV delimiter')
    parser.add_argument('--is_header', action="store_true", required=False, help='CSV file have header')
    parser.add_argument('--class_name', type=str, required=True, help='Label name in file')
    parser.add_argument('--test_file', type=str, required=True, help='testing data path file ')
    parser.add_argument('--max_categories', type=int, required=True, help='maximal number of unique values in categorical dimension')
    parser.add_argument('--min_samples_split', type=int, required=True, help='nombre minimal of examples')
    parser.add_argument('--max_depth', type=int, required=True, help='Valeur d\'alpha')
    parser.add_argument('--min_information_gain', type=float, required=True, help='Portion du jeu de données')
    parser.add_argument('--is_classification', action="store_true", required=False, help='If classification')
    # Récupération des arguments
    args = parser.parse_args()

    # print(args)
    # <input_file> <delimiter> <is_header>  <class_name>  [<test_file> <is_classification> <max_depth> <min_samples_split> <min_information_gain> <max_categories>]
    train_data = process_file("data/train.csv" if args.input_file == ""  else args.input_file, "," if args.delimiter == "" else args.delimiter, True if not args.is_header else args.is_header)
    test_data = process_file("data/test.csv" if args.test_file == ""  else args.test_file, "," if args.delimiter == "" else args.delimiter, True if not args.is_header else args.is_header)

    labels = test_data[args.class_name]
    print(f"""
        {count_elements(labels)}
        ------------------------
        {count_elements(train_data[args.class_name])}

        """)
    del test_data[args.class_name]
    del train_data['Index']
    del test_data['Index']
    # print(f'{train_data.keys()}')

    print(f"""
        Gini Impurity: {gini_impurity(train_data['Class'])}
        Entropy Impurity: {entropy(train_data['Class'])}
        Variability Impurity: {variance(train_data['Class'])}
        Information Gain: {information_gain(train_data['Class'],getMask(train_data['Attribute_2'],NumCond,12))}
        Max Information Gain Split: {max_information_gain_split(train_data['Attribute_2'], train_data['Class'], entropy)}
        Best split criteria: {get_best_split('Class', train_data)}
        Split Data ([len(data1), len(data2)]): {[len(el['Class'])  for (variable, value, _, is_numeric) in [get_best_split('Class', train_data)] for el in make_split(variable, value, train_data, is_numeric)]}
        Make Prediction: {[make_prediction(ar['Class'],True) for ar in [el  for (variable, value, _, is_numeric) in [get_best_split('Class', train_data)] for el in make_split(variable, value, train_data, is_numeric)]]}
        
        """)

    print(f"Training data size: {len(train_data)} {len(train_data[list(test_data.keys())[0]])}")
    print(f"Test data size: {len(test_data)} {len(test_data[list(test_data.keys())[0]])}")
    start_time = time.time()
    IDT = interative_train_tree(train_data,args.class_name,args.is_classification,args.max_depth,args.min_samples_split,args.min_information_gain,0,args.max_categories)
    end_time = time.time()
    print(f"Interative train tree root condition is: {IDT['condition']}")

    result1 = []
    for i in list(range(len(test_data[list(test_data.keys())[0]]))):
        row = {key: test_data[key][i] for key in list(test_data.keys()) }
        result1.append(predict(row, tree=IDT))

    # print({"pred":result1,"real":test_labels})
    # Example usage
    true_labels = labels
    predicted_labels = result1
    labels = list(count_elements(labels).keys())

    cm = compute_confusion_matrix(true_labels, predicted_labels, labels)
    print(cm)

    print(f"""
    Accuracy: {accuracy(cm)},
    precision: {precision(cm)},
    Recall: {recall(cm)},
    F1-score: {f1_score(cm)}
    """)
    write_file("data/idt.csv", f"{IDT}")

    metric_names = [ 'Class Name', 'Version',  "precision","Accuracy", "Recall", "F1-score","Elapsed Time (second)", "Max depth", "Rows", "Cols"]
    metric_values = [args.class_name, "Sequential python", precision(cm),accuracy(cm),  recall(cm), f1_score(cm),f'{end_time - start_time:.6f}', args.max_depth, len(train_data[args.class_name]), len(train_data) ]
    save_metrics_to_csv("data/metrics.csv", metric_names, metric_values)


    train_data = read_csv("data/train1.csv" if args.input_file == ""  else args.input_file, sep = "," if args.delimiter == "" else args.delimiter, header=0)
    test_data = read_csv("data/test1.csv" if args.test_file == ""  else args.test_file, sep = "," if args.delimiter == "" else args.delimiter, header=0)

    start_time = time.time()
    IDTC = pack(train_data, args.class_name)
    end_time = time.time()

    result1 = IDTC.predict(DataFrame(test_data.drop(args.class_name, axis=1)))

    # print({"pred":result1,"real":test_labels})
    # Example usage
    labels = list(test_data[args.class_name].unique())
    true_labels = test_data[args.class_name]
    predicted_labels = result1
    labels = list(count_elements(labels).keys())

    cm = compute_confusion_matrix(true_labels, predicted_labels, labels)
    print(cm)

    print(f"""
    Accuracy: {accuracy(cm)},
    precision: {precision(cm)},
    Recall: {recall(cm)},
    F1-score: {f1_score(cm)}
    """)
    write_file("data/idt.csv", f"{IDTC}")

    metric_names = [ 'Class Name', 'Version',  "precision","Accuracy", "Recall", "F1-score","Elapsed Time (second)", "Max depth", "Rows", "Cols"]
    metric_values = [args.class_name, "Sequential python", precision(cm),accuracy(cm),  recall(cm), f1_score(cm),f'{end_time - start_time:.6f}', args.max_depth, train_data.shape[0], train_data.shape[1] ]
    save_metrics_to_csv("data/metrics.csv", metric_names, metric_values)
