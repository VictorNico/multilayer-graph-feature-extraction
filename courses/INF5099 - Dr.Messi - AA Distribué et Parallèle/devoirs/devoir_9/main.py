if __name__ == '__main__':
    from utils.file import *
    from utils.decision_tree import *
    from utils.metrics import *
    if False:

        data = process_file("data/german.csv", ",", True)




    data = process_file("data/german.csv", ",", True)

    print(f"""
        Gini Impurity: {gini_impurity(data['Class'])}
        Entropy Impurity: {entropy(data['Class'])}
        Variability Impurity: {variance(data['Class'])}
        Information Gain: {information_gain(data['Class'],getMask(data['Attribute_2'],NumCond,12))}
        Max Information Gain Split: {max_information_gain_split(data['Attribute_2'], data['Class'], entropy)}
        Best split criteria: {get_best_split('Class', data)}
        Split Data ([len(data1), len(data2)]): {[len(el['Class'])  for (variable, value, _, is_numeric) in [get_best_split('Class', data)] for el in make_split(variable, value, data, is_numeric)]}
        Make Prediction: {[make_prediction(ar['Class'],True) for ar in [el  for (variable, value, _, is_numeric) in [get_best_split('Class', data)] for el in make_split(variable, value, data, is_numeric)]]}
        
        """)

    labels = data['Class']
    del data['Class']

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=.5, random_state=42)

    print(f"""
    #########################################
    ##### Description of Dataset ############
    #########################################
    ------- Original Dataset ---------
    Number of columns : {len(list(data.keys()))-1}
    Number of rows : {sum(list(count_elements(data).values()))}
    repartition of classes: count_elements(data)
    ------- Training Dataset ---------
    Number of rows : {sum(list(count_elements(train_data).values()))}
    repartition of classes: count_elements(train_data)
    ------- Test Dataset ---------
    Number of rows : {sum(list(count_elements(test_data).values()))}
    repartition of classes: count_elements(test_data)
    -------------------------------
    """)

    # print("Training labels:")
    # print(count_elements(train_labels))
    #
    # print("Test labels:")
    # print(count_elements(test_labels))

    max_depth = 5
    min_samples_split = 20
    min_information_gain = 1e-5
    dat = dict(train_data)
    dat['Class'] = train_labels
    IDT = interative_train_tree(dat, 'Class', True, max_depth, min_samples_split, min_information_gain)
    result1 = []
    for i in list(range(len(test_data[list(test_data.keys())[0]]))):
        row = {key: test_data[key][i] for key in list(test_data.keys()) }
        result1.append(predict(row, tree=IDT))

    # print({"pred":result1,"real":test_labels})
    # Example usage
    true_labels = test_labels
    predicted_labels = result1
    labels = list(count_elements(test_labels).keys())

    write_file("data/pl.txt", f"{predicted_labels}")
    write_file("data/tl.txt", f"{true_labels}")
    write_file("data/l.txt", f"{labels}")
    cm = compute_confusion_matrix(true_labels, predicted_labels, labels)
    print(cm)

    # In[31]:

    print(f"""
    Accuracy: {accuracy(cm)},
    precision: {precision(cm)},
    Recall: {recall(cm)},
    F1-score: {f1_score(cm)}
    """)
    write_file("data/outputs.csv", f"Iterative Decision, Accuracy: {accuracy(cm)},precision: {precision(cm)},Recall: {recall(cm)},F1-score: {f1_score(cm)}")

    metric_names = ["Métrique 1", "Métrique 2", "Métrique 3"]
metric_values = [1.23, 4.56, 7.89]
save_metrics_to_csv("metrics.csv", metric_names, metric_values)