//
// Created by DJIEMBOU TIENCTHEU VICTOR NICO on 25/05/2024.
//

#include "headers/main.h"
#include "headers/files.h"
#include "headers/metrics.h"
#include "headers/decisionTree.h"


int main() {
    // // // // Metrics

    // Tester la fonction compute_confusion_matrix()
    std::vector<int> true_labels = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int> predicted_labels = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int> unique_labels = {0, 1};

    std::vector<std::vector<int>> confusion_matrix = compute_confusion_matrix(true_labels, predicted_labels, unique_labels);

    std::cout << "Confusion Matrix:\n";
    for (const auto& row : confusion_matrix) {
        for (int value : row) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }

    std::cout << "precision: " << precision(confusion_matrix) << std::endl;
    std::cout << "Accuracy: " << accuracy(confusion_matrix) << std::endl;
    std::cout << "recall: " << recall(confusion_matrix) << std::endl;
    std::cout << "f1: " << f1_score(confusion_matrix) << std::endl;


    // Tester la fonction count_elements()
    std::vector<std::variant<int, double, bool, std::string>> elements = {"apple", "banana", "cherry", "apple", "banana", "date"};
    std::unordered_map<std::string, int> element_counts = count_elements(elements);

    std::cout << "\nElement Counts:\n";
    for (const auto& pair : element_counts) {
        std::cout << pair.first << ": " << pair.second << "\n";
    }

    // // // // files


    std::string input = "42";
    std::cout << "Input: " << input << std::endl;
    std::cout << "Type: " << getType(input)->name() << std::endl;

    input = "3.14";
    std::cout << "Input: " << input << std::endl;
    std::cout << "Type: " << getType(input)->name() << std::endl;

    input = "true";
    std::cout << "Input: " << input << std::endl;
    std::cout << "Type: " << getType(input)->name() << std::endl;

    input = "hello";
    std::cout << "Input: " << input << std::endl;
    std::cout << "Type: " << getType(input)->name() << std::endl;

    std::string file_path = "data/german.csv";
    std::string details_separator = ",";
    bool header_flag = true;
    std::string className = "Class";

    std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> data = process_file(file_path, details_separator, header_flag);
    std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> data1 = data;
    data1.erase("Index");
    // Afficher le contenu de la map
    for (const auto& [key, values] : data) {
        std::cout << "Key: " << key << "->" << values.size() << std::endl;
//        for (const auto& value : values) {
//            std::cout << "  Value: ";
//            std::visit([](const auto& v) { std::cout << v; }, value);
//            std::cout << std::endl;
//        }
    }

    std::vector<std::variant<int, double, bool, std::string>> labels = data["Class"];
    data.erase(className);
    auto [train_data, test_data, y_train, y_test] = train_test_split(data, labels, 0.2, 42);

    write_dataset(train_data, test_data, y_train, y_test, "outputs", className);

    train_data[className] = y_train;

    //// Decision Tree
    std::cout << "Variance: " << variance(labels) << std::endl;
    std::cout << "Enrtopy: " << entropy(labels) << std::endl;
    std::cout << "Gini: " << gini_impurity(labels) << std::endl;

    std::vector<bool> mask = getMask(data["Attribute_2"], NumCond, 12);
    std::cout << "Mask: ";
    for (bool b : mask) {
        std::cout << b << " ";
    }
    std::cout << std::endl;

    std::cout << "Gain: " << information_gain(labels,mask) << std::endl;
    // review
    std::vector<std::variant<int, double, bool, std::string>> labels1 = {1,0,0,0,1,1,1};
    std::vector<std::variant<int, double, bool, std::string>> att = {12,10,2,13,2,1,1};
//    std::vector<bool> mask1 = getMask(att, NumCond, 2);
//    std::cout << "Mask 1: ";
//    for (bool b : mask1) {
//        std::cout << b << " ";
//    }
//    std::cout << std::endl;
//    std::cout << "Gaini: " << information_gain(labels1,mask1) << std::endl;
    //  generate combination num
    std::vector<int> nums = {1, 2, 3};
    std::vector<std::vector<int>> result;
    std::vector<int> path;
    generate_combinations(nums, 0, path, result);
    std::cout << "size comb: " << result.size() << std::endl;
    std::cout << std::endl;
    // cat option
    std::vector<int> a = {1, 2, 3};
    std::vector<std::vector<int>> options = categorical_options(a);
    std::cout << "size comb: " << options.size() << std::endl;
    std::cout << std::endl;
    // max split
    auto [max_gain, split_feature, is_numeric, flag] = max_information_gain_split(att, labels1,entropy);
    std::cout << "Test case 1 - Maximum information gain: " << max_gain << std::endl;
    std::cout << "Test case 1 - is numeric: " << is_numeric << std::endl;
    if (std::holds_alternative<std::string>(split_feature)) {
        std::cout << "Test case 1 - Split feature: " << std::get<std::string>(split_feature) << std::endl;
    }
    else if (std::holds_alternative<int>(split_feature)) {
        std::cout << "Test case 1 - Split feature: " << std::get<int>(split_feature) << std::endl;
    }
    else if (std::holds_alternative<double>(split_feature)) {
        std::cout << "Test case 1 - Split feature: " << std::get<double>(split_feature) << std::endl;
    }
    else if (std::holds_alternative<bool>(split_feature)) {
        std::cout << "Test case 1 - Split feature: " << std::get<bool>(split_feature) << std::endl;
    }
    std::cout << std::endl;
    // best split
    auto [var, val, ig, is_num] = get_best_split(className, data1);
    std::cout << "Test case 2 - var: " << var << std::endl;
    std::cout << "Test case 2 - ig: " << ig << std::endl;
    std::cout << "Test case 2 - is numeric: " << is_num << std::endl;
    if (std::holds_alternative<std::string>(val)) {
        std::cout << "Test case 2 - val: " << std::get<std::string>(val) << std::endl;
    }
    else if (std::holds_alternative<int>(val)) {
        std::cout << "Test case 2 - val: " << std::get<int>(val) << std::endl;
    }
    else if (std::holds_alternative<double>(val)) {
        std::cout << "Test case 2 - val: " << std::get<double>(val) << std::endl;
    }
    else if (std::holds_alternative<bool>(val)) {
        std::cout << "Test case 2 - val: " << std::get<bool>(val) << std::endl;
    }
    std::cout << std::endl;
    // split

    auto [data11, data12] = make_split(var, val, data1, is_num);
    std::cout << "data11:" << data11[className].size() << std::endl;
    std::cout << "data12:" << data12[className].size() << std::endl;
    // predict
    auto stringPrediction = makePrediction(data11[className], true);
    std::cout << "String Prediction: " ;
    for (const auto& cl:data11[className]){
        std::visit([](const auto& v) { std::cout << v; }, cl);
    }
    std::cout << std::endl;
    std::visit([](const auto& v) { std::cout << v; }, stringPrediction);
    std::cout << std::endl;

    stringPrediction = makePrediction(data12[className], true);
    std::cout << "String Prediction: " ;
    for (const auto& cl:data12[className]){
        std::visit([](const auto& v) { std::cout << v; }, cl);
    }
    std::cout << std::endl;
    std::visit([](const auto& v) { std::cout << v; }, stringPrediction);
    std::cout << std::endl;

    // stack
    TreeNode* root = interativeTrainTree(train_data,className,true);
    std::cout << "received " << typeid(root).name() << std::endl;
    if (root != nullptr) {
        if (std::holds_alternative<std::string>(root->condition)) {
            std::cout << std::get < std::string > (root->condition) << std::endl;
        }
    }

    return 0;
}