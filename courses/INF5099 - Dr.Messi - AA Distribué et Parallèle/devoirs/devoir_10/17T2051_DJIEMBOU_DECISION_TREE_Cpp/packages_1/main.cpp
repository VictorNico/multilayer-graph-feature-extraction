//
// Created by DJIEMBOU TIENCTHEU VICTOR NICO on 28/05/2024.
//


#include "headers/main.h"
#include "headers/files.h"
#include "headers/metrics.h"
#include "headers/decisionTree.h"


bool checkExistanceOfFile(std::string path, std::string sep){
    if (path.length() >= 4 && path.substr(path.length() - 4) == ".csv") {

        // Vérifier l'existence du fichier
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Error: file '" << path << "' not found." << std::endl;
            return false;
        }
        std::string line;
        bool flag;
        if (std::getline(file, line)) {
            // Vérifier si la première ligne contient des virgules
            flag =  (line.find(sep) != std::string::npos);
        }
        // Le fichier existe, vous pouvez maintenant le traiter
        std::cout << "File '" << path << "' found." << std::endl;

        // Fermer le fichier
        file.close();
        return flag;
    }
    return false;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <delimiter> <is_header>  <class_name>  [<test_file> <is_classification> <max_depth> <min_samples_split> <min_information_gain> <max_categories>]" << std::endl;
        return 1;
    }
//    std::string className,
    bool is_classification = true;
    int max_depth = -1;
    int min_samples_split = 2;
    double min_information_gain = 1e-20;
    int counter = 0;
    int max_categories = 20;
    //// global variable

    std::string className;
    std::string delimiter = ",";
    bool is_header = true;
    //// args and data loading
    std::unordered_map<std::string, std::vector<double>> train_data,test_data;
    std::vector<double> y_train,y_test;

    if(argc == 11){
        std::string train_file = argv[1];
        std::string test_file = argv[5];
        className = argv[4];
        delimiter = argv[2];
        is_header = std::stoi(argv[3]);
        is_classification = std::stoi(argv[6]);
        max_depth = std::stoi(argv[7]);
        min_samples_split = std::stoi(argv[8]);
        min_information_gain = std::stod(argv[9]);
        max_categories = std::stoi(argv[10]);

        // check if input file is valid
        bool exist = checkExistanceOfFile(train_file,delimiter);
        std::cout << "Train and test which are inside the same '" << train_file << "'" <<(exist ? "has been found."  : "hasn't been found." ) << std::endl;
        if (!exist) {
            std::cout << "use a valid csv file path" << std::endl;
            return 1;
        }

        exist = checkExistanceOfFile(test_file,delimiter);
        std::cout << "Train and test which are inside the same '" << train_file << "'" <<(exist ? "has been found."  : "hasn't been found." ) << std::endl;
        if (!exist) {
            std::cout << "use a valid csv file path" << std::endl;
            return 1;
        }
        // load data
        train_data = process_file(train_file, delimiter, is_header);
//        for (const auto& [key, values] : train_data) {
//            std::cout << key << ":" << values.size() << std::endl;
//        }
        test_data = process_file(test_file, delimiter, is_header);
//        for (const auto& [key, values] : test_data) {
//            std::cout << key << ":" << values.size() << std::endl;
//        }

        // pretrait data
        y_test = test_data[className];
        std::cout << "y_test " << y_test.size() << std::endl;
        test_data.erase(className);
        test_data.erase("Index");
        train_data.erase("Index");

    }
    else if (argc == 5){
        // get args values
        std::string input_file = argv[1];
        className = argv[4];
        delimiter = argv[2];

        is_header = std::stoi(argv[3]);

        // check if input file is valid
        bool exist = checkExistanceOfFile(input_file,delimiter);
        std::cout << "Train and test which are inside the same '" << input_file << "'" <<(exist ? "has been found."  : "hasn't been found." ) << std::endl;
        if (!exist) {
            std::cout << "use a valid csv file path" << std::endl;
            return 1;
        }

        // load data
        std::unordered_map<std::string, std::vector<double>> data = process_file(input_file, delimiter, is_header);
//        std::cout << argv[3] << std::endl;
        // gen train and test dataset
        for (const auto& [key, values] : data) {
            std::cout << key << ":" << values.size() << std::endl;
        }
        std::vector<double> labels = data[className];

        data.erase(className);
        std::tie(train_data, test_data, y_train, y_test) = train_test_split(data, labels, 0.2, 42);
        train_data.erase("Index");
        test_data.erase("Index");
        write_dataset(train_data, test_data, y_train, y_test, "outputs", className);

        train_data[className] = y_train;
        std::cout << "train:" << train_data[className].size() << std::endl;
        std::cout << "test:" << y_test.size() << std::endl;
        data.clear();

    }

    // build tree
    auto start = std::chrono::high_resolution_clock::now();
    TreeNode* root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << "received " << typeid(root).name() << std::endl;
    if (root != nullptr) {
        if (std::holds_alternative<std::string>(root->condition)) {
            std::cout << "first tree node condition is : " << std::get<std::string>(root->condition) << std::endl;
        }
    }

    // prediction
    std::vector<double> pred = predictions(test_data, root);

    // confusion matrix
    std::set<double> unique_values;
    // Supposons que 'values' soit un ensemble de valeurs à convertir
    for (const auto& value : y_test) {
        unique_values.insert(value);
    }

    std::cout << "y_test " << y_test.size() << std::endl;
    std::cout << "pred " << pred.size() << std::endl;
    // Convertir l'ensemble en vecteur
    std::vector<double> unique_vector(unique_values.begin(), unique_values.end());

    std::vector<std::vector<int>> confusion_matrix = compute_confusion_matrix(y_test, pred, unique_vector);

    std::cout << "Confusion Matrix:\n";
    for (const auto& row : confusion_matrix) {
        for (int value : row) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }

    // print metrics
    std::cout << "precision: " << precision(confusion_matrix) << std::endl;
    std::cout << "Accuracy: " << accuracy(confusion_matrix) << std::endl;
    std::cout << "recall: " << recall(confusion_matrix) << std::endl;
    std::cout << "f1: " << f1_score(confusion_matrix) << std::endl;
    // save metrics
    std::vector<std::string> metricNames = {"Precision", "Accuracy", "Recall","F1_Score","elapsed time (s)", "max depth", "rows", "cols"};
    std::vector<double> metricValues = {precision(confusion_matrix), accuracy(confusion_matrix), recall(confusion_matrix), f1_score(confusion_matrix), duration.count(), static_cast<double>(max_depth), static_cast<double>(train_data[className].size()), static_cast<double>(train_data.size())};
    saveMetricsToCSV("outputs/metrics.csv", metricNames, metricValues, className, "Sequential C++");
    // save model
    std::string modelNamePath = "outputs/"+className+".bin";
    saveTreeModel(modelNamePath, root);
    // delete model in memory
    destroyTree(root);
    train_data.clear();
    test_data.clear();

    return 0;
}