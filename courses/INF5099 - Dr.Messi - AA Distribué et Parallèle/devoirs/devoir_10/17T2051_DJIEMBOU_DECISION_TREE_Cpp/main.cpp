//
// Created by DJIEMBOU TIENCTHEU VICTOR NICO on 28/05/2024.
//


#include "headers/main.h"


int Logic = -1;
int usedThreads = 0;
// -1 means sequentiel
// 1 means parallelism mask Computation 
// 2 means parallelism ig Computation
// 3 means parallelism mask and ig Computation
// 4 means parallelism best Split criteria search
// 5 means parallelism best Split criteria search and mask and ig
// 6 means parallelism ID growth
// 7 means parallelism ID growth and all bellow



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

void test_and_save(
    TreeNode* root, 
    std::unordered_map<std::string, std::vector<float>> test_data,
    std::unordered_map<std::string, std::vector<float>> train_data,
    std::vector<float> y_test,
    std::string approach,
    std::string className,
    int depth,
    int thread
    ){
     std::cout << "received " << typeid(root).name() << std::endl;
    if (root != nullptr) {
        if (std::holds_alternative<std::string>(root->condition)) {
            std::cout << "first tree node condition is : " << std::get<std::string>(root->condition) << std::endl;
        }
    }

    // prediction
    std::vector<float> pred = predictions(test_data, root);

    // confusion matrix
    std::set<float> unique_values;
    // Supposons que 'values' soit un ensemble de valeurs à convertir
    for (const auto& value : y_test) {
        unique_values.insert(value);
    }

    std::cout << "y_test " << y_test.size() << std::endl;
    std::cout << "pred " << pred.size() << std::endl;
    // Convertir l'ensemble en vecteur
    std::vector<float> unique_vector(unique_values.begin(), unique_values.end());

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
    std::vector<std::string> metricNames = {"Precision", "Accuracy", "Recall","F1_Score", "Num Threads", "max depth", "rows", "cols"};
    std::vector<double> metricValues = {precision(confusion_matrix), accuracy(confusion_matrix), recall(confusion_matrix), f1_score(confusion_matrix), static_cast<double>(thread), static_cast<double>(depth), static_cast<double>(train_data[className].size()), static_cast<double>(train_data.size())};
    std::string metricNamePath = "outputs/"+approach+"_"+className+".csv";
    saveMetricsToCSV(metricNamePath, metricNames, metricValues, className, approach);
    // save model
    // std::string modelNamePath = "outputs/"+approach+"_"+std::to_string(depth)+"_"+std::to_string(thread)+"_"+className+".bin";
    // saveTreeModel(modelNamePath, root);
    // delete model in memory
    destroyTree(root);
    
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <delimiter> <is_header>  <class_name>  <num_threads>  [<test_file> <is_classification> <max_depth> <min_samples_split> <min_information_gain> <max_categories>]" << std::endl;
        return 1;
    }
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
    std::unordered_map<std::string, std::vector<float>> train_data,test_data;
    std::vector<float> y_train,y_test;

    if(argc == 12){
        std::string train_file = argv[1];
        std::string test_file = argv[6];
        num_threads_ = std::stoi(argv[5]);
        className = argv[4];
        delimiter = argv[2];
        is_header = std::stoi(argv[3]);
        is_classification = std::stoi(argv[7]);
        max_depth = std::stoi(argv[8]);
        min_samples_split = std::stoi(argv[9]);
        min_information_gain = std::stod(argv[10]);
        max_categories = std::stoi(argv[11]);

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
    else if (argc == 6){
        // get args values
        std::string input_file = argv[1];
        className = argv[4];
        delimiter = argv[2];
        num_threads_ = std::stoi(argv[5]);
        is_header = std::stoi(argv[3]);

        // check if input file is valid
        bool exist = checkExistanceOfFile(input_file,delimiter);
        std::cout << "Train and test which are inside the same '" << input_file << "'" <<(exist ? "has been found."  : "hasn't been found." ) << std::endl;
        if (!exist) {
            std::cout << "use a valid csv file path" << std::endl;
            return 1;
        }

        // load data
        std::unordered_map<std::string, std::vector<float>> data = process_file(input_file, delimiter, is_header);
//        std::cout << argv[3] << std::endl;
        // gen train and test dataset
        for (const auto& [key, values] : data) {
            std::cout << key << ":" << values.size() << std::endl;
        }
        std::vector<float> labels = data[className];

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

    TreeNode* root;
    if (num_threads_ == -1){
        std::cout << "Yo" << std::endl;
        for (max_depth = -1; max_depth < 1; max_depth+=6){
            usedThreads = 0;
            // seq
            auto startSeq = std::chrono::high_resolution_clock::now();
            root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
            auto endSeq = std::chrono::high_resolution_clock::now();
            auto durationSeq = std::chrono::duration_cast<std::chrono::milliseconds>(endSeq - startSeq);
            test_and_save(
            root, 
            test_data,
            train_data,
            y_test,
            className,
            "Sequential_cpp",
            max_depth,
            num_threads_
            );
            for (num_threads_ = 2; num_threads_ < static_cast<int>(std::thread::hardware_concurrency())+1; num_threads_++){
                // -1 means sequentiel
                // 1 means parallelism mask Computation 
                // 2 means parallelism ig Computation
                // 3 means parallelism mask and ig Computation
                // 4 means parallelism best Split criteria search
                // 6 means parallelism ID growth
                // 7 means parallelism ID growth and ig
                
                std::cout << num_threads_ << std::endl;
                // exit(0);
                std::cout << train_data.at(className).size() << std::endl;
                usedThreads = 0;
                // par 1
                Logic = 1;
                auto startPar1 = std::chrono::high_resolution_clock::now();
                root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
                auto endPar1 = std::chrono::high_resolution_clock::now();
                auto durationPar1 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar1 - startPar1);
                test_and_save(
                root, 
                test_data,
                train_data,
                y_test,
                className,
                "Parallel_mask_cpp",
                max_depth,
                num_threads_
                );
                usedThreads = 0;
                // par 2
                Logic = 2;
                auto startPar2 = std::chrono::high_resolution_clock::now();
                root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
                auto endPar2 = std::chrono::high_resolution_clock::now();
                auto durationPar2 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar2 - startPar2);
                test_and_save(
                root, 
                test_data,
                train_data,
                y_test,
                className,
                "Parallel_ig_cpp",
                max_depth,
                num_threads_
                );
                usedThreads = 0;
                // par 3
                Logic = 3;
                auto startPar3 = std::chrono::high_resolution_clock::now();
                root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
                auto endPar3 = std::chrono::high_resolution_clock::now();
                auto durationPar3 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar3 - startPar3);
                test_and_save(
                root, 
                test_data,
                train_data,
                y_test,
                className,
                "Parallel_mask_and_ig_cpp",
                max_depth,
                num_threads_
                );
                usedThreads = 0;
                // par 4
                Logic = 4;
                auto startPar4 = std::chrono::high_resolution_clock::now();
                root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
                auto endPar4 = std::chrono::high_resolution_clock::now();
                auto durationPar4 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar4 - startPar4);
                test_and_save(
                root, 
                test_data,
                train_data,
                y_test,
                className,
                "Parallel_best_split_search_cpp",
                max_depth,
                num_threads_
                );
                usedThreads = 0;
                // par 6
                Logic = 6;
                auto startPar6 = std::chrono::high_resolution_clock::now();
                root = interativeTrainTree_Parallel(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
                auto endPar6 = std::chrono::high_resolution_clock::now();
                auto durationPar6 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar6 - startPar6);
                test_and_save(
                root, 
                test_data,
                train_data,
                y_test,
                className,
                "Parallel_ID_growth_cpp",
                max_depth,
                num_threads_
                );
                usedThreads = 0;
                // par 7
                Logic = 7;
                auto startPar7 = std::chrono::high_resolution_clock::now();
                root = interativeTrainTree_Parallel(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
                auto endPar7 = std::chrono::high_resolution_clock::now();
                auto durationPar7 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar7 - startPar7);
                test_and_save(
                root, 
                test_data,
                train_data,
                y_test,
                className,
                "Parallel_ID_growth_and_best_split_search_and_mask_and_ig_cpp",
                max_depth,
                num_threads_
                );
                usedThreads = 0;
                std::vector<std::string> metricNames = {
                    "Sequential Time", 
                    "Parallel 1 Time", 
                    "Parallel 2 Time",
                    "Parallel 3 Time", 
                    "Parallel 4 Time", 
                    "Parallel 6 Time", 
                    "Parallel 7 Time", 
                    "Max depth", 
                    "Num Threads",
                    "rows",
                    "cols"
                };
                std::vector<double> metricValues = {
                    static_cast<double>(durationSeq.count()), 
                    static_cast<double>(durationPar1.count()), 
                    static_cast<double>(durationPar2.count()), 
                    static_cast<double>(durationPar3.count()), 
                    static_cast<double>(durationPar4.count()), 
                    static_cast<double>(durationPar6.count()), 
                    static_cast<double>(durationPar7.count()), 
                    static_cast<double>(max_depth),
                    static_cast<double>(num_threads_),
                    static_cast<double>(train_data[className].size()), 
                    static_cast<double>(train_data.size())
                };
                std::string timeNamePath = "outputs/times_"+className+"_"+std::to_string(max_depth)+"_millisecond.csv";
                saveMetricsToCSV(timeNamePath, metricNames, metricValues, className, "times");

                metricNames = {
                    "Parallel 1 up", 
                    "Parallel 2 up",
                    "Parallel 3 up", 
                    "Parallel 4 up", 
                    "Parallel 6 up", 
                    "Parallel 7 up", 
                    "Max depth", 
                    "Num Threads",
                    "rows",
                    "cols"
                };
                metricValues = {
                    static_cast<double>(durationSeq.count())/static_cast<double>(durationPar1.count()), 
                    static_cast<double>(durationSeq.count())/static_cast<double>(durationPar2.count()), 
                    static_cast<double>(durationSeq.count())/static_cast<double>(durationPar3.count()), 
                    static_cast<double>(durationSeq.count())/static_cast<double>(durationPar4.count()), 
                    static_cast<double>(durationSeq.count())/static_cast<double>(durationPar6.count()), 
                    static_cast<double>(durationSeq.count())/static_cast<double>(durationPar7.count()), 
                    static_cast<double>(max_depth),
                    static_cast<double>(num_threads_),
                    static_cast<double>(train_data[className].size()), 
                    static_cast<double>(train_data.size())
                };
                timeNamePath = "outputs/speedup_"+className+"_"+std::to_string(max_depth)+"_millisecond.csv";
                saveMetricsToCSV(timeNamePath, metricNames, metricValues, className, "speedup");
            }
        }
    }else{
        auto startSeq = std::chrono::high_resolution_clock::now();
        root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
        auto endSeq = std::chrono::high_resolution_clock::now();
        auto durationSeq = std::chrono::duration_cast<std::chrono::milliseconds>(endSeq - startSeq);
        test_and_save(
        root, 
        test_data,
        train_data,
        y_test,
        className,
        "Sequential_cpp",
        max_depth,
        num_threads_
        );

        // par 1
        Logic = 1;
        auto startPar1 = std::chrono::high_resolution_clock::now();
        root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
        auto endPar1 = std::chrono::high_resolution_clock::now();
        auto durationPar1 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar1 - startPar1);
        test_and_save(
        root, 
        test_data,
        train_data,
        y_test,
        className,
        "Parallel_mask_cpp",
        max_depth,
        num_threads_
        );
        // par 2
        Logic = 2;
        auto startPar2 = std::chrono::high_resolution_clock::now();
        root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
        auto endPar2 = std::chrono::high_resolution_clock::now();
        auto durationPar2 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar2 - startPar2);
        test_and_save(
        root, 
        test_data,
        train_data,
        y_test,
        className,
        "Parallel_ig_cpp",
        max_depth,
        num_threads_
        );
        // par 3
        Logic = 3;
        auto startPar3 = std::chrono::high_resolution_clock::now();
        root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
        auto endPar3 = std::chrono::high_resolution_clock::now();
        auto durationPar3 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar3 - startPar3);
        test_and_save(
        root, 
        test_data,
        train_data,
        y_test,
        className,
        "Parallel_mask_and_ig_cpp",
        max_depth,
        num_threads_
        );
        // par 4
        Logic = 4;
        auto startPar4 = std::chrono::high_resolution_clock::now();
        root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
        auto endPar4 = std::chrono::high_resolution_clock::now();
        auto durationPar4 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar4 - startPar4);
        test_and_save(
        root, 
        test_data,
        train_data,
        y_test,
        className,
        "Parallel_best_split_search_cpp",
        max_depth,
        num_threads_
        );
        // par 5
        Logic = 5;
        auto startPar5 = std::chrono::high_resolution_clock::now();
        root = interativeTrainTree(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
        auto endPar5 = std::chrono::high_resolution_clock::now();
        auto durationPar5 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar5 - startPar5);
        test_and_save(
        root, 
        test_data,
        train_data,
        y_test,
        className,
        "Parallel_best_split_search_and_mask_and_ig_cpp",
        max_depth,
        num_threads_
        );
        // par 6
        Logic = 6;
        auto startPar6 = std::chrono::high_resolution_clock::now();
        root = interativeTrainTree_Parallel(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
        auto endPar6 = std::chrono::high_resolution_clock::now();
        auto durationPar6 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar6 - startPar6);
        test_and_save(
        root, 
        test_data,
        train_data,
        y_test,
        className,
        "Parallel_ID_growth_cpp",
        max_depth,
        num_threads_
        );
        // par 7
        Logic = 7;
        auto startPar7 = std::chrono::high_resolution_clock::now();
        root = interativeTrainTree_Parallel(train_data,className,is_classification,max_depth,min_samples_split,min_information_gain,counter,max_categories);
        auto endPar7 = std::chrono::high_resolution_clock::now();
        auto durationPar7 = std::chrono::duration_cast<std::chrono::milliseconds>(endPar7 - startPar7);
        test_and_save(
        root, 
        test_data,
        train_data,
        y_test,
        className,
        "Parallel_ID_growth_and_best_split_search_and_mask_and_ig_cpp",
        max_depth,
        num_threads_
        );

        std::vector<std::string> metricNames = {
            "Sequential Time", 
            "Parallel 1 Time", 
            "Parallel 2 Time",
            "Parallel 3 Time", 
            "Parallel 3 Time", 
            "Parallel 4 Time", 
            "Parallel 5 Time", 
            "Parallel 6 Time", 
            "Parallel 7 Time", 
            "Max depth", 
            "Num Threads",
            "rows",
            "cols"
        };
        std::vector<double> metricValues = {
            static_cast<double>(durationSeq.count()), 
            static_cast<double>(durationPar1.count()), 
            static_cast<double>(durationPar2.count()), 
            static_cast<double>(durationPar3.count()), 
            static_cast<double>(durationPar4.count()), 
            static_cast<double>(durationPar5.count()), 
            static_cast<double>(durationPar6.count()), 
            static_cast<double>(durationPar7.count()), 
            static_cast<double>(max_depth),
            static_cast<double>(num_threads_),
            static_cast<double>(train_data[className].size()), 
            static_cast<double>(train_data.size())
        };
        std::string timeNamePath = "outputs/times_"+className+"_"+std::to_string(max_depth)+"_millisecond.csv";
        saveMetricsToCSV(timeNamePath, metricNames, metricValues, className, "times");

        metricNames = {
            "Parallel 1 up", 
            "Parallel 2 up",
            "Parallel 3 up", 
            "Parallel 3 up", 
            "Parallel 4 up", 
            "Parallel 5 up", 
            "Parallel 6 up", 
            "Parallel 7 up", 
            "Max depth", 
            "Num Threads",
            "rows",
            "cols"
        };
        metricValues = {
            static_cast<double>(durationSeq.count())/static_cast<double>(durationPar1.count()), 
            static_cast<double>(durationSeq.count())/static_cast<double>(durationPar2.count()), 
            static_cast<double>(durationSeq.count())/static_cast<double>(durationPar3.count()), 
            static_cast<double>(durationSeq.count())/static_cast<double>(durationPar4.count()), 
            static_cast<double>(durationSeq.count())/static_cast<double>(durationPar5.count()), 
            static_cast<double>(durationSeq.count())/static_cast<double>(durationPar6.count()), 
            static_cast<double>(durationSeq.count())/static_cast<double>(durationPar7.count()), 
            static_cast<double>(max_depth),
            static_cast<double>(num_threads_),
            static_cast<double>(train_data[className].size()), 
            static_cast<double>(train_data.size())
        };
        timeNamePath = "outputs/speedup_"+className+"_"+std::to_string(max_depth)+"_millisecond.csv";
        saveMetricsToCSV(timeNamePath, metricNames, metricValues, className, "speedup");
    }
    train_data.clear();
    test_data.clear();
    
    return 0;
}







