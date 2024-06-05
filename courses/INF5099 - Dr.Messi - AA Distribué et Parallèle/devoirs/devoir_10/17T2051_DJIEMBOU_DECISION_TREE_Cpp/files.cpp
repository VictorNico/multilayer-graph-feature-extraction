//
// Created by DJIEMBOU TIENCTHEU VICTOR NICO on 25/05/2024.
//

#include "headers/files.h"

// using namespace std;

//// Helper function to cast a string to the appropriate type

std::unordered_map<std::string, std::vector<float>> process_file(const std::string& file_path, const std::string& details_separator, bool header_flag) {
    /**
     * Processes a file and extracts the descriptions.
     *
     * @param file_path The path to the file.
     * @param details_separator The separator used for details in the file.
     * @param header_flag Indicates whether the file contains header information.
     * @return The descriptions extracted from the file.
     */
    std::unordered_map<std::string, std::vector<float>> descriptions;


    // Read the text file
    std::ifstream file(file_path);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    // Determine the start index based on the presence of a header line
    size_t start_index = header_flag ? 1 : 0;

    // Retrieve the header line if it exists
    std::vector<std::string> header;
    if (header_flag) {
        std::istringstream iss(lines[0]);
        std::string field;
        while (std::getline(iss, field, *details_separator.c_str())) {
            header.push_back(field.empty() ? "Index" : field);
        }
    }
    // Process each line and extract the descriptions
    for (size_t i = start_index; i < lines.size(); i++) {
        if (!lines[i].empty()) {
            std::istringstream iss(lines[i]);
            std::string field;
            size_t fieldIndex = 0;
            while (std::getline(iss, field, *details_separator.c_str())) {
                // Trim the field and convert to the appropriate type
                field = ltrim(rtrim(field));
                const std::type_info* typeInfo = getType(field);
                if (header.empty()) {
                    if (descriptions.find(std::to_string(fieldIndex)) == descriptions.end()) {
                        descriptions[std::to_string(fieldIndex)] = std::vector<float>();
                    }
                    descriptions[std::to_string(fieldIndex)].push_back(castToAppropriateType(field));

                } else {
                    if (descriptions.find(header[fieldIndex]) == descriptions.end()) {
                        descriptions[header[fieldIndex]] = std::vector<float>();
                    }
                        descriptions[header[fieldIndex]].push_back(castToAppropriateType(field));
                }
                fieldIndex++;
            }
        }
    }

    if (descriptions.find("Index") == descriptions.end()) {
        std::cout << "inside" <<std::endl;
        // Add the 'Index' column
        descriptions["Index"] = std::vector<float>();

        for (int i = 1; i < lines.size(); i++) {
            descriptions["Index"].push_back(i);
        }
    }
//    std::cout << "nb rows :" << descriptions[descriptions.begin()->first].size() <<std::endl;
    return descriptions;
}

//// Helper functions to trim strings
std::string ltrim(const std::string& s) {
    return std::string(std::find_if_not(s.begin(), s.end(), [](unsigned char ch) {
        return std::isspace(ch);
    }), s.end());
}

std::string rtrim(const std::string& s) {
    return std::string(s.begin(), std::find_if_not(s.rbegin(), s.rend(), [](unsigned char ch) {
        return std::isspace(ch);
    }).base());
}

float castToAppropriateType(const std::string& data) {
    try {
        return std::stof(data);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(data+" ne peut pas etre converti en floatant");
    }
}

const std::type_info* getType(const std::string& data) {
    if (std::all_of(data.begin(), data.end(), ::isdigit)) {
        return &typeid(int);
    }
    if (data == "true" || data == "false") {
        return &typeid(bool);
    }
    return &typeid(float);
}

std::tuple<
    std::unordered_map<std::string, std::vector<float>>,
    std::unordered_map<std::string, std::vector<float>>,
    std::vector<float>,
    std::vector<float>
    >
    train_test_split(
        std::unordered_map<std::string, std::vector<float>>& data,
        std::vector<float>& labels,
        double test_size = 0.2,
        int random_state = -1){
    // Vérifier que le nombre de labels correspond au nombre d'exemples
    if (data.empty() || data.begin()->second.size() != labels.size()) {
        throw std::invalid_argument("Le nombre de labels doit correspondre au nombre d'exemples");
    }

    // Initialiser le générateur de nombres aléatoires
    std::random_device rd;
    std::mt19937 gen(random_state == -1 ? rd() : static_cast<unsigned int>(random_state));
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Créer un vecteur d'index
    std::vector<size_t> indexes(labels.size());
    std::iota(indexes.begin(), indexes.end(), 0);

    // Mélanger les index de manière aléatoire
    std::shuffle(indexes.begin(), indexes.end(), gen);

    // Calculer le nombre d'exemples de test
    size_t test_count = static_cast<size_t>(std::ceil(indexes.size() * test_size));

    // Séparer les données d'entraînement et de test
    std::vector<float> train_labels(indexes.size() - test_count);
    std::vector<float> test_labels(test_count);
    for (size_t i = 0; i < indexes.size(); ++i) {
        if (i < indexes.size() - test_count) {
            train_labels[i] = labels[indexes[i]];
        } else {
            test_labels[i - (indexes.size() - test_count)] = labels[indexes[i]];
        }
    }

    // Créer les dictionnaires d'entraînement et de test
    std::unordered_map<std::string, std::vector<float>> train_data, test_data;
    for (const auto& key : data) {
        std::vector<float> train_values(indexes.size() - test_count);
        std::vector<float> test_values(test_count);
        for (size_t i = 0; i < indexes.size(); ++i) {
            if (i < indexes.size() - test_count) {
                train_values[i] = key.second[indexes[i]];
            } else {
                test_values[i - (indexes.size() - test_count)] = key.second[indexes[i]];
            }
        }
        train_data[key.first] = std::move(train_values);
        test_data[key.first] = std::move(test_values);
    }

    return std::make_tuple(std::move(train_data), std::move(test_data), std::move(train_labels), std::move(test_labels));
}

void write_dataset(
        const std::unordered_map<std::string, std::vector<float>>& train_data,
        const std::unordered_map<std::string, std::vector<float>>& test_data,
        const std::vector<float>& y_train,
        const std::vector<float>& y_test,
        const std::string& output_dir, const std::string& className) {
    // Créer le répertoire de sortie s'il n'existe pas déjà
    std::filesystem::create_directories(output_dir);

    // Écrire les données d'entraînement dans des fichiers
    std::ofstream train_file(output_dir + "/train.csv");
    for (const auto& pair : train_data) {
        train_file << pair.first << ",";
    }
    train_file << className << "\n";
//    train_file << "target\n";
//    std::cout << "hello" << std::endl;
    for (size_t i = 0; i < train_data.begin()->second.size(); ++i) {
        for (const auto& key : train_data) {
            train_file << key.second[i] << ",";
        }
        train_file << y_train[i] << "\n";    
    }
    train_file.close();

    // Écrire les données de test dans des fichiers
    std::ofstream test_file(output_dir + "/test.csv");
    for (const auto& pair : test_data) {
        test_file << pair.first << ",";
    }
    test_file << className << "\n";
    for (size_t i = 0; i < test_data.begin()->second.size(); ++i) {
        for (const auto& key : test_data) {
            test_file << key.second[i] << ",";
        }
        test_file << y_test[i] << "\n";
    }
    test_file.close();
}

void saveTreeModel(const std::string& filename, TreeNode* root) {
    std::ofstream outfile(filename, std::ios::binary);
    if (outfile.is_open()) {
        // Enregistrer l'arbre binaire dans le fichier
        saveTreeNode(outfile, root);
        outfile.close();
        std::cout << "Arbre binaire enregistré dans : " << filename << std::endl;
    } else {
        std::cerr << "Impossible d'ouvrir le fichier pour enregistrer l'arbre binaire." << std::endl;
    }
}

void saveTreeNode(std::ofstream& outfile, TreeNode* node) {
    if (node == nullptr) {
        outfile.write((char*)&node, sizeof(TreeNode*)); // Enregistrer un pointeur nul
    } else {
        outfile.write((char*)node, sizeof(TreeNode)); // Enregistrer les données du nœud
        saveTreeNode(outfile, node->left);
        saveTreeNode(outfile, node->right);
    }
}

TreeNode* loadTreeModel(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (infile.is_open()) {
        // Charger l'arbre binaire à partir du fichier
        TreeNode* root = loadTreeNode(infile);
        infile.close();
        std::cout << "Arbre binaire chargé depuis : " << filename << std::endl;
        return root;
    } else {
        std::cerr << "Impossible d'ouvrir le fichier pour charger l'arbre binaire." << std::endl;
        return nullptr;
    }
}

TreeNode* loadTreeNode(std::ifstream& infile) {
    TreeNode* node;
    infile.read((char*)&node, sizeof(TreeNode*));
    if (node != nullptr) {
        node = new TreeNode;
        infile.read((char*)node, sizeof(TreeNode));
        node->left = loadTreeNode(infile);
        node->right = loadTreeNode(infile);
    }
    return node;
}

void saveMetricsToCSV(const std::string& filename, const std::vector<std::string>& metricNames, const std::vector<double>& metricValues, const std::string& className, const std::string& version) {
    // Vérifier que le nombre de métriques correspond
    if (metricNames.size() != metricValues.size()) {
        std::cerr << "Nombre de noms de métriques et de valeurs de métriques différents." << std::endl;
        return;
    }

    // Obtenir l'heure actuelle
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::string timestamp = std::ctime(&now_c);
    timestamp.pop_back(); // Supprimer le caractère de fin de ligne

    // Ouvrir le fichier en mode ajout
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        // Écrire l'en-tête s'il n'existe pas déjà
        if (file.tellp() == 0) {
            file << "Temps";
            file << ",ClassName";
            file << ",Version";
            for (const auto& name : metricNames) {
                file << "," << name;
            }
            file << "\n";
        }

        // Écrire les données de métriques
        file << timestamp;
        file << "," << className;
        file << "," << version;
        for (const auto& value : metricValues) {
            file << "," << value;
        }
        file << "\n";

        file.close();
        std::cout << "Métriques enregistrées dans le fichier : " << filename << std::endl;
    } else {
        std::cerr << "Impossible d'ouvrir le fichier : " << filename << std::endl;
    }
}