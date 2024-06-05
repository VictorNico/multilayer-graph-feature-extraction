//
// Created by DJIEMBOU TIENCTHEU VICTOR NICO on 25/05/2024.
//

#include "headers/metrics.h"
#include "headers/decisionTree.h"

//using namespace std;

double gini_impurity(const std::vector<std::variant<int, double, bool, std::string>>& y) {
    if (y.empty()) {
        throw std::runtime_error("Input vector must not be empty.");
    }

    // Initialiser un dictionnaire vide pour stocker les comptes
    std::unordered_map<std::string, int> count_dict = count_elements(y);


    // Calculer la probabilité de chaque classe
    double total = y.size();
    std::vector<double> p(count_dict.size());
    int i = 0;
    for (auto& [key, val] : count_dict) {
        p[i++] = val / total;
    }

    // Calculer l'indice de Gini
    double gini = 1.0;
    for (double p1 : p) {
        gini -= p1 * p1;
    }

    return gini;
}

double entropy(const std::vector<std::variant<int, double, bool, std::string>>& y) {
    // Vérifier si l'entrée est un vecteur
    if (y.empty()) {
        throw std::runtime_error("Input must be a non-empty vector.");
    }

    // Initialiser un dictionnaire vide pour stocker les comptes
    std::unordered_map<std::string, int> count_dict = count_elements(y);

    double total = y.size();
    double entropy = 0.0;
    double epsilon = 1e-9; // Petite valeur pour éviter log(0)

    // Calculer l'entropie
    for (auto& [key, val] : count_dict) {
        double p = val / total;
        p = std::max(p, epsilon); // Assurer que la valeur n'est pas zéro pour éviter log(0)
        entropy += -p * std::log2(p);
    }

    return entropy;
}

double variance(const std::vector<std::variant<int, double, bool, std::string>>& y) {
    // Vérifier si l'entrée est un vecteur non vide
    if (y.empty()) {
        throw std::runtime_error("Input must be a non-empty vector.");
    }

    // Calculer la moyenne
    double mean = 0.0;
    int count = 0;
    for (const auto& value : y) {
        double val;
        std::stringstream ss;
        std::visit([&ss, &val](const auto& v) { ss << v; val = std::stod(ss.str()); }, value);
        mean += val;
        count++;
    }
    mean /= count;

    // Calculer la variance
    double variance = 0.0;
    for (const auto& value : y) {
        double val;
        std::stringstream ss;
        std::visit([&ss, &val](const auto& v) { ss << v; val = std::stod(ss.str()); }, value);
        variance += (val - mean) * (val - mean);
    }
    variance /= (count - 1);

    return variance;
}

// Fonction getMask
std::vector<bool> getMask(const std::vector<std::variant<int, double, bool, std::string>>& X, std::function<bool(const std::variant<int, double, bool, std::string>&, const std::variant<int, double, bool, std::string>&)> cond, const std::variant<int, double, bool, std::string>& val) {
    std::vector<bool> mask;
    for (const auto& x : X) {
        mask.push_back(std::visit([&](const auto& v) { return std::visit([&](const auto& w) {
            return cond(v, w);
            }, val);
        }, x));
    }
    return mask;
}

double information_gain(const std::vector<std::variant<int, double, bool, std::string>>& y,
                        const std::vector<bool>& mask,
                        std::function<double(const std::vector<std::variant<int, double, bool, std::string>>&)> func) {
    size_t a = 0;
    for (bool m : mask) {
        if (m) {
            a++;
        }
    }
    size_t b = mask.size() - a;

    double ig = 0.0;
    if (a == 0 || b == 0) {
        ig = 0.0;
    } else {
        std::vector<std::variant<int, double, bool, std::string>> pos_data, neg_data;
        for (size_t i = 0; i < y.size(); ++i) {
            if (mask[i]) {
                pos_data.push_back(y[i]);
            } else {
                neg_data.push_back(y[i]);
            }
        }
        if (std::holds_alternative<int>(y[0]) || std::holds_alternative<double>(y[0])) {
            // std::cout << "@";
            ig = variance(y) - (static_cast<double>(a) / (a + b) * variance(pos_data)) -
                 (static_cast<double>(b) / (a + b) * variance(neg_data));
        } else {
            ig = func(y) - (static_cast<double>(a) / (a + b) * func(pos_data)) -
                    (static_cast<double>(b) / (a + b) * func(neg_data));
        }
    }

    return ig;
}

void generate_combinations(const std::vector<int>& nums, int start, std::vector<int>& path, std::vector<std::vector<int>>& result) {
    if (start == static_cast<int>(nums.size())) {
        if (!path.empty()) {
            result.push_back(path);
        }
        return;
    }

    generate_combinations(nums, start + 1, path, result);
    path.push_back(nums[start]);
    generate_combinations(nums, start + 1, path, result);
    path.pop_back();
}

std::vector<std::vector<int>> categorical_options(const std::vector<int>& a) {
    std::set<int> unique_vals(a.begin(), a.end());
    std::vector<int> unique_vals_vec(unique_vals.begin(), unique_vals.end());
    std::vector<std::vector<int>> options;
    std::vector<int> path;
    generate_combinations(unique_vals_vec, 0, path, options);

    // Supprime le premier et le dernier éléments de l'options
    return std::vector<std::vector<int>>(options.begin() + 1, options.end() - 1);
}


std::tuple<double, std::variant<int, double, bool, std::string>, bool, bool> max_information_gain_split(
        const std::vector<std::variant<int, double, bool, std::string>>& x,
        const std::vector<std::variant<int, double, bool, std::string>>& y,
        std::function<double(const std::vector<std::variant<int, double, bool, std::string>>&)> func) {
    std::vector<std::variant<int, double, bool, std::string>> split_value;
    std::vector<double> ig;

    bool numeric_variable = true;
    if (!x.empty()) {
        numeric_variable = std::holds_alternative<int>(x[0]) || std::holds_alternative<double>(x[0]);
    }

    std::vector<std::variant<int, double, bool, std::string>> options;
    if (numeric_variable) {
        std::set<std::variant<int, double, bool, std::string>> unique_options(x.begin(), x.end());
        options.assign(unique_options.begin(), unique_options.end());
        std::sort(options.begin(), options.end());
    } else {
        std::set<std::variant<int, double, bool, std::string>> unique_options(x.begin(), x.end());
        options.assign(unique_options.begin(), unique_options.end());
    }

    for (const auto& val : options) {
        std::vector< bool> mask;
        if (numeric_variable) {
            mask = getMask(x, NumCond,val);
        } else {
            mask= getMask(x, CatCond,val);
        }
        double val_ig = information_gain(y,mask);
        ig.push_back(val_ig);
        split_value.push_back(val);
    }
    if (ig.empty()) {
        return std::make_tuple(0.0, 0.0, false, false);
    } else {
        auto best_ig_it = std::max_element(ig.begin(), ig.end());
        size_t best_ig_index = std::distance(ig.begin(), best_ig_it);
        double best_ig = *best_ig_it;
        std::variant<int, double, bool, std::string> best_split = split_value[best_ig_index];
        return std::make_tuple(best_ig, best_split, numeric_variable, true);
    }
}

bool customComparator(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) {
        return false; // Les NaN sont égaux
    } else if (std::isnan(a)) {
        return true; // Les NaN sont plus petits
    } else if (std::isnan(b)) {
        return false; // Les NaN sont plus petits
    } else {
        return a > b; // Comparaison numérique standard
    }
}

std::tuple<std::string, std::variant<int, double, bool, std::string>, double, bool>
get_best_split(const std::string& y,
               const std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>& data) {
    std::unordered_map<std::string, std::tuple<double, std::variant<int, double, bool, std::string>, bool, bool>> masks;

    for (const auto& [column, x] : data) {
        if (column != y) {
            auto [ig, split_feature, is_categorical, flag] = max_information_gain_split(x, data.at(y));
            masks[column] = std::make_tuple(ig, split_feature, is_categorical, flag);
        }
    }

    bool any_valid = false;
    for (const auto& [_, mask] : masks) {
        if (std::get<3>(mask)) {
            any_valid = true;
            break;
        }
    }

    if (!any_valid) {
        return std::make_tuple(std::string(), std::variant<int, double, bool, std::string>(), 0.0, false);
    } else {
        std::vector<std::pair<std::string, double>> valid_columns;
        for (const auto& [column, mask] : masks) {
            if (std::get<3>(mask)) {
                valid_columns.emplace_back(column, std::get<0>(mask));
            }
        }
        std::sort(valid_columns.begin(), valid_columns.end(),
        [](const auto& a, const auto& b) { return customComparator(a.second, b.second); });

        // for (const auto& [col, ig]:valid_columns){
        //     std::cout << col << " " << ig << std::endl;
        // }

        std::string split_variable = valid_columns[0].first;
        std::variant<int, double, bool, std::string> split_value = std::get<1>(masks[split_variable]);
        double split_ig = std::get<0>(masks[split_variable]);
        bool split_numeric = std::get<2>(masks[split_variable]);

        return std::make_tuple(split_variable, split_value, split_ig, split_numeric);
    }
}

std::pair<std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>,
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>>
make_split(const std::string& variable,
           const std::variant<int, double, bool, std::string>& value,
           const std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>& data,
           bool is_numeric) {
    std::vector<size_t> index;
    std::vector<size_t> index2;

    if (is_numeric) {
        for (size_t i = 0; i < data.at(variable).size(); ++i) {
            if (std::holds_alternative<int>(data.at(variable)[i])) {
                if (std::get<int>(data.at(variable)[i]) < std::get<int>(value)) {
                    index.push_back(i);
                }
            } else if (std::holds_alternative<double>(data.at(variable)[i])) {
                if (std::get<double>(data.at(variable)[i]) < std::get<double>(value)) {
                    index.push_back(i);
                }
            }

        }

    } else {
        for (size_t i = 0; i < data.at(variable).size(); ++i) {
            if (data.at(variable)[i] == value) {
                index.push_back(i);
            }
        }
    }
    for (size_t i = 0; i < data.at(variable).size(); ++i) {
        if (std::find(index.begin(), index.end(), i) == index.end()) {
            index2.push_back(i);
        }
    }

    std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> data_1;
    std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> data_2;

    for (const auto& [key, value_list] : data) {
        std::vector<std::variant<int, double, bool, std::string>> temp1, temp2;
        for (size_t i : index) {
            temp1.push_back(value_list[i]);
        }
        for (size_t i : index2) {
            temp2.push_back(value_list[i]);
        }

        data_1[key] = std::move(temp1);
        data_2[key] = std::move(temp2);
    }

    return {data_1, data_2};
}

std::variant<int, double, bool, std::string> makePrediction(const std::vector<std::variant<int, double, bool, std::string>>& data, bool targetFactor) {
    std::unordered_map<int, size_t> intFrequencyMap;
    std::unordered_map<double, size_t> doubleFrequencyMap;
    std::unordered_map<bool, size_t> boolFrequencyMap;
    std::unordered_map<std::string, size_t> stringFrequencyMap;

    for (const auto& value : data) {
        if (std::holds_alternative<int>(value)) {
            intFrequencyMap[std::get<int>(value)]++;
        } else if (std::holds_alternative<double>(value)) {
            doubleFrequencyMap[std::get<double>(value)]++;
        } else if (std::holds_alternative<bool>(value)) {
            boolFrequencyMap[std::get<bool>(value)]++;
        } else if (std::holds_alternative<std::string>(value)) {
            stringFrequencyMap[std::get<std::string>(value)]++;
        }
    }

    if (targetFactor) {
        // Find the mode (most common value) in the data
        if (!intFrequencyMap.empty()) {
            auto it = std::max_element(intFrequencyMap.begin(), intFrequencyMap.end(),
                                       [](const auto& a, const auto& b) { return a.second < b.second; });
            return it->first;
        } else if (!doubleFrequencyMap.empty()) {
            auto it = std::max_element(doubleFrequencyMap.begin(), doubleFrequencyMap.end(),
                                       [](const auto& a, const auto& b) { return a.second < b.second; });
            return it->first;
        } else if (!boolFrequencyMap.empty()) {
            auto it = std::max_element(boolFrequencyMap.begin(), boolFrequencyMap.end(),
                                       [](const auto& a, const auto& b) { return a.second < b.second; });
            return it->first;
        } else if (!stringFrequencyMap.empty()) {
            auto it = std::max_element(stringFrequencyMap.begin(), stringFrequencyMap.end(),
                                       [](const auto& a, const auto& b) { return a.second < b.second; });
            return it->first;
        }
    } else {
        // Calculate the mean of the data
        int intSum = 0;
        size_t intCount = 0;
        double doubleSum = 0.0;
        size_t doubleCount = 0;

        for (const auto& value : data) {
            if (std::holds_alternative<int>(value)) {
                intSum += std::get<int>(value);
                intCount++;
            } else if (std::holds_alternative<double>(value)) {
                doubleSum += std::get<double>(value);
                doubleCount++;
            }
        }

        if (intCount > 0) {
            return static_cast<double>(intSum) / static_cast<double>(intCount);
        } else if (doubleCount > 0) {
            return doubleSum / static_cast<double>(doubleCount);
        }
    }

    // If no valid prediction can be made, return a default value
    return std::variant<int, double, bool, std::string>();
}

std::pair<bool, bool> checkConditionsDTree(int depth, const std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>& xy_current, int max_categories, int max_depth, int counter, int min_samples_split) {
    // Check for depth conditions
    bool depth_cond;
    // std::cout << "here " << max_depth << std::endl;
    if (max_depth == -1) {
        depth_cond = true;
    } else {
//        std::cout << "count " << counter << std::endl;
        if (depth < max_depth) {
            depth_cond = true;
        } else {
            depth_cond = false;
        }
        // std::cout << "count " << depth << " : " << depth_cond << std::endl;
    }

    // Check for sample conditions
    bool sample_cond;
    if (min_samples_split == -1) {
        sample_cond = true;
    } else {
        if (xy_current.begin()->second.size() > min_samples_split) {
            sample_cond = true;
        } else {
            sample_cond = false;
        }
    }

    // Check for category conditions
    if (depth == 0) {
        for (const auto& [key, values] : xy_current) {
            if (std::holds_alternative<std::string>(values[0])) {
                std::set<std::variant<int, double, bool, std::string>> unique_values;
                for (const auto& value : values) {
                    unique_values.insert(value);
                }
                int variable_length = static_cast<int>(unique_values.size());
                if (variable_length > max_categories) {
                    throw std::runtime_error("The variable " + key + " has " + std::to_string(variable_length) + " unique values, which is more than the accepted ones: " + std::to_string(max_categories));
                }
            }
        }
    }

    return {depth_cond, sample_cond};
}

void sub_tree(
        TreeNode* current_node,
        std::string var,
        double ig,
        std::string question,
        int depth,
        std::variant<int, double, bool, std::string> val,
        std::vector<NodeData>& stack,
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> left,
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> right) {
    // Update current node with split information
//    std::cout << "pred curr" << std::endl;
    current_node->col = std::move(var);
    current_node->cutoff = ig;
    current_node->val = std::move(val);
    current_node->condition = std::move(question);
    current_node->depth = depth;
//    std::cout << "curr" << std::endl;

    // Create left and right child nodes
//    std::cout << "pred left right" << std::endl;
    current_node->left = new TreeNode;
    current_node->right = new TreeNode;
//    std::cout << "left right" << std::endl;

    // Push child nodes onto the stack
//    std::cout << "pred stack" << (depth + 1) << std::endl;
    int next = depth + 1;
    stack.emplace_back(NodeData{std::move(left), next, current_node->left});
    stack.emplace_back(NodeData{std::move(right), next, current_node->right});
//    std::cout << "stack" << std::endl;
}

void leaf_tree(std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> xy_current,
               std::string y, bool target_factor, std::string var, double ig,
               std::variant<int, double, bool, std::string> val,
               TreeNode* current_node) {
//    std::cout << "pre pred" << std::endl;
    std::variant<int, double, bool, std::string> pred = makePrediction(xy_current[y], target_factor);
    // print(f'{col} - {val} - {cutoff} - condition: {pred}')
   // std::cout << var << " - ";
   // std::visit([](const auto& value) {
   //      std::cout << value;
   //  }, val);
   // std::cout << " - " << ig << " - ";
   // std::visit([](const auto& value) {
   //      std::cout << value << std::endl;
   //  }, pred);

    // Utiliser std::visit() pour afficher le contenu du variant
    
//    std::cout << "pre leaf" << std::endl;
    current_node->col = std::move(var);
    current_node->cutoff = std::move(ig);
    current_node->val = std::move(val);
    current_node->condition = std::move(pred);
//    std::cout << "leaf" << std::endl;
}

void init_tree(
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> xy_current,
        std::vector<NodeData>& stack,
        TreeNode* root) {
    NodeData n = {
            std::move(xy_current),
            0,
            root
    };
//    std::cout << "root null" << (root == nullptr) << std::endl;
    stack.push_back(std::move(n));
}


TreeNode* interativeTrainTree(
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> data,
        std::string y,
        bool target_factor,
        int max_depth,
        int min_samples_split,
        double min_information_gain,
        int counter,
        int max_categories) {

    std::vector<NodeData> stack;
    TreeNode* root = new TreeNode;;
    init_tree(data, stack, root);

    while (!stack.empty()) {
        NodeData current = std::move(stack.back());
        stack.pop_back();
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> xy_current = current.data;
        int depthg = current.depth;
        TreeNode* current_node = std::move(current.node);

//        std::cout << "depth" << depthg << std::endl;
//        std::cout << "pre cond" << stack.size() << std::endl;
        std::pair<bool, bool> conditions = checkConditionsDTree(depthg, xy_current, max_categories, max_depth, counter, min_samples_split);
//        std::cout << "cond" << std::endl;
        if (conditions.first && conditions.second) {
//            std::cout << "pre bs" << std::endl;
            auto [var, val, ig, is_num] = get_best_split(y, xy_current);
//            std::cout << "bs " << var << std::endl;
            // std::cout << var << " - ";
            // std::visit([](const auto& value) {
            // std::cout << value;
            // }, val);
            // std::cout << " - " << ig << " - " << is_num << std::endl;

            if (ig != std::numeric_limits<double>::infinity() && ig >= min_information_gain) {
//                std::cout << "pre ms" << std::endl;
                auto [left, right] = make_split(var, val, xy_current, is_num);
//                std::cout << "ms" << std::endl;
//                std::cout << "data11:" << left[y].size() << std::endl;
//                std::cout << "data12:" << right[y].size() << std::endl;

                std::string split_type = is_num ? "<=" : "in";
                std::string question;

                if (std::holds_alternative<std::string>(val)) {
                    question = var + " " + split_type + " " + std::get<std::string>(val);
                }
                else if (std::holds_alternative<int>(val)) {
                    question = var + " " + split_type + " " + std::to_string(std::get<int>(val));
                }
                else if (std::holds_alternative<double>(val)) {
                    question = var + " " + split_type + " " + std::to_string(std::get<double>(val));
                }
                else if (std::holds_alternative<bool>(val)) {
                    question = var + " " + split_type + " " + std::to_string(std::get<bool>(val));
                }

//                std::cout << "pre Sub "<< std::endl ;
                sub_tree(current_node, var, ig, question, depthg, val, stack, left, right);
//                std::cout << "Sub " << (root == nullptr )<< std::endl;
            } else {
//                std::cout << "pre leaf 1"<< std::endl;
                leaf_tree(xy_current, y, target_factor, var, ig, val, current_node);
//                std::cout << "leaf 1"<< std::endl;
            }
        } else {
//            std::cout << "pre leaf 2" << std::endl;
            leaf_tree(xy_current, y, target_factor, std::string(), std::numeric_limits<double>::infinity(), std::variant<int, double, bool, std::string>(), current_node);
//            std::cout << "leaf 2"<< std::endl;
        }
        // std::cout << "ok"<< std::endl;
        xy_current.clear();
        // std::cout << "ok"<< std::endl;
        current.data.clear();
    }
//    std::cout << "end"<< std::endl;
    return std::move(root);
}

std::vector<std::string> split_string(const std::string& input, const std::string& delimiter) {
    std::regex re(delimiter);
    std::sregex_token_iterator begin(input.begin(), input.end(), re, -1), end;
    return {begin, end};
}

std::variant<int, double, bool, std::string> predict(
        const std::unordered_map<std::string, std::variant<int, double, bool, std::string>>& observation,
        TreeNode* tree_node) {

    //  if root is leaf
    if (!std::holds_alternative<std::string>(tree_node->condition)) {
            return tree_node->condition;
    }
    // Get the split condition from the current node
    const std::string& question = std::get<std::string>(tree_node->condition);
    std::vector<std::string> parts = split_string(question, " ");
    TreeNode* answer;
    const auto& first_element = observation.at(parts[0]);
    if (parts[1] == "<=" ){
        if (std::holds_alternative<int>(first_element)) {
            if (std::get<int>(first_element) <= std::stoi(parts[2])){
                answer = tree_node->left;
            } else{
                answer = tree_node->right;
            }
        }
        else if (std::holds_alternative<double>(first_element)) {
            if (std::get<double>(first_element) <= std::stod(parts[2])){
                answer = tree_node->left;
            } else{
                answer = tree_node->right;
            }
        }
    } else if (parts[1] == "in" ){

        if (std::holds_alternative<std::string>(first_element)) {
            if (std::get<std::string>(first_element) == parts[2]){
                answer = tree_node->left;
            } else{
                answer = tree_node->right;
            }
        }
    }

    if (!std::holds_alternative<std::string>(answer->condition)) {
        return answer->condition;
    }

    // If the current node is a leaf node, return the prediction
    return predict(observation, answer);
}

std::vector<std::variant<int, double, bool, std::string>> predictions(
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> test_data, TreeNode* IDT){
    // prediction vector
    std::vector<std::variant<int, double, bool, std::string>> predict_values;

    for (size_t i = 0; i < test_data[test_data.begin()->first].size(); i++) {
        std::unordered_map<std::string, std::variant<int, double, bool, std::string>> row;
        for (const auto& [key, values] : test_data) {
            row[key] = values[i];
        }
        predict_values.push_back(predict(row, IDT));
    }
    return predict_values;
}

void destroyTree(TreeNode* node) {
    if (node != nullptr) {
        destroyTree(node->left);
        destroyTree(node->right);
        delete node;
    }
}