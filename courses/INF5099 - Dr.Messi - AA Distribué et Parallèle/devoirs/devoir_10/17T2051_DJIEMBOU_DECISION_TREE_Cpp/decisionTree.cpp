//
// Created by DJIEMBOU TIENCTHEU VICTOR NICO on 25/05/2024.
//

#include "headers/metrics.h"
#include "headers/decisionTree.h"

//using namespace std;

std::unordered_map<std::string, std::vector<float>> data_; /** Result Matrix */
std::unordered_map<std::string, std::tuple<float, float, bool, bool>> masks_; /** Result Matrix */
int num_threads_ = -1; /** Number of thread to use */
std::string y_;

std::vector<NodeData> stack_trace;
std::mutex mutex_;
int avalaible_;
double duration;

float gini_impurity(const std::vector<float>& y) {
    if (y.empty()) {
        throw std::runtime_error("Input vector must not be empty.");
    }

    // Initialiser un dictionnaire vide pour stocker les comptes
    std::unordered_map<std::string, int> count_dict = count_elements(y);


    // Calculer la probabilité de chaque classe
    float total = y.size();
    std::vector<float> p(count_dict.size());
    int i = 0;
    for (auto& [key, val] : count_dict) {
        p[i++] = val / total;
    }

    // Calculer l'indice de Gini
    float gini = 1.0;
    for (float p1 : p) {
        gini -= p1 * p1;
    }

    return gini;
}

float entropy(const std::vector<float>& y) {
    // Vérifier si l'entrée est un vecteur
    if (y.empty()) {
        throw std::runtime_error("Input must be a non-empty vector.");
    }

    // Initialiser un dictionnaire vide pour stocker les comptes
    std::unordered_map<std::string, int> count_dict = count_elements(y);

    float total = y.size();
    float entropy = 0.0;
    float epsilon = 1e-9; // Petite valeur pour éviter log(0)

    // Calculer l'entropie
    for (auto& [key, val] : count_dict) {
        float p = val / total;
        p = std::max(p, epsilon); // Assurer que la valeur n'est pas zéro pour éviter log(0)
        entropy += -p * std::log2(p);
    }

    return entropy;
}

float variance(const std::vector<float>& y) {
    // Vérifier si l'entrée est un vecteur non vide
    if (y.empty()) {
        throw std::runtime_error("Input must be a non-empty vector.");
    }

    // Calculer la moyenne
    float mean = 0.0;
    int count = 0;
    for (const float& value : y) {
        mean += value;
        count++;
    }
    mean /= count;

    // Calculer la variance
    float variance = 0.0;
    for (const float& value : y) {
        variance += (value - mean) * (value - mean);
    }
    variance /= (count - 1);

    return variance;
}

///////////////////////////////////////////////////////////////////////////////

// Fonction de thread pour calculer une partie du masque
void process_thread(SharedData0& data, int thread_id) {
    for (size_t i = thread_id; i < data.X.size(); i += data.completed_threads) {
        // data.mutex.lock();
        (*data.mask)[i] = data.X[i] < data.val;
        // data.mutex.unlock();
        // std::cout << thread_id << "("<<data.X[i] << "," << data.val << ") ...> " << (*data.mask)[i] << " i: " << std::to_string(i) << std::endl;
    }
}

// Fonction parallèle pour obtenir le masque
std::vector<bool> getMaskParallel(const std::vector<float>& X, const float& val) {
    std::vector<bool> mask(X.size()); // Pré-allouer le vecteur de masque
    SharedData0 data { X, val, &mask ,std::mutex(), std::condition_variable(), num_threads_ };

    // Créer les threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads_; ++i) {
        threads.emplace_back(process_thread, std::ref(data), i);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    
    return mask;
}



// Fonction de thread pour compter les 'true' dans le masque
void process_thread_count(SharedData& data, int thread_id) {
    size_t l = 0; // Variable locale pour compter les "true"
    for (size_t i = thread_id; i < data.mask.size(); i += data.completed_threads) {
        if (data.mask[i]) {
            ++l;
            // std::cout << "<--->" << std::endl;
        }
    }
    std::atomic_size_t& a = data.a;
    a += l;
}


// Fonction parallèle pour compter les 'true' dans le masque
size_t count_true_parallel(const std::vector<bool>& mask, int num_threads) {
    SharedData data { mask, 0,std::mutex(), std::condition_variable(), num_threads_ };

    // Créer les threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads_; ++i) {
        threads.emplace_back(process_thread_count, std::ref(data), i);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    // std::cout << data.a << s td::endl;
    return data.a;
}


// Fonction de thread pour séparer les données
void process_thread_separate(SharedData2& data, int thread_id) {
    std::vector<bool> d1, d2;
    d1.reserve(data.y.size() / data.completed_threads); // Allouer de la mémoire pour d1
    d2.reserve(data.y.size() / data.completed_threads); // Allouer de la mémoire pour d2
    for (size_t i = thread_id; i < data.y.size(); i += data.completed_threads) {
        if (data.mask[i]) {
            data.mutex.lock();
            (*data.pos_data).push_back(data.y[i]);
            data.mutex.unlock();
        } else {
            data.mutex.lock();
            (*data.neg_data).push_back(data.y[i]);
            data.mutex.unlock();
        }
    }
    // data.mutex.lock();
    // (*data.pos_data).insert((*data.pos_data).end(), std::make_move_iterator(d1.begin()), std::make_move_iterator(d1.end()));
    // (*data.neg_data).insert((*data.neg_data).end(), std::make_move_iterator(d2.begin()), std::make_move_iterator(d2.end()));
    // data.mutex.unlock();
}

// Fonction parallèle pour séparer les données
void separate_data_parallel(const std::vector<float>& y, const std::vector<bool>& mask, std::vector<float>& pos_data, std::vector<float>& neg_data) {
    SharedData2 data { y, mask, &pos_data, &neg_data,std::mutex(), std::condition_variable(), num_threads_ };

    // Créer les threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads_; ++i) {
        threads.emplace_back(process_thread_separate, std::ref(data), i);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    
}

// Fonction parallèle pour calculer le gain d'information
float information_gain_parallel(const std::vector<float>& y,
                        const std::vector<bool>& mask,
                        std::function<float(const std::vector<float>&)> func) {
    size_t a = count_true_parallel(mask, num_threads_);
    size_t b = mask.size() - a;

    float ig = 0.0;
    if (a == 0 || b == 0) {
        ig = 0.0;
    } else {
        std::vector<float> pos_data, neg_data;
        separate_data_parallel(y, mask, pos_data, neg_data);

        ig = func(y) - (static_cast<float>(a) / (a + b) * func(pos_data)) -
                (static_cast<float>(b) / (a + b) * func(neg_data));
    }

    return ig;
}


///////////////////////////////////////////////////////////////////////////////////

// Fonction getMask
std::vector<bool> getMask(const std::vector<float>& X, const float& val) {
    std::vector<bool> mask;
    for (const auto& x : X) {
        mask.push_back(x < val);
    }
    return mask;
}

float information_gain(const std::vector<float>& y,
                        const std::vector<bool>& mask,
                        std::function<float(const std::vector<float>&)> func) {
    size_t a = 0;
    for (bool m : mask) {
        if (m) {
            a++;
        }
    }
    size_t b = mask.size() - a;

    float ig = 0.0;
    if (a == 0 || b == 0) {
        ig = 0.0;
    } else {
        std::vector<float> pos_data, neg_data;
        for (size_t i = 0; i < y.size(); ++i) {
            if (mask[i]) {
                pos_data.push_back(y[i]);
            } else {
                neg_data.push_back(y[i]);
            }
        }
        
        ig = func(y) - (static_cast<float>(a) / (a + b) * func(pos_data)) -
                (static_cast<float>(b) / (a + b) * func(neg_data));
        
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


std::tuple<float, float, bool, bool> max_information_gain_split(
        const std::vector<float>& x,
        const std::vector<float>& y,
        std::function<float(const std::vector<float>&)> func) {
    std::vector<float> split_value;
    std::vector<double> ig;

    // bool numeric_variable = true;
    // if (!x.empty()) {
    //     numeric_variable = std::holds_alternative<int>(x[0]) || std::holds_alternative<double>(x[0]);
    // }

    std::vector<float> options;
    // if (numeric_variable) {
    std::set<float> unique_options(x.begin(), x.end());
    options.assign(unique_options.begin(), unique_options.end());
    std::sort(options.begin(), options.end());
    // } else {
    //     std::set<float> unique_options(x.begin(), x.end());
    //     options.assign(unique_options.begin(), unique_options.end());
    // }

    for (const float& val : options) {
        std::vector<bool> mask;
        if (((Logic == 1) || (Logic == 3) ) && (x.size() > 200)) {
            // std::cout << "pp" << std::endl;
            mask= getMaskParallel(x,val);
        } else {
            mask = getMask(x,val);
        }
        float val_ig;
        if (((Logic == 2) || (Logic == 3) ) && (y.size() > 200)){
            val_ig = information_gain_parallel(y,mask);
        }else{
            val_ig = information_gain(y,mask);
        }
        ig.push_back(val_ig);
        split_value.push_back(val);
        // std::cout << val_ig << std::endl;
        // exit(1);
    }
    if (ig.empty()) {
        return std::make_tuple(0.0, 0.0, false, false);
    } else {
        auto best_ig_it = std::max_element(ig.begin(), ig.end());
        size_t best_ig_index = std::distance(ig.begin(), best_ig_it);
        float best_ig = *best_ig_it;
        float best_split = split_value[best_ig_index];
        return std::make_tuple(best_ig, best_split, true, true);
    }
}

bool customComparator(float a, float b) {
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

void iterate_search_(int thread_id) {
    // Utilisez atomic pour un accès thread-safe aux masks
    std::atomic<size_t> current_index(thread_id);

    while (current_index < data_.size()) {
        size_t i = current_index.fetch_add(num_threads_);
        if (i >= data_.size()) {
            break;
        }

        // Itérer sur la map en utilisant un itérateur
        auto it = data_.begin();
        std::advance(it, i);

        auto column = it->first; // Obtenir la clé (colonne)
        auto x = it->second; // Obtenir la valeur (vector<float>)
        if (column != y_) {
            auto [ig, split_feature, is_categorical, flag] =
                max_information_gain_split(x, data_.at(y_));
            masks_[column] = std::make_tuple(ig, split_feature, is_categorical, flag);
        }
    }
}

std::tuple<std::string, float, float, bool>
get_best_split(const std::string& y,
               const std::unordered_map<std::string, std::vector<float>>& data
               ) {
    
    std::unordered_map<std::string, std::tuple<float, float, bool, bool>> masks;
    if ((Logic == 4) || ((Logic == 7) && (num_threads_ - usedThreads) >= 2 )){
        y_ = y;
        data_ = data;
    // Créer les threads
        std::vector<std::thread> threads;
        for (int i = 0; i < (num_threads_ - usedThreads); ++i) {
            threads.emplace_back(iterate_search_, i);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    else{
        for (const auto& [column, x] : data) {
            if (column != y) {
                auto [ig, split_feature, is_categorical, flag] = max_information_gain_split(x, data.at(y));
                masks[column] = std::make_tuple(ig, split_feature, is_categorical, flag);
            }
        }
    }
    
    // else{
    //     // std::cout << num_threads_ << std::endl; 
    //     y_ = y;
    //     data_ = data;
    //     masks_ = masks;

    //     std::vector<std::thread> threads(num_threads_);

    //     for (int i = 0; i < num_threads_; ++i) {
    //         threads[i] = std::thread(iterate_search_, i);
    //     }

    //     for (auto& thread : threads) {
    //         thread.join();
    //     }
    //     masks = masks_;
    // }

    bool any_valid = false;
    for (const auto& [_, mask] : masks) {
        if (std::get<3>(mask)) {
            any_valid = true;
            break;
        }

    }

    if (!any_valid) {
        return std::make_tuple(std::string(), 0.0, 0.0, false);
    } else {
        std::vector<std::pair<std::string, float>> valid_columns;
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
        float split_value = std::get<1>(masks[split_variable]);
        float split_ig = std::get<0>(masks[split_variable]);
        bool split_numeric = std::get<2>(masks[split_variable]);

        return std::make_tuple(split_variable, split_value, split_ig, split_numeric);
    }
}

std::pair<std::unordered_map<std::string, std::vector<float>>,
        std::unordered_map<std::string, std::vector<float>>>
make_split(const std::string& variable,
           const float& value,
           const std::unordered_map<std::string, std::vector<float>>& data,
           bool is_numeric) {
    std::vector<size_t> index;
    std::vector<size_t> index2;

    for (size_t i = 0; i < data.at(variable).size(); ++i) {
        if (data.at(variable)[i] < value) {
            index.push_back(i);
        }else{
            index2.push_back(i);
        }
    }

    // for (size_t i = 0; i < data.at(variable).size(); ++i) {
    //     if (std::find(index.begin(), index.end(), i) == index.end()) {
    //         index2.push_back(i);
    //     }
    // }

    std::unordered_map<std::string, std::vector<float>> data_1;
    std::unordered_map<std::string, std::vector<float>> data_2;
    // if(Logic != 0){
    //     for (const auto& [key, values] : data) {
    //         std::cout << "Key: " << key << "->" << values.size() << std::endl;
    //     }
    // }
    for (const auto& [key, value_list] : data) {
        std::vector<float> temp1, temp2;
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

float makePrediction(const std::vector<float>& data, bool targetFactor) {
    std::unordered_map<float, size_t> floatFrequencyMap;

    for (const float& value : data) {
            floatFrequencyMap[value]++;
    }

    if (targetFactor) {
        // Find the mode (most common value) in the data
        auto it = std::max_element(floatFrequencyMap.begin(), floatFrequencyMap.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; });
        return it->first;
        
    } else {
        // Calculate the mean of the data
        float floatSum = 0.0;
        size_t floatCount = 0;

        for (const float& value : data) {
            floatSum += value;
            floatCount++;
        }

        return floatSum / static_cast<float>(floatCount);
        
    }

    // If no valid prediction can be made, return a default value
    return 0.0;
}

std::pair<bool, bool> checkConditionsDTree(int depth, const std::unordered_map<std::string, std::vector<float>>& xy_current, int max_categories, int max_depth, int counter, int min_samples_split) {
    // Check for depth conditions
    bool depth_cond;
    if (max_depth == -1) {
        depth_cond = true;
    } else {
//        std::cout << "count " << counter << std::endl;
        if (depth < max_depth) {
            depth_cond = true;
        } else {
            depth_cond = false;
        }
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

    return {depth_cond, sample_cond};
}

void sub_tree(
        TreeNode* current_node,
        std::string var,
        float ig,
        std::string question,
        int depth,
        float val,
        std::vector<NodeData>& stack,
        std::unordered_map<std::string, std::vector<float>> left,
        std::unordered_map<std::string, std::vector<float>> right) {
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
    if (num_threads_ != -1){ mutex_.lock();}
    stack.emplace_back(NodeData{std::move(left), next, current_node->left});
    stack.emplace_back(NodeData{std::move(right), next, current_node->right});
    if (num_threads_ != -1){ mutex_.unlock();}
//    std::cout << "stack" << std::endl;
}

void leaf_tree(std::unordered_map<std::string, std::vector<float>> xy_current,
               std::string y, bool target_factor, std::string var, float ig,
               float val,
               TreeNode* current_node) {
//    std::cout << "pre pred" << std::endl;
    float pred = makePrediction(xy_current[y], target_factor);
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
        std::unordered_map<std::string, std::vector<float>> xy_current,
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
        std::unordered_map<std::string, std::vector<float>> data,
        std::string y,
        bool target_factor,
        int max_depth,
        int min_samples_split,
        float min_information_gain,
        int counter,
        int max_categories) {

    std::vector<NodeData> stack;
    TreeNode* root = new TreeNode;;
    init_tree(data, stack, root);

    while (!stack.empty()) {
        NodeData current = std::move(stack.back());
        stack.pop_back();
        std::unordered_map<std::string, std::vector<float>> xy_current = current.data;
        int depthg = current.depth;
        TreeNode* current_node = std::move(current.node);

       // std::cout << "depth" << depthg << std::endl;
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
            
            if (ig != std::numeric_limits<float>::infinity() && ig >= min_information_gain) {
//                std::cout << "pre ms" << std::endl;
                auto [left, right] = make_split(var, val, xy_current, is_num);
//                std::cout << "ms" << std::endl;
//                std::cout << "data11:" << left[y].size() << std::endl;
//                std::cout << "data12:" << right[y].size() << std::endl;

                std::string split_type = is_num ? "<=" : "in";
                std::string question;

                question = var + " " + split_type + " " + std::to_string(val);
                // std::cout << "ok" << std::endl;
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
            leaf_tree(xy_current, y, target_factor, std::string(), std::numeric_limits<float>::infinity(), float(), current_node);
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

void DecisionTreeGrowth(
    GrowthShareData& GrowthShareData_,
    int thread_id
    ){
    for (size_t iter = thread_id; iter < stack_trace.size(); iter+=GrowthShareData_.nb_threads){
        NodeData current = stack_trace[iter];
        std::unordered_map<std::string, std::vector<float>> xy_current = current.data;
        int depthg = current.depth;
        TreeNode* current_node = std::move(current.node);
        std::pair<bool, bool> conditions = checkConditionsDTree(depthg, xy_current, GrowthShareData_.max_categories, GrowthShareData_.max_depth, 0, GrowthShareData_.min_samples_split);
        if (conditions.first && conditions.second) {
            auto [var, val, ig, is_num] = get_best_split(GrowthShareData_.y, xy_current);

            if (ig != std::numeric_limits<float>::infinity() && ig >= GrowthShareData_.min_information_gain) {
                auto [left, right] = make_split(var, val, xy_current, is_num);

                std::string split_type = is_num ? "<=" : "in";
                std::string question;

                question = var + " " + split_type + " " + std::to_string(val);

                sub_tree(current_node, var, ig, question, depthg, val, GrowthShareData_.stack, left, right);
            } else {
                leaf_tree(xy_current, GrowthShareData_.y, GrowthShareData_.target_factor, var, ig, val, current_node);
            }
        } else {
            leaf_tree(xy_current, GrowthShareData_.y, GrowthShareData_.target_factor, std::string(), std::numeric_limits<float>::infinity(), float(), current_node);
        }
        xy_current.clear();
        current.data.clear();
    }
    
}

TreeNode* interativeTrainTree_Parallel(
        std::unordered_map<std::string, std::vector<float>> data,
        std::string y,
        bool target_factor,
        int max_depth,
        int min_samples_split,
        float min_information_gain,
        int counter,
        int max_categories) {

    std::vector<NodeData> stack;
    TreeNode* root = new TreeNode;
    init_tree(data, stack, root);
    clock_t start, end;
    double duration;
    while (!stack.empty()) {
       stack_trace = std::move(stack);
       int nb_threads = stack_trace.size() < num_threads_ ? stack_trace.size() : num_threads_;
       avalaible_ = num_threads_ - nb_threads;
       stack.resize(0);
       // std::cout << nb_threads << " inse " << stack_trace.size() << std::endl;
       GrowthShareData GrowthShareData_ {stack,
                    nb_threads, 
                    y,
                    target_factor,
                    max_depth,
                    min_samples_split,
                    min_information_gain,
                    counter,
                    max_categories};
        
        if (nb_threads > 1){
           // std::cout << "inse" << std::endl;
            usedThreads = nb_threads;
            std::vector<std::thread> threads;
            for (int i = 0; i < nb_threads; i++){
                threads.emplace_back(
                    DecisionTreeGrowth, 
                    std::ref(GrowthShareData_),
                    i
                    );
            }
            for (auto& thread : threads) {
                thread.join();
            }
        }else{
            usedThreads = 0;
            DecisionTreeGrowth(std::ref(GrowthShareData_),0);
        }
        
        
        
    }
//    std::cout << "end"<< std::endl;
    return std::move(root);
}

std::vector<std::string> split_string(const std::string& input, const std::string& delimiter) {
    std::regex re(delimiter);
    std::sregex_token_iterator begin(input.begin(), input.end(), re, -1), end;
    return {begin, end};
}

float predict(
        const std::unordered_map<std::string, float>& observation,
        TreeNode* tree_node) {

    //  if root is leaf
    if (!std::holds_alternative<std::string>(tree_node->condition)) {
            return std::get<float>(tree_node->condition);
    }
    // Get the split condition from the current node
    const std::string& question = std::get<std::string>(tree_node->condition);
    std::vector<std::string> parts = split_string(question, " ");
    TreeNode* answer;
    const float& first_element = observation.at(parts[0]);
    if (parts[1] == "<=" ){
        if (first_element <= std::stod(parts[2])){
            answer = tree_node->left;
        } else{
            answer = tree_node->right;
        }
        
    }

    if (!std::holds_alternative<std::string>(answer->condition)) {
        return std::get<float>(answer->condition);
    }

    // If the current node is a leaf node, return the prediction
    return predict(observation, answer);
}

std::vector<float> predictions(
        std::unordered_map<std::string, std::vector<float>> test_data, TreeNode* IDT){
    // prediction vector
    std::vector<float> predict_values;

    for (size_t i = 0; i < test_data[test_data.begin()->first].size(); i++) {
        std::unordered_map<std::string, float> row;
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