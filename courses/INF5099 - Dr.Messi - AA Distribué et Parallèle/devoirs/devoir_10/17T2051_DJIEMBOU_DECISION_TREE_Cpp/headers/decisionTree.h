//
// Created by DJIEMBOU TIENCTHEU VICTOR NICO on 25/05/2024.
//

#ifndef DTREE_DECISIONTREE_H
#define DTREE_DECISIONTREE_H

#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <variant>
#include <numeric>
#include <sstream>
#include <set>
#include <regex>
#include <thread>
#include <mutex>
#include <atomic>
#include <unordered_set>
#include <condition_variable>



float gini_impurity(const std::vector<float>&);

float entropy(const std::vector<float>&);

float variance(const std::vector<float>&);

std::vector<bool> getMask(const std::vector<float>&, const float&);

// // Fonction NumCond
// auto NumCond = [](const auto& x, const auto& val) {
//     if (std::holds_alternative<int>(x) && std::holds_alternative<int>(val)) {
//         return std::get<int>(x) < std::get<int>(val);
//     }
//     else if (std::holds_alternative<double>(x) && std::holds_alternative<double>(val)) {
//         return std::get<double>(x) < std::get<double>(val);
//     }
//     else if (std::holds_alternative<int>(x) && std::holds_alternative<int>(val)) {
//         return std::get<int>(x) == std::get<int>(val);
//     }
//     else if (std::holds_alternative<double>(x) && std::holds_alternative<double>(val)) {
//         return std::get<double>(x) == std::get<double>(val);
//     }
//     else {
//         throw std::runtime_error("Cannot compare types");
//     }
// };

// // Fonction CatCond
// auto CatCond = [](const auto& x, const auto& val) {
//     if (std::holds_alternative<std::string>(x) && std::holds_alternative<std::string>(val)) {
//         return std::get<std::string>(x) == std::get<std::string>(val);
//     }
//     else if (std::holds_alternative<bool>(x) && std::holds_alternative<bool>(val)) {
//         return std::get<bool>(x) == std::get<bool>(val);
//     }
//     else if (std::holds_alternative<int>(x) && std::holds_alternative<int>(val)) {
//         return std::get<int>(x) == std::get<int>(val);
//     }
//     else if (std::holds_alternative<double>(x) && std::holds_alternative<double>(val)) {
//         return std::get<double>(x) == std::get<double>(val);
//     }
//     else {
//         throw std::runtime_error("Cannot compare types");
//     }
// };

float information_gain(const std::vector<float>&,
                        const std::vector<bool>&,
                        std::function<float(const std::vector<float>&)> func = gini_impurity);

void generate_combinations(const std::vector<int>&, int, std::vector<int>&, std::vector<std::vector<int>>&);

std::vector<std::vector<int>> categorical_options(const std::vector<int>&);

bool customComparator(double, double);

std::tuple<float, float, bool, bool> max_information_gain_split(
        const std::vector<float>&,
        const std::vector<float>&,
        std::function<float(const std::vector<float>&)> func = gini_impurity);

std::tuple<std::string, float, float, bool>
get_best_split(const std::string&,
               const std::unordered_map<std::string, std::vector<float>>&);

std::pair<std::unordered_map<std::string, std::vector<float>>,
std::unordered_map<std::string, std::vector<float>>>
make_split(const std::string&,
           const float&,
           const std::unordered_map<std::string, std::vector<float>>&,
           bool);

float makePrediction(const std::vector<float>&, bool targetFactor = 1);


std::pair<bool, bool> checkConditionsDTree(
        int,
        const std::unordered_map<std::string, std::vector<float>>&,
        int,
        int,
        int,
        int);

struct TreeNode {
        std::string col;
        float cutoff;
        std::variant<float, std::string> condition;
        int depth;
        float val;
        TreeNode* left;
        TreeNode* right;

};

struct NodeData {
    std::unordered_map<std::string, std::vector<float>> data;
    int depth;
    TreeNode* node;

};


void init_tree(
        std::unordered_map<std::string, std::vector<float>> xy_current,
        std::vector<NodeData>&,
        TreeNode*);

void sub_tree(
        TreeNode*,
        std::string,
        float,
        std::string,
        int,
        float,
        std::vector<NodeData>&,
        std::unordered_map<std::string, std::vector<float>>,
        std::unordered_map<std::string, std::vector<float>>);

void leaf_tree(
        std::unordered_map<std::string, std::vector<float>>,
        std::string,
        bool,
        std::string,
        float,
        float,
        TreeNode*);

TreeNode* interativeTrainTree(
        std::unordered_map<std::string,std::vector<float>>,
        std::string,
        bool,
        int max_depth = -1,
        int min_samples_split = 2,
        float min_information_gain = 1e-20,
        int counter = 0,
        int max_categories = 20);


float predict(
        const std::unordered_map<std::string, float>&,
        TreeNode*);

std::vector<std::string> split_string(const std::string&, const std::string&);

std::vector<float> predictions(
        std::unordered_map<std::string, std::vector<float>>, TreeNode*);

void destroyTree(TreeNode* node);

typedef struct {
    
} ThreadArgs;

// global instance
extern std::unordered_map<std::string, std::vector<float>> data_; /** Result Matrix */
extern std::unordered_map<std::string, std::tuple<float, float, bool, bool>> masks_; /** Result Matrix */
extern int num_threads_; /** Number of thread to use */
extern std::string y_;

extern int Logic;
extern int usedThreads;

// Structure de données partagées pour les threads
struct SharedData {
    const std::vector<bool>& mask;
    std::atomic<size_t> a;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<int> completed_threads;
};
// Structure de données partagées pour les threads
struct SharedData2 {
    const std::vector<float>& y;
    const std::vector<bool>& mask;
    std::vector<float>* pos_data;
    std::vector<float>* neg_data;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<int> completed_threads;
};
// Structure de données partagées pour les threads
struct SharedData0 {
    const std::vector<float>& X;
    const float val;
    std::vector<bool>* mask;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<int> completed_threads;
};

float information_gain_parallel(const std::vector<float>&, const std::vector<bool>&, std::function<float(const std::vector<float>&)> func=gini_impurity);

void separate_data_parallel(const std::vector<float>&, const std::vector<bool>&, std::vector<float>&, std::vector<float>&);

void process_thread_separate(SharedData2&, int);



size_t count_true_parallel(const std::vector<bool>&, int);

void process_thread_count(SharedData&, int);



std::vector<bool> getMaskParallel(const std::vector<float>&, const float&);

void process_thread(SharedData0&, int);


struct GrowthShareData {
    std::vector<NodeData>& stack;
    int nb_threads;
    std::string y;
    bool target_factor;
    int max_depth;
    int min_samples_split;
    float min_information_gain;
    int counter;
    int max_categories;
};


TreeNode* interativeTrainTree_Parallel(
        std::unordered_map<std::string, std::vector<float>>,
        std::string,
        bool,
        int,
        int,
        float,
        int,
        int);


void DecisionTreeGrowth(
    GrowthShareData& GrowthShareData_,
    int thread_id
    );


#endif //DTREE_DECISIONTREE_H
