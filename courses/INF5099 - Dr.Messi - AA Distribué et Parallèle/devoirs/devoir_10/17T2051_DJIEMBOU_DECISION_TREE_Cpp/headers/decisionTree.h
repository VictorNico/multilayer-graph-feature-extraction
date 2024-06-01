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



double gini_impurity(const std::vector<std::variant<int, double, bool, std::string>>&);

double entropy(const std::vector<std::variant<int, double, bool, std::string>>&);

double variance(const std::vector<std::variant<int, double, bool, std::string>>&);

std::vector<bool> getMask(const std::vector<std::variant<int, double, bool, std::string>>&, std::function<bool(const std::variant<int, double, bool, std::string>&, const std::variant<int, double, bool, std::string>&)>, const std::variant<int, double, bool, std::string>&);

// Fonction NumCond
auto NumCond = [](const auto& x, const auto& val) {
    if (std::holds_alternative<int>(x) && std::holds_alternative<int>(val)) {
        return std::get<int>(x) < std::get<int>(val);
    }
    else if (std::holds_alternative<double>(x) && std::holds_alternative<double>(val)) {
        return std::get<double>(x) < std::get<double>(val);
    }
    else if (std::holds_alternative<int>(x) && std::holds_alternative<int>(val)) {
        return std::get<int>(x) == std::get<int>(val);
    }
    else if (std::holds_alternative<double>(x) && std::holds_alternative<double>(val)) {
        return std::get<double>(x) == std::get<double>(val);
    }
    else {
        throw std::runtime_error("Cannot compare types");
    }
};

// Fonction CatCond
auto CatCond = [](const auto& x, const auto& val) {
    if (std::holds_alternative<std::string>(x) && std::holds_alternative<std::string>(val)) {
        return std::get<std::string>(x) == std::get<std::string>(val);
    }
    else if (std::holds_alternative<bool>(x) && std::holds_alternative<bool>(val)) {
        return std::get<bool>(x) == std::get<bool>(val);
    }
    else if (std::holds_alternative<int>(x) && std::holds_alternative<int>(val)) {
        return std::get<int>(x) == std::get<int>(val);
    }
    else if (std::holds_alternative<double>(x) && std::holds_alternative<double>(val)) {
        return std::get<double>(x) == std::get<double>(val);
    }
    else {
        throw std::runtime_error("Cannot compare types");
    }
};

double information_gain(const std::vector<std::variant<int, double, bool, std::string>>&,
                        const std::vector<bool>&,
                        std::function<double(const std::vector<std::variant<int, double, bool, std::string>>&)> func = entropy);

void generate_combinations(const std::vector<int>&, int, std::vector<int>&, std::vector<std::vector<int>>&);

std::vector<std::vector<int>> categorical_options(const std::vector<int>&);

bool customComparator(double, double);

std::tuple<double, std::variant<int, double, bool, std::string>, bool, bool> max_information_gain_split(
        const std::vector<std::variant<int, double, bool, std::string>>&,
        const std::vector<std::variant<int, double, bool, std::string>>&,
        std::function<double(const std::vector<std::variant<int, double, bool, std::string>>&)> func = entropy);

std::tuple<std::string, std::variant<int, double, bool, std::string>, double, bool>
get_best_split(const std::string&,
               const std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>&);

std::pair<std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>,
std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>>
make_split(const std::string&,
           const std::variant<int, double, bool, std::string>&,
           const std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>&,
           bool);

std::variant<int, double, bool, std::string> makePrediction(const std::vector<std::variant<int, double, bool, std::string>>&, bool targetFactor = 1);


std::pair<bool, bool> checkConditionsDTree(
        int,
        const std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>&,
        int,
        int,
        int,
        int);

struct TreeNode {
        std::variant<int, double, bool, std::string> col;
        std::string var;
        double cutoff;
        std::variant<int, double, bool, std::string> condition;
        int depth;
        std::variant<int, double, bool, std::string> val;
        TreeNode* left;
        TreeNode* right;

};

struct NodeData {
    std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> data;
    int depth;
    TreeNode* node;

};


void init_tree(
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>> xy_current,
        std::vector<NodeData>&,
        TreeNode*);

void sub_tree(
        TreeNode*,
        std::string,
        double,
        std::string,
        int,
        std::variant<int, double, bool, std::string>,
        std::vector<NodeData>&,
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>,
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>);

void leaf_tree(
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>,
        std::string,
        bool,
        std::string,
        double,
        std::variant<int, double, bool, std::string>,
        TreeNode*);

TreeNode* interativeTrainTree(
        std::unordered_map<std::string,std::vector<std::variant<int, double, bool, std::string>>>,
        std::string,
        bool,
        int max_depth = -1,
        int min_samples_split = 2,
        double min_information_gain = 1e-20,
        int counter = 0,
        int max_categories = 20);


std::variant<int, double, bool, std::string> predict(
        const std::unordered_map<std::string, std::variant<int, double, bool, std::string>>&,
        TreeNode*);

std::vector<std::string> split_string(const std::string&, const std::string&);

std::vector<std::variant<int, double, bool, std::string>> predictions(
        std::unordered_map<std::string, std::vector<std::variant<int, double, bool, std::string>>>, TreeNode*);

void destroyTree(TreeNode* node);

#endif //DTREE_DECISIONTREE_H
