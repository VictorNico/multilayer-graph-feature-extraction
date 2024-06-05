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



double gini_impurity(const std::vector<double>&);

double entropy(const std::vector<double>&);

double variance(const std::vector<double>&);

std::vector<bool> getMask(const std::vector<double>&, std::function<bool(const double&, const double&)>, const double&);

// Fonction NumCond
auto NumCond = [](const auto& x, const auto& val) {
        return x < val;
};

// Fonction CatCond
auto CatCond = [](const auto& x, const auto& val) {
        return x == val;
};

double information_gain(const std::vector<double>&,
                        const std::vector<bool>&,
                        std::function<double(const std::vector<double>&)> func = entropy);

void generate_combinations(const std::vector<int>&, int, std::vector<int>&, std::vector<std::vector<int>>&);

std::vector<std::vector<int>> categorical_options(const std::vector<int>&);

bool customComparator(double, double);

std::tuple<double, double, bool, bool> max_information_gain_split(
        const std::vector<double>&,
        const std::vector<double>&,
        std::function<double(const std::vector<double>&)> func = entropy);

std::tuple<std::string, double, double, bool>
get_best_split(const std::string&,
               const std::unordered_map<std::string, std::vector<double>>&);

std::pair<std::unordered_map<std::string, std::vector<double>>,
std::unordered_map<std::string, std::vector<double>>>
make_split(const std::string&,
           const double&,
           const std::unordered_map<std::string, std::vector<double>>&,
           bool);

double makePrediction(const std::vector<double>&, bool targetFactor = 1);


std::pair<bool, bool> checkConditionsDTree(
        int,
        const std::unordered_map<std::string, std::vector<double>>&,
        int,
        int,
        int,
        int);

struct TreeNode {
        std::string col;
        std::string var;
        double cutoff;
        std::variant<std::string, double> condition;
        int depth;
        double val;
        TreeNode* left;
        TreeNode* right;

};

struct NodeData {
    std::unordered_map<std::string, std::vector<double>> data;
    int depth;
    TreeNode* node;

};


void init_tree(
        std::unordered_map<std::string, std::vector<double>> xy_current,
        std::vector<NodeData>&,
        TreeNode*);

void sub_tree(
        TreeNode*,
        std::string,
        double,
        std::string,
        int,
        double,
        std::vector<NodeData>&,
        std::unordered_map<std::string, std::vector<double>>,
        std::unordered_map<std::string, std::vector<double>>);

void leaf_tree(
        std::unordered_map<std::string, std::vector<double>>,
        std::string,
        bool,
        std::string,
        double,
        double,
        TreeNode*);

TreeNode* interativeTrainTree(
        std::unordered_map<std::string,std::vector<double>>,
        std::string,
        bool,
        int max_depth = -1,
        int min_samples_split = 2,
        double min_information_gain = 1e-20,
        int counter = 0,
        int max_categories = 20);


double predict(
        const std::unordered_map<std::string, double>&,
        TreeNode*);

std::vector<std::string> split_string(const std::string&, const std::string&);

std::vector<double> predictions(
        std::unordered_map<std::string, std::vector<double>>, TreeNode*);

void destroyTree(TreeNode* node);

#endif //DTREE_DECISIONTREE_H
