//
// Created by DJIEMBOU TIENCTHEU VICTOR NICO on 25/05/2024.
//

#ifndef DTREE_FILES_H
#define DTREE_FILES_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <cctype>
#include <variant>
#include <algorithm>
#include <random>
#include <chrono>

#include "./decisionTree.h"

std::unordered_map<std::string, std::vector<double>> process_file(const std::string&, const std::string&, bool);

std::string ltrim(const std::string&);

std::string rtrim(const std::string&);

const std::type_info* getType(const std::string&);

double castToAppropriateType(const std::string&);

std::tuple<
std::unordered_map<std::string, std::vector<double>>,
std::unordered_map<std::string, std::vector<double>>,
std::vector<double>,
std::vector<double>
> train_test_split(
        std::unordered_map<std::string, std::vector<double>>& ,
        std::vector<double>& ,
        double,
        int);

void write_dataset(
        const std::unordered_map<std::string, std::vector<double>>& train_data,
        const std::unordered_map<std::string, std::vector<double>>& test_data,
        const std::vector<double>& y_train,
        const std::vector<double>& y_test,
        const std::string& output_dir,
        const std::string& className);

TreeNode* loadTreeNode(std::ifstream&);
TreeNode* loadTreeModel(const std::string&);

void saveTreeNode(std::ofstream&, TreeNode*);
void saveTreeModel(const std::string&, TreeNode*);

void saveMetricsToCSV(const std::string&, const std::vector<std::string>&, const std::vector<double>&, const std::string&, const std::string&);

#endif //DTREE_FILES_H
