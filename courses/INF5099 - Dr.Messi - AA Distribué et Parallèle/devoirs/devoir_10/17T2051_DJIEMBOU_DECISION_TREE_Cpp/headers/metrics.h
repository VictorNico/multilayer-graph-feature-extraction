//
// Created by DJIEMBOU TIENCTHEU VICTOR NICO on 25/05/2024.
//

#ifndef DTREE_METRICS_H
#define DTREE_METRICS_H

#pragma once

#include <vector>
#include <unordered_map>
#include <cmath>
#include <stdexcept>
#include <variant>
#include <numeric>
#include <sstream>


double precision(const std::vector<std::vector<int>>&);

double accuracy(const std::vector<std::vector<int>>&);

std::vector<std::vector<int>> compute_confusion_matrix(const std::vector<int>&,
                                                       const std::vector<int>&,
                                                       const std::vector<int>&);

std::vector<std::vector<int>> compute_confusion_matrix(
        const std::vector<float>&,
        const std::vector<float>&,
        const std::vector<float>&);

double f1_score(const std::vector<std::vector<int>>&);

double recall(const std::vector<std::vector<int>>&);

std::unordered_map<std::string, int> count_elements(const std::vector<float>&);

#endif //DTREE_METRICS_H
