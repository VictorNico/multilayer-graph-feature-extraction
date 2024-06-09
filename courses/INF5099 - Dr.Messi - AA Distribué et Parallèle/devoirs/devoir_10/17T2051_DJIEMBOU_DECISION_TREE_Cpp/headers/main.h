//
// Created by DJIEMBOU TIENCTHEU VICTOR NICO on 25/05/2024.
//

#ifndef DTREE_MAIN_H
#define DTREE_MAIN_H

#pragma once

#include <iostream>
#include <thread>
#include <random>
#include <mutex>
#include <fstream>
#include <string>
#include <chrono>

#include "./files.h"
#include "./metrics.h"
#include "./decisionTree.h"

void test_and_save(
    TreeNode*, 
    std::unordered_map<std::string, std::vector<float>>&,
    std::unordered_map<std::string, std::vector<float>>&,
    std::vector<float>,
    std::string,
    int,
    int
    );

bool checkExistanceOfFile(std::string, std::string);

#endif //DTREE_MAIN_H
