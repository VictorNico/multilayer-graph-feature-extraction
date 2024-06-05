#ifndef FILE_H
#define FILE_H


#include <iostream>
#include <fstream>
#include <string>


void write_csv(std::string line, std::string path);

bool fileExists(const std::string& filePath);

#endif