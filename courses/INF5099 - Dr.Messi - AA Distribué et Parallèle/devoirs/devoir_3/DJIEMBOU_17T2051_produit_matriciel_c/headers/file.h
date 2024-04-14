#ifndef FILE_H
#define FILE_H


#include <stdio.h>
#include <string.h>



void write_csv(const char* line, const char* path);

int fileExists(const char* filePath) ;

#endif