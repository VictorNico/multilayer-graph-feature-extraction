#include "headers/file.h"


void write_csv(const char* line, const char* path) {
    FILE* file = fopen(path, "a");
    if (file != NULL) {
        fprintf(file, "%s\n", line);
        fclose(file);
    }
}

int fileExists(const char* filePath) {
    FILE* file = fopen(filePath, "r");
    if (file != NULL) {
        fclose(file);
        return 1;
    }
    return 0;
}