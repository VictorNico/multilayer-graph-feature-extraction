#include "headers/time.h"


char* getCurrentDate() {
    // Obtention de la date actuelle
    time_t now = time(NULL);
    struct tm* timeinfo = localtime(&now);

    // Formatage de la date
    char* buffer = (char*)malloc(20 * sizeof(char));
    strftime(buffer, 20, "%d_%m_%Y_%H_%M_%S", timeinfo);

    return buffer;
}

