#include "headers/time.h"


char* getCurrentDate() {
    // Obtention de la date actuelle
    time_t now = time(NULL);
    struct tm* timeinfo = localtime(&now);

    // Formatage de la date
    char* buffer = (char*)malloc(11 * sizeof(char));
    strftime(buffer, 11, "%d_%m_%Y", timeinfo);

    return buffer;
}

