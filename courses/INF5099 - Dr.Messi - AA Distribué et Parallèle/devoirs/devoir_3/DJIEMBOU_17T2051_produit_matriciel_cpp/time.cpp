#include "headers/time.h"


using namespace std;

string getCurrentDate() {
    // Obtention de la date actuelle
    time_t now = time(nullptr);
    tm* timeinfo = localtime(&now);

    // Formatage de la date
    char buffer[11];
    strftime(buffer, sizeof(buffer), "%d_%m_%Y", timeinfo);

    return string(buffer);
}

