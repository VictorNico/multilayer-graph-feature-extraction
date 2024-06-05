#include "headers/file.h"



using namespace std;

void write_csv(string line, string path){
	ofstream file(path, ios::app);
    if (!file) {
        cout << "failed to open file" << endl;
    }

    file << line << "\n";

    file.close();

    cout << "file have been wrote with success" << endl;
}

bool fileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}