#include <cassert>
#include <cstdio>

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <unordered_map>
using std::unordered_map;

#include <set>
using std::multiset;

#include <iostream>
using std::cout;
using std::endl;

using std::pair;
using std::make_pair;

#include <cstring>
#include "testing.hpp"

const char *distance_functions[] = { "hamming", "euclidean" };

// TODO: Change these two functions to Test class as static functions
vector<string> ParseCSV(char* line) {
    char *field;
    vector<string> attributes;

    field = strtok(line, ",");
    while (field != NULL) {
        if (field[strlen(field) - 1] == '\n') {
            field[strlen(field) - 1] = '\0'; 
        }
        attributes.push_back(field);
        field = strtok(NULL, ",");
    }

    return attributes; 
}

vector<Test> load(const char* filename) {
        FILE *fp;
        char *line = NULL;
        size_t len = 0;

        vector<Test> result;

        fp = fopen(filename, "r");
        assert(fp != NULL);

        while (getline(&line, &len, fp) != -1) {

            Test t(ParseCSV(line)); 
            result.push_back(t); 
        }

        return result; 
}



int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage: %s <test_file>", argv[0]);
        exit(1);
    }

    vector<Test> tests = load(argv[1]); 

    for (Test t : tests) {
        cout << t << t.run() << endl; 
    }

    return 0;
}
