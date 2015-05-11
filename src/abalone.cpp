#include <cassert>

#include <string>
using std::string;

#include <unordered_map>
using std::unordered_map;

#include <vector>
using std::vector;

#include <set>
using std::multiset;

#include <iostream>
using std::cout;
using std::endl;
using std::flush;

using std::pair;
using std::make_pair;

#include "point_instances.hpp"
#include "instance_selection.hpp"
#include "classifiers.hpp"
#include "fitness.hpp"

pair<bool, vector<float> > ParseCSV(unordered_map<string, float>& rule,
                                    char* line);

multiset<AbalonePoint> ParseFile(unordered_map<string, float>& rule, 
                                 const char* file);

int main(int argc, char *argv[]) {
    multiset<AbalonePoint> points;
    unordered_map<string, float> rule = { {"M", 0}, {"F", 1}, {"I", 2} };

    points = ParseFile(rule, argv[1]);
    PopulationMap<AbalonePoint, int, OneNN, SquaredQuality> pop_map(points, 1);

    cout << pop_map.TotalSize() << " = ";
    pop_map.GenerateRandomSolution(); 
    cout << pop_map.SelectedPointsSize() << " + " << pop_map.UnselectedPointsSize() << endl;

    cout << "Starting local search ... " << flush;

    PopulationMap<AbalonePoint,int,OneNN,SquaredQuality> best_map = 
        LocalSearchFirstFound<AbalonePoint,int,OneNN,SquaredQuality>(pop_map, 10);

    cout << "done" << endl << "Results:" << endl;
    cout << best_map.TotalSize() << " = ";
    cout << best_map.SelectedPointsSize() << " + " << best_map.UnselectedPointsSize() << endl;
    cout << best_map;

    return 0;
}

pair<bool, vector<float> > ParseCSV(unordered_map<string, float>& rule,
                                    char* line) {
    char *next, *field;
    vector<float> attributes;

    field = strtok(line, ",");
    next = strtok(NULL, ",");
    while (next != NULL) {

        if (isalpha(field[0])) {
            attributes.push_back(rule[field]);
        } else {
            attributes.push_back(atof(field));
        }

        field = next;
        next = strtok(NULL, ",");
    }

    field[strlen(field)-1] = '\0';
    return make_pair(atoi(field), attributes);
}

multiset<AbalonePoint> ParseFile(unordered_map<string, float>& rule, 
                                 const char* file) {
    FILE *fp;
    char *line = NULL;
    size_t len = 0;

    multiset<AbalonePoint> points;

    fp = fopen(file, "r");
    assert(fp != NULL);

    while (getline(&line, &len, fp) != -1) {

        auto inst_pair = ParseCSV(rule, line);
        points.insert(AbalonePoint(inst_pair.first, inst_pair.second));
    }

    return points;
}
