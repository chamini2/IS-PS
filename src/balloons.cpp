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

#include <gflags/gflags.h>

#include "point_instances.hpp"
#include "instance_selection.hpp"
#include "classifiers.hpp"
#include "fitness.hpp"

// TODO: Remove gflags and use getopt

// Flags to use
DEFINE_int32(points_to_toggle, 2, "Puntos a incluir/excluir del conjunto solución");
DEFINE_int32(local_iterations, 10, "Número de iteraciones antes de parar la búsqueda local");
DEFINE_string(data_file, "./data.dat", "Archivo de datos para pruebas");


pair<bool, vector<float> > ParseCSV(unordered_map<string, float>& rule,
                                    unordered_map<string, bool>& class_rule, 
                                    char* line);
multiset< BalloonPoint > ParseFile(unordered_map<string, float>& rule, 
                                            unordered_map<string, bool>& class_rule, 
                                            const char* file);

int main(int argc, char *argv[]) {

    google::ParseCommandLineFlags(&argc, &argv, true);

    multiset<BalloonPoint> points;
    unordered_map<string, float> rule = {
                                        {"YELLOW",0.0}, {"PURPLE",1.0}, 
                                        {"SMALL",0.0},  {"LARGE",1.0}, 
                                        {"DIP",0.0},    {"STRETCH",1.0}, 
                                        {"ADULT",0.0},  {"CHILD",1.0}
                                    };
    unordered_map<string, bool> class_rule = {{"T",true}, {"F",false}};

    printf("Reading data from file %s ... ", FLAGS_data_file.c_str());
    fflush(stdout);
    points = ParseFile(rule, class_rule, FLAGS_data_file.c_str());
    printf("done\n");
    fflush(stdout);

    PopulationMap<BalloonPoint, bool, 
                  OneNN, EulerQuality> pop_map(points, FLAGS_points_to_toggle); 


    printf("Starting local search ... ");
    fflush(stdout);
    PopulationMap<BalloonPoint, bool, 
                  OneNN, EulerQuality> best_map = LocalSearchFirstFound<BalloonPoint, bool, OneNN, EulerQuality>(pop_map, FLAGS_local_iterations);
    printf("done\n");
    fflush(stdout);

    printf("Results:\n");
    cout << pop_map << endl; 
    cout << "----------------------" << endl; 
    cout << best_map << endl; 

    return 0;
}

multiset<BalloonPoint> ParseFile(unordered_map<string, float>& rule, 
                                            unordered_map<string, bool>& class_rule, 
                                            const char* file) {
    FILE *fp;
    char *line = NULL;
    size_t len = 0;

    multiset< BalloonPoint > points;

    fp = fopen(file, "r");
    assert(fp != NULL);

    while (getline(&line, &len, fp) != -1) {
        auto balloon_pair = ParseCSV(rule, class_rule, line);
        points.insert(BalloonPoint(balloon_pair.first, balloon_pair.second));
    }

    return points;
}

pair<bool, vector<float> > ParseCSV(unordered_map<string, float>& rule,
                                    unordered_map<string, bool>& class_rule, 
                                    char* line) {
    char *next, *field;
    vector<float> attributes;

    field = strtok(line, ",");
    next = strtok(NULL, ",");
    while (next != NULL) {

        attributes.push_back(rule[field]);
        field = next;
        next = strtok(NULL, ",");
    }

    field[strlen(field)-1] = '\0';
    return make_pair(class_rule[field], attributes);
}
