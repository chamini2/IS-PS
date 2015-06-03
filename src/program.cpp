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

#include "point_instances.hpp"
#include "instance_selection.hpp"
#include "classifiers.hpp"
#include "fitness.hpp"

const char *distance_functions[] = { "hamming", "euclidean" };

int main(int argc, char *argv[]) {

    if (argc < 3) {
        printf("Usage: %s <data_file> <distance_function>", argv[0]);
        exit(1);
    }

    MeasureTime mt("principal");

    if (strcmp(argv[2], "hamming") == 0) {
        multiset<GenericPoint<HammingDistance> > points = GenericPoint<HammingDistance>::load(argv[1]);
        PopulationMap<GenericPoint<HammingDistance>, int,
                      OneNN, SquaredQuality> pop_map(points, 1);

        pop_map.InitialSolution();
        cout << "Original number of points: " << pop_map.SelectedPointsSize() << endl << flush;
        PopulationMap<GenericPoint<HammingDistance>,
                      int, OneNN, SquaredQuality> best_map =
                          LocalSearchFirstFound<GenericPoint<HammingDistance>,
                                                int, OneNN, SquaredQuality>(pop_map, 20);

        cout << "Result number of points: " << best_map.SelectedPointsSize() << endl << flush;
    } else if (strcmp(argv[2], "euclidean") == 0){

        multiset<GenericPoint<EuclideanDistance> > points = GenericPoint<EuclideanDistance>::load(argv[1]);
        PopulationMap<GenericPoint<EuclideanDistance>, int,
                      OneNN, WeightedQuality> pop_map(points, 1);

        pop_map.InitialSolution();
        cout << "Original number of points: " << pop_map.SelectedPointsSize() << endl << flush;
        PopulationMap<GenericPoint<EuclideanDistance>,
                      int, OneNN, WeightedQuality> best_map = IteratedLocalSearch<GenericPoint<EuclideanDistance>,
                                                                                  int, OneNN, WeightedQuality>(pop_map, 20);

        //pop_map.RandomSolution();
        //cout << "Original number of points: " << pop_map.SelectedPointsSize() << endl << flush;
        //PopulationMap<GenericPoint<EuclideanDistance>,
                      //int, OneNN, WeightedQuality> best_map =
                          //LocalSearchFirstFound<GenericPoint<EuclideanDistance>,
                                                //int, OneNN, WeightedQuality>(pop_map, 20);

        cout << "Result number of points: " << best_map.SelectedPointsSize() << endl << flush;
    } else {
        printf("Invalid distance function\n");
        printf("Available distances: \n");
        for (auto dist : distance_functions) {
            printf(" - %s\n", dist);
        }
    }

    return 0;
}
