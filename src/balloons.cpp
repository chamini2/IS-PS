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

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Must specify data"); 
        exit(1); 
    }

    printf("Reading data from file %s ... ", argv[1]);
    fflush(stdout);
    printf("done\n");
    fflush(stdout);

    //multiset<GenericPoint<HammingDistance> > points = GenericPoint<HammingDistance>::load(argv[1]); 
    multiset<GenericPoint<EuclideanDistance> > points = GenericPoint<EuclideanDistance>::load(argv[1]); 

    //PopulationMap<GenericPoint<HammingDistance>, int, 
                  //OneNN, SquaredQuality> pop_map(points, 1); 
    PopulationMap<GenericPoint<EuclideanDistance>, int, 
                  OneNN, WeightedQuality> pop_map(points, 1); 

    pop_map.GenerateRandomSolution(); 

    cout << "Original set of points:\n";
    cout << pop_map << endl; 
    cout << "----------------------" << endl; 
    printf("Starting local search ... ");
    fflush(stdout);
    //PopulationMap<GenericPoint<HammingDistance>, 
                  //int, OneNN, SquaredQuality> best_map = 
                      //LocalSearchFirstFound<GenericPoint<HammingDistance>, 
                                            //int, OneNN, SquaredQuality>(pop_map, 20);
                                            
    PopulationMap<GenericPoint<EuclideanDistance>, 
                  int, OneNN, WeightedQuality> best_map = 
                      LocalSearchFirstFound<GenericPoint<EuclideanDistance>, 
                                            int, OneNN, WeightedQuality>(pop_map, 20);
    printf("done\n");
    fflush(stdout);

    printf("Results:\n");
    cout << best_map << endl; 

    return 0;
}
