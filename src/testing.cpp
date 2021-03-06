#include "testing.hpp"

#include "measure_time.hpp"
#include "point_instances.hpp"
#include "instance_selection.hpp"
#include "fitness.hpp"
#include "classifiers.hpp"

using std::make_pair;

#include <unordered_map>
using std::unordered_map;

typedef unordered_map<string, PopulationMap<GenericPoint, int>::Evaluator> EvaluatorMap;
typedef unordered_map<string, MetaheuristicType> MetaHeuristicTypeMap;
typedef unordered_map<string, string> PathMap;

// Objectif function map
EvaluatorMap em = {
                    { "eul", &EulerQuality },
                    { "weight", &WeightedQuality },
                    { "squared", &SquaredQuality }
                  };

// Metaheuristics map
MetaHeuristicTypeMap mthm = {
                                { "ls", LOCAL_SEARCH },
                                { "ils", ITERATED_LOCAL_SEARCH },
                                { "grasp", GRASP },
                                { "gga", GGA },
                                { "sga", SGA }
                            };

int g_max_label = 0;

Result Test::run(int iterations) {

    Result r;
    MeasureTime mt("Test", &r, false);

    pair<float, float> stats, rstats;
    pair<int, int> n_points;
    pair<float, float> correct, reducti, quality;

    // attributes_[0] = Training dataset
    // attributes_[1] = Testing dataset
    // attributes_[2] = Metaheuristic to use
    // attributes_[3] = Evaluator function

    // Read points
    set<GenericPoint> points(GenericPoint::load(attributes_[0].c_str()));
    PopulationMap<GenericPoint, int> pop_map(points, 1, &OneNN<GenericPoint, int>,
                                             em[attributes_[3]], mthm[attributes_[2]]);

    pop_map.InitialSolution();

    // Read testing set
    set<GenericPoint> testing_set(GenericPoint::load(attributes_[1].c_str()));
    stats = pop_map.SolutionStatistics(testing_set);

    PopulationMap<GenericPoint, int> best_map = pop_map.resolve(iterations);

    rstats = best_map.SolutionStatistics(testing_set);

    n_points = make_pair(pop_map.SelectedPointsSize(), best_map.SelectedPointsSize());
    correct  = make_pair(stats.first, rstats.first);
    reducti  = make_pair(stats.second, rstats.second);
    quality  = make_pair(pop_map.EvaluateQuality(testing_set), best_map.EvaluateQuality(testing_set));

    r.addPoints(n_points);
    r.addClassification(correct);
    r.addReduction(reducti);
    r.addQuality(quality);

    return r;
}

std::ostream& operator<<(std::ostream& os, const Test& obj) {
    for (auto attr : obj.attributes_) {
        os << attr << ",";
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const Result& obj) {
    os << obj.src_points << ","
       << obj.src_correct << ","
       << obj.src_reducti << ","
       << obj.src_quality << ","
       << obj.dst_points << ","
       << obj.dst_correct << ","
       << obj.dst_reducti << ","
       << obj.dst_quality << ","
       << obj.time;

    return os;
}

void Result::addPoints(pair<int,int> points) {
    src_points = points.first;
    dst_points = points.second;
}

void Result::addReduction(pair<float,float> reducti) {
    src_reducti = reducti.first;
    dst_reducti = reducti.second;
}
void Result::addClassification(pair<float,float> correct) {
    src_correct = correct.first;
    dst_correct = correct.second;
}
void Result::addQuality(pair<float,float> quality) {
    src_quality = quality.first;
    dst_quality = quality.second;
}
