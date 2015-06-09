#include "testing.hpp"

#include "point_instances.hpp"
#include "instance_selection.hpp"
#include "fitness.hpp"
#include "classifiers.hpp"


using std::make_pair;

#include <unordered_map>
using std::unordered_map;

typedef unordered_map<string, PopulationMap<GenericPoint, int>::Evaluator> EvaluatorMap;
typedef unordered_map<string, int> MetaHeuristicTypeMap;
typedef unordered_map<string, string> PathMap;

// Objectif function map
EvaluatorMap em = {
                    { "eul", &EulerQuality },
                    { "weight", &WeightedQuality },
                    { "qualityuared", &qualityuaredQuality }
                  };

// Metaheuristics map
MetaHeuristicTypeMap mhtm = {
                                { "ls", LOCAL_SEARCH },
                                { "ils", ITERATED_LOCAL_SEARCH },
                                { "grasp", GRASP }
                            };

// Path to testing data map
PathMap paths = {
                    {"glass", "../data/small/glass/glass.data"},
                    {"iris", "../data/small/iris/iris.data"},
                    {"blogger", "../data/small/blogger/blogger.data"},
                    {"bcw", "../data/medium/breast-cancer-wisconsin/breast-cancer-wisconsin.data"},
                    {"australian", "../data/medium/australian/australian.dat"},
                    {"pima", "../data/medium/pima-indians-diabetes/pima-indians-diabetes.data"}
                };

int g_max_label = 0;

Result Test::run() {


    Result r;
    MeasureTime mt("Test", &r, false);

    pair<float, float> stats, rstats;
    pair<int, int> n_points;
    pair<float, float> correct, reducti, quality;


    // attributes_[0] = dataset to test
    // attributes_[1] = Metaheuristic to use
    // attributes_[2] = Evaluator function

    multiset<GenericPoint> points = GenericPoint::load(paths[attributes_[0]].c_str());
    PopulationMap<GenericPoint, int> pop_map(points, 1, &OneNN<GenericPoint, int>,
                                             em[attributes_[2]], mhtm[attributes_[1]]);

    pop_map.InitialSolution();
    stats = pop_map.SolutionStatistics();

    PopulationMap<GenericPoint, int> best_map = pop_map.resolve();
    rstats = best_map.SolutionStatistics();

    n_points = make_pair(pop_map.SelectedPointsSize(), best_map.SelectedPointsSize());
    correct  = make_pair(stats.first, rstats.first);
    reducti  = make_pair(stats.second, rstats.second);
    quality  = make_pair(pop_map.EvaluateQuality(), best_map.EvaluateQuality());

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
