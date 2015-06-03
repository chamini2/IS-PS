#include "testing.hpp"

#include "point_instances.hpp"
#include "instance_selection.hpp"
#include "classifiers.hpp"
#include "fitness.hpp"


using std::make_pair; 

#include <unordered_map>
using std::unordered_map; 


Result Test::run() {

    MeasureTime mt("Test");

    pair<float, float> stats, rstats; 
    pair<int, int> n_points; 
    pair<float, float> cc, rp, sq; 

    if (attributes_[1] == "ham") {
        multiset<GenericPoint<HammingDistance> > points = GenericPoint<HammingDistance>::load(attributes_[0].c_str());
        PopulationMap<GenericPoint<HammingDistance>, int,
                                        OneNN, WeightedQuality> pop_map(points, 1);
        PopulationMap<GenericPoint<HammingDistance>,
            int, OneNN, WeightedQuality> best_map = IteratedLocalSearch<GenericPoint<HammingDistance>,
            int, OneNN, WeightedQuality>(pop_map, 20);
        if (attributes_[2] == "ls"){
            pop_map.InitialSolution();
            if (attributes_[3] == "eul") {
            } else if (attributes_[3] == "quad") {
            } else {
            }
        } else if (attributes_[1] == "iteratedls") {
            pop_map.InitialSolution();
        } else {
            if (attributes_[3] == "eul") {
            } else if (attributes_[3] == "quad") {
            } else {
            }
        }
    } else {
        multiset<GenericPoint<EuclideanDistance> > points = GenericPoint<EuclideanDistance>::load(attributes_[0].c_str());
        if (attributes_[2] == "ls"){
            if (attributes_[3] == "eul") {
                PopulationMap<GenericPoint<EuclideanDistance>, int,
                                                OneNN, EulerQuality> pop_map(points, 1);
                pop_map.InitialSolution();
                PopulationMap<GenericPoint<EuclideanDistance>,
                              int, OneNN, EulerQuality> best_map =
                                  LocalSearchFirstFound<GenericPoint<EuclideanDistance>,
                                                        int, OneNN, EulerQuality>(pop_map, 20);
                stats = pop_map.SolutionStatistics();

                n_points = make_pair(pop_map.SelectedPointsSize(), best_map.SelectedPointsSize());
                cc       = make_pair(stats.first, rstats.first);
                rp       = make_pair(stats.second, rstats.second);
                sq       = make_pair(pop_map.EvaluateQuality(), best_map.EvaluateQuality());
            } else if (attributes_[3] == "quad") {
                PopulationMap<GenericPoint<EuclideanDistance>, int,
                                                OneNN, SquaredQuality> pop_map(points, 1);
                pop_map.InitialSolution();
                stats = pop_map.SolutionStatistics();
                PopulationMap<GenericPoint<EuclideanDistance>,
                              int, OneNN, SquaredQuality> best_map =
                                  LocalSearchFirstFound<GenericPoint<EuclideanDistance>,
                                                        int, OneNN, SquaredQuality>(pop_map, 20);
                rstats = best_map.SolutionStatistics();

                n_points = make_pair(pop_map.SelectedPointsSize(), best_map.SelectedPointsSize());
                cc       = make_pair(stats.first, rstats.first);
                rp       = make_pair(stats.second, rstats.second);
                sq       = make_pair(pop_map.EvaluateQuality(), best_map.EvaluateQuality());

            } else {
                PopulationMap<GenericPoint<EuclideanDistance>, int,
                                                OneNN, WeightedQuality> pop_map(points, 1);
                pop_map.InitialSolution();
                stats = pop_map.SolutionStatistics();
                PopulationMap<GenericPoint<EuclideanDistance>,
                              int, OneNN, WeightedQuality> best_map =
                                  LocalSearchFirstFound<GenericPoint<EuclideanDistance>,
                                                        int, OneNN, WeightedQuality>(pop_map, 20);
                rstats = best_map.SolutionStatistics();

                n_points = make_pair(pop_map.SelectedPointsSize(), best_map.SelectedPointsSize());
                cc       = make_pair(stats.first, rstats.first);
                rp       = make_pair(stats.second, rstats.second);
                sq       = make_pair(pop_map.EvaluateQuality(), best_map.EvaluateQuality());

            }
        } else if (attributes_[2] == "iteratedls") {
            if (attributes_[3] == "eul") {
                PopulationMap<GenericPoint<EuclideanDistance>, int,
                                                OneNN, EulerQuality> pop_map(points, 1);
                pop_map.InitialSolution();
                stats = pop_map.SolutionStatistics();
                PopulationMap<GenericPoint<EuclideanDistance>,
                    int, OneNN, EulerQuality> best_map = IteratedLocalSearch<GenericPoint<EuclideanDistance>,
                    int, OneNN, EulerQuality>(pop_map, 20);
                rstats = best_map.SolutionStatistics();

                n_points = make_pair(pop_map.SelectedPointsSize(), best_map.SelectedPointsSize());
                cc       = make_pair(stats.first, rstats.first);
                rp       = make_pair(stats.second, rstats.second);
                sq       = make_pair(pop_map.EvaluateQuality(), best_map.EvaluateQuality());
            } else if (attributes_[3] == "quad") {
                PopulationMap<GenericPoint<EuclideanDistance>, int,
                                                OneNN, SquaredQuality> pop_map(points, 1);
                pop_map.InitialSolution();
                stats = pop_map.SolutionStatistics();
                PopulationMap<GenericPoint<EuclideanDistance>,
                    int, OneNN, SquaredQuality> best_map = IteratedLocalSearch<GenericPoint<EuclideanDistance>,
                    int, OneNN, SquaredQuality>(pop_map, 20);
                rstats = best_map.SolutionStatistics();

                n_points = make_pair(pop_map.SelectedPointsSize(), best_map.SelectedPointsSize());
                cc       = make_pair(stats.first, rstats.first);
                rp       = make_pair(stats.second, rstats.second);
                sq       = make_pair(pop_map.EvaluateQuality(), best_map.EvaluateQuality());
            } else {
                PopulationMap<GenericPoint<EuclideanDistance>, int,
                                                OneNN, WeightedQuality> pop_map(points, 1);
                pop_map.InitialSolution();
                stats = pop_map.SolutionStatistics();
                PopulationMap<GenericPoint<EuclideanDistance>,
                    int, OneNN, WeightedQuality> best_map = IteratedLocalSearch<GenericPoint<EuclideanDistance>,
                    int, OneNN, WeightedQuality>(pop_map, 20);
                rstats = best_map.SolutionStatistics();

                n_points = make_pair(pop_map.SelectedPointsSize(), best_map.SelectedPointsSize());
                cc       = make_pair(stats.first, rstats.first);
                rp       = make_pair(stats.second, rstats.second);
                sq       = make_pair(pop_map.EvaluateQuality(), best_map.EvaluateQuality());
            }
        }
    }

    return Result(n_points, cc, rp, sq); 
}

std::ostream& operator<<(std::ostream& os, const Test& obj) {
    for (auto attr : obj.attributes_) {
        os << attr << ", "; 
    }

    return os; 
}

std::ostream& operator<<(std::ostream& os, const Result& obj) {
    os << obj.src_points << ", "
       << obj.src_cc << ", "
       << obj.src_rp << ", "
       << obj.src_sq << ", "
       << obj.dst_points << ", "
       << obj.dst_cc << ", "
       << obj.dst_rp << ", "
       << obj.dst_sq; 

    return os; 
}
