#ifndef __INSTANCE_SELECTION_HPP__
#define __INSTANCE_SELECTION_HPP__

#define repeat(N) for(int i = 0; i < N; ++i)

#include <cassert>
#include <cmath>
#include <cstdlib>

#include <set>
using std::multiset; 

// Template class to handle IS-PS solution representation
// Template arguments: 
// * Point: Representation of the classification problem points
// * Class: Representation of the classification problem classes
// * classify: Function that classifies the points (Ex: KNN)
template <typename Point, 
          typename Class, 
          Class (*classify)(Point, const multiset<Point>&),
          float (*fitness)(float, float)>
class PopulationMap {
public:
    PopulationMap() { }
    PopulationMap(const PopulationMap& obj) {
        points_to_toggle_ = obj->points_to_toggle_;
        selected_points_ = obj->selected_points_;
        unselected_points_ = obj->unselected_points_;
        correctness_weight_ = obj->correctness_weight_;
    }
    // TODO: Generate (or not) initial solution
    PopulationMap(multiset<Point> data, 
                  int points_to_toggle) : points_to_toggle_ ( points_to_toggle ), 
                                          selected_points_ ( data ), 
                                          correctness_weight_ ( 0.5 ) {
    }

    PopulationMap(multiset<Point> data, 
                  int points_to_toggle, 
                  float correctness_weight) : points_to_toggle_ ( points_to_toggle ),
                                              selected_points_ ( data ), 
                                              correctness_weight_ ( correctness_weight ) {
    }

    // Function that modifies the map to generate a new neighbor solution map
    void NeighborhoodOperator(void) {

        // This function may toggle the same point more than once
        repeat(points_to_toggle_) {
            auto random_point_iterator =
                std::next(std::begin(selected_points_), 
                          std::rand() % selected_points_.size()); 

            Point random_point = *random_point_iterator; 
            toggle(random_point); 
        }
    }
    // Function that evaluates the current map's quality
    // TODO: Check return type of evaluation
    float EvaluateQuality(void) {

        float classification_correctness = RunClassifier();
        float reduction_percentage = unselected_points_.size() / 
                                    (unselected_points_.size() +
                                        selected_points_.size());
        return fitness(classification_correctness, reduction_percentage, correctness_weight_);
    }

private:
    // Toggles points between selected and unselected points sets.
    void toggle(Point p) {
        if (selected_points_.count(p) == 0) {
            unselected_points_.erase(p); 
            selected_points_.insert(p); 
        } else {
            selected_points_.erase(p); 
            unselected_points_.insert(p); 
        }
    }

    // Returns the percentage of correct classified points (from 0 to 1)
    // TODO: Consider to multiply by 100 the percentage
    float RunClassifier() {
        int correct = 0; 

        for (auto kv : selected_points_) {
            Point p = kv->first; 
            if (p.ClassLabel() == classify(p, selected_points_)) {
                ++correct; 
            }
        }

        return (float) correct / selected_points_.size(); 
    }

    int points_to_toggle_; 
    multiset<Point> selected_points_;   
    multiset<Point> unselected_points_;   
    float correctness_weight_;   
};

// Performs a local search on the current map
template <typename Point, 
          typename Class, 
          Class (*classify)(Point, const multiset<Point>&), 
          float (*fitness)(float, float)>
PopulationMap<Point,Class,classify,fitness> LocalSearchFirstFound(const PopulationMap<Point,Class,classify,fitness>& map, int iterations) {
    float map_quality = map.EvaluateQuality;

    assert(iterations > 0);

    return LocalSearchFirstFound<Point,Class,classify,fitness>(map, map_quality, iterations, iterations);
}

template <typename Point, 
          typename Class, 
          Class (*classify)(Point, const multiset<Point>&),
          float (*fitness)(float,float)>
PopulationMap<Point,Class,classify,fitness> LocalSearchFirstFound(const PopulationMap<Point,Class,classify,fitness>& map,
                                          float map_quality,
                                          int curr_iterations, int max_tierations) {
    PopulationMap<Point,Class,classify,fitness> copy_map(map);
    copy_map.NeighborhoodOperator();

    if (curr_iterations == 0) {
        return map;
    }

    float copy_quality = copy_map.EvaluateQuality();

    if (copy_quality > copy_map) {
        return LocalSearchFirstFound<Point,Class,classify,fitness>(copy_map, copy_quality, max_tierations, max_tierations);
    } else {
        return LocalSearchFirstFound<Point,Class,classify,fitness>(map, map_quality, curr_iterations - 1, max_tierations);
    }
}

// PopulationMap LocalSearchBestOfAll(const PopulationMap&);
// PopulationMap LocalSearchBestOfPartial(const PopulationMap&, int); // The argument is the percentage from 1 to 100
// PopulationMap LocalSearchTopSolutions(const PopulationMap&, int);  // The argument is the number of best solutions to keep

#endif
