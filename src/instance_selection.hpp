#ifndef __INSTANCE_SELECTION_HPP__
#define __INSTANCE_SELECTION_HPP__

#define repeat(N) for(int i = 0; i < N; ++i)

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include <set>
using std::multiset; 

#include <iostream>
using std::cout; 
using std::endl; 
using std::ostream; 

// Template class to handle IS-PS solution representation
// Template arguments: 
// * Point: Representation of the classification problem points
// * Class: Representation of the classification problem classes
// * classify: Function that classifies the points (Ex: KNN)
template <typename Point, 
          typename Class, 
          Class (*classify)(Point, const multiset<Point>&),
          float (*fitness)(float, float, float)>
class PopulationMap {
public:
    PopulationMap() { }
    PopulationMap(const PopulationMap& obj) {
        points_to_toggle_   = obj.points_to_toggle_;
        selected_points_    = obj.selected_points_;
        unselected_points_  = obj.unselected_points_;
        correctness_weight_ = obj.correctness_weight_;
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

        cout << selected_points_.size() << " -> ";  
        // This function may toggle the same point more than once
        // FIXME: Only taking from selected to unselected, make a randome 
        // choice of that first
        repeat(points_to_toggle_) {
            auto random_point_iterator =
                std::next(std::begin(selected_points_), 
                          std::rand() % selected_points_.size()); 

            Point random_point = *random_point_iterator; 
            toggle(random_point); 
        }
        cout << selected_points_.size() << "\n"; 
    }
    // Function that evaluates the current map's quality
    float EvaluateQuality(void) const {

        float classification_correctness = RunClassifier();
        float reduction_percentage = unselected_points_.size() / 
                                    (unselected_points_.size() +
                                        selected_points_.size());
        return fitness(classification_correctness, reduction_percentage, correctness_weight_);
    }

    
    multiset<Point> selected_points() const { return selected_points_; }   

    int TotalSize() const { return selected_points_.size() + unselected_points_.size(); }
    int size() const { return selected_points_.size(); }

private:
    // Toggles points between selected and unselected points sets.
    void toggle(Point p) {
        if (selected_points_.find(p) == selected_points_.end()) {
            unselected_points_.erase(p); 
            selected_points_.insert(p); 
        } else {
            selected_points_.erase(p); 
            unselected_points_.insert(p); 
        }
    }

    // Returns the percentage of correct classified points (from 0 to 1)
    // TODO: Consider to multiply by 100 the percentage
    float RunClassifier() const {
        int correct = 0; 

        for (Point p: selected_points_) {
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

template <typename Point, 
          typename Class, 
          Class (*classify)(Point, const multiset<Point>&), 
          float (*fitness)(float,float,float)>
std::ostream& operator<<(std::ostream& os, const PopulationMap<Point, Class, classify, fitness>& obj) {
    os << "Number of points " << obj.size() << endl; 
    os << "Points: " << endl; 
    for (Point p : obj.selected_points()) {
        os << p << endl;
    }

    return os; 
}

// Performs a local search on the current map
template <typename Point, 
          typename Class, 
          Class (*classify)(Point, const multiset<Point>&), 
          float (*fitness)(float,float,float)>
PopulationMap<Point,Class,classify,fitness> LocalSearchFirstFound(const PopulationMap<Point,Class,classify,fitness>& map, int iterations) {
    float map_quality = map.EvaluateQuality();

    assert(iterations > 0);

    return LocalSearchFirstFound<Point,Class,classify,fitness>(map, map_quality, iterations, iterations);
}

template <typename Point, 
          typename Class, 
          Class (*classify)(Point, const multiset<Point>&),
          float (*fitness)(float,float,float)>
PopulationMap<Point,Class,classify,fitness> LocalSearchFirstFound(const PopulationMap<Point,Class,classify,fitness>& map,
                                          float map_quality,
                                          int curr_iterations, int max_iterations) {
    PopulationMap<Point,Class,classify,fitness> copy_map(map);
    cout << copy_map.TotalSize() << " -- " << copy_map.size() << " -- " << copy_map.TotalSize() - copy_map.size() << endl; 

    if (curr_iterations == 0) {
        return map;
    }

    copy_map.NeighborhoodOperator();
    float copy_quality = copy_map.EvaluateQuality();

    if (copy_quality > map_quality) {
        return LocalSearchFirstFound<Point,Class,classify,fitness>(copy_map, copy_quality, max_iterations, max_iterations);
    } else {
        return LocalSearchFirstFound<Point,Class,classify,fitness>(map, map_quality, curr_iterations - 1, max_iterations);
    }
}

// PopulationMap LocalSearchBestOfAll(const PopulationMap&);
// PopulationMap LocalSearchBestOfPartial(const PopulationMap&, int); // The argument is the percentage from 1 to 100
// PopulationMap LocalSearchTopSolutions(const PopulationMap&, int);  // The argument is the number of best solutions to keep

#endif
