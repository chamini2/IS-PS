#ifndef __INSTANCE_SELECTION_HPP__
#define __INSTANCE_SELECTION_HPP__

#define repeat(N) for(int i = 0; i < N; ++i)

#include <cstdlib>

#include <unordered_set>
using std::unordered_set; 

#include <unordered_map>
using std::unordered_map; 

// Template class to handle IS-PS solution representation
// Template arguments: 
// * Point: Representation of the classification problem points
// * Class: Representation of the classification problem classes
// * classify: Function that classifies the points (Ex: KNN)
template <typename Point, 
          typename Class, 
          Class (*classify)(Point, const unordered_set<Point>&)>
class PopulationMap {
public:
    PopulationMap() { }
    PopulationMap(const PopulationMap& obj) {
        points_to_toggle_ = obj->points_to_toggle_;
        selected_points_ = obj->selected_points_;
        unselected_points_ = obj->unselected_points_;
    }
    // TODO: Generate (or not) initial solution
    PopulationMap(unordered_set<Point> data, int points_to_toggle) : 
                                                points_to_toggle_ ( points_to_toggle ),
                                                selected_points_ ( data ) {
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
    unordered_set<Point> selected_points_;   
    unordered_set<Point> unselected_points_;   
};

// Performs a local search on the current map
template <typename P, typename C, C (*F)(P, const unordered_set<P>&)>
PopulationMap<P,C,F> LocalSearchFirstFound(const PopulationMap<P,C,F>& map, int iterations) {
    unordered_map<PopulationMap<P,C,F>,bool> visited;
    float map_quality = map.EvaluateQuality;

    return LocalSearchFirstFound<P,C,F>(map, map_quality, iterations, iterations, visited);
}
template <typename P, typename C, C (*F)(P, const unordered_set<P>&)>
PopulationMap<P,C,F> LocalSearchFirstFound(const PopulationMap<P,C,F>& map,
                                          float map_quality,
                                          int curr_iterations, int max_tierations, 
                                          unordered_map<PopulationMap<P,C,F>,bool>& visited) {
    PopulationMap<P,C,F> copy_map(map);
    copy_map.NeighborhoodOperator();

    if (curr_iterations == 0) {
        return map;
    }

    if (visited[copy_map]) {
        return LocalSearchFirstFound<P,C,F>(map, map_quality, curr_iterations, max_tierations, visited);

    } else {
        visited[copy_map] = true;

        float copy_quality = copy_map.EvaluateQuality();

        if (copy_quality > copy_map) {
            return LocalSearchFirstFound<P,C,F>(copy_map, copy_quality, max_tierations, max_tierations, visited);
        } else {
            return LocalSearchFirstFound<P,C,F>(map, map_quality, curr_iterations - 1, max_tierations, visited);
        }
        
    }
}

// PopulationMap LocalSearchBestOfAll(const PopulationMap&);
// PopulationMap LocalSearchBestOfPartial(const PopulationMap&, int); // The argument is the percentage from 1 to 100
// PopulationMap LocalSearchTopSolutions(const PopulationMap&, int);  // The argument is the number of best solutions to keep

#endif
