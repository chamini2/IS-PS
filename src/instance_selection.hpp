#ifndef __INSTANCE_SELECTION_HPP__
#define __INSTANCE_SELECTION_HPP__

#define repeat(N) for(int i = 0; i < N; ++i)
#define TEST_PRIVATE_ATTRIBUTES 0 // 1 => enable testing in private members of the classes
#define N_THREADS 100

// Metaheuristics flags
#define LOCAL_SEARCH 0
#define ITERATED_LOCAL_SEARCH 1
#define GRASP 2

// #include "classifiers.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include <iostream>
using std::make_pair; 

#include <vector>
using std::vector;

#include <set>
using std::multiset;

#include <unordered_map>
using std::unordered_map; 

#include <map>
using std::map; 

#include <iostream>
using std::cout;
using std::endl;
using std::ostream;
using std::flush;

#include <algorithm>
using std::min_element;
using std::max_element;


#include <string>
using std::string;

#include <thread>
using std::thread;

#include <mutex>
using std::mutex;

#include <array>
using std::array;

#if TEST_PRIVATE_ATTRIBUTES
#include "gtest-1.7.0/include/gtest/gtest.h"
#endif

#include <ctime>
using std::clock_t;
using std::clock;

#include "point_instances.hpp"

// Forward declaration 
#include "testing.hpp"
 
// Class to measure time. The objects will serve as "function decorators"
class MeasureTime {
public:
    MeasureTime(string fn) {
        function_name_ = fn;
        begin_         = clock();
        print_         = true; 
        ++deepness;
    }

    MeasureTime(string fn, Result *result) {
        function_name_ = fn;
        begin_         = clock();
        result_        = result; 
        print_         = true; 
        ++deepness;
    }

    MeasureTime(string fn, Result *result, bool print) {
        function_name_ = fn;
        begin_         = clock();
        result_        = result; 
        print_         = print; 
        ++deepness;
    }


    ~MeasureTime() {
        double elapsed_time = double(clock() - begin_) / CLOCKS_PER_SEC;
        if (print_) {
            repeat(deepness) { cout << "-"; }
            cout << function_name_ << " : "
                 << elapsed_time << " seconds\n" << flush;
        }

        if (result_ != NULL) {
            result_->addTime(elapsed_time); 
        } 

        --deepness;
    }

private:
    clock_t begin_;
    string function_name_;
    static int deepness;
    Result *result_; 
    bool print_; 
};



// Template class to handle IS-pS solution representation
// Template arguments:
// * Point: Representation of the classification problem points
// * Class: Representation of the classification problem classes
template <typename Point, typename Class>
class PopulationMap {
public:
    // Empty constructor
    PopulationMap() { }
    // Copy constructor
    PopulationMap(const PopulationMap& obj) {
        points_to_toggle_   = obj.points_to_toggle_;
        selected_points_    = obj.selected_points_;
        unselected_points_  = obj.unselected_points_;
        correctness_weight_ = obj.correctness_weight_;
        error_rate_         = obj.error_rate_; 
        evaluate_           = obj.evaluate_; 
        classify_           = obj.classify_; 
        resolve_            = obj.resolve_; 
    }

    // Map <flag, metaheuristic function>
    typedef PopulationMap<Point, Class> (*Metaheuristic)(const PopulationMap<Point,Class>&, int); 
    typedef unordered_map<int,  Metaheuristic> MetaHeuristicMap; 
    // Function pointers typedefs
    typedef Class (*Classifier)(Point, const multiset<Point>&); 
    typedef float (*Evaluator)(float, float, float); 

    // Just for styling
    typedef int MetaheuristicType; 

    // Constructor without:
    // * weight specification : default 0.5
    // * resolver function : default LocalSearch
    PopulationMap(multiset<Point> data, int points_to_toggle, 
                  Classifier cls, Evaluator eval) : points_to_toggle_ ( points_to_toggle ), 
                                                    selected_points_ ( data ), 
                                                    correctness_weight_ ( 0.5 ), 
                                                    classify_ (cls), 
                                                    evaluate_ (eval),
                                                    resolve_ (LOCAL_SEARCH) {
    }

    // Constructor without:
    // * resolver function : default LocalSearch
    PopulationMap(multiset<Point> data, int points_to_toggle, 
                  float correctness_weight, Classifier cls, Evaluator eval) : points_to_toggle_ ( points_to_toggle ), 
                                                                              selected_points_ ( data ), 
                                                                              correctness_weight_ ( correctness_weight ), 
                                                                              classify_ (cls), 
                                                                              evaluate_ (eval),
                                                                              resolve_ (LOCAL_SEARCH) {
    }

    // Constructor without:
    // * weight specification : default 0.5
    PopulationMap(multiset<Point> data, int points_to_toggle, 
                  Classifier cls, Evaluator eval, MetaheuristicType mht) : points_to_toggle_ ( points_to_toggle ), 
                                                    selected_points_ ( data ), 
                                                    correctness_weight_ ( 0.5 ), 
                                                    classify_ (cls), 
                                                    evaluate_ (eval),
                                                    resolve_ (mhm[mht]) {
    }

    // Constructor with all arguments
    PopulationMap(multiset<Point> data, int points_to_toggle, 
                  float correctness_weight, Classifier cls, 
                  Evaluator eval, MetaheuristicType mht) : points_to_toggle_ ( points_to_toggle ), 
                                                           selected_points_ ( data ), 
                                                           correctness_weight_ ( correctness_weight ), 
                                                           classify_ (cls), 
                                                           evaluate_ (eval),
                                                           resolve_ (mhm[mht]) {
    }

    // Initial solution to the Map
    void InitialSolution() {
        // Greedy
        CNN(); 
    }

    // Resolve method that calls the metaheuristic function
    PopulationMap<Point, Class> resolve() { return resolve_(*this, 1000); }

    void CNN() {

        // Decorator to measure time
        //MeasureTime mt("CNN");

        // Start with the empty set C `selected_points_`
        unselected_points_ = selected_points_;
        selected_points_.clear();

        srand(time(NULL));
        auto random_point_iterator =
            std::next(std::begin(unselected_points_),
                      std::rand() % unselected_points_.size());

        Point point = *random_point_iterator;


        selected_points_.insert(point);
        unselected_points_.erase(point);

        bool changed = true;
        while (changed) {
            changed = false;

            for (auto curr = unselected_points_.begin(); curr != unselected_points_.end(); ) {
                Point p = *curr;
                curr++;
                if (p.ClassLabel() != classify_(p, selected_points_)) {
                    selected_points_.insert(p);
                    unselected_points_.erase(p);
                    changed = true;
                }
            }

            // Code for batch proccessing
            // for (Point p : unselected_points_) {
            //     if (p.ClassLabel() != classify_(p, selected_points_)) {
            //         points_to_remove.insert(p);
            //         changed = true;
            //     }
            // }
            // for (Point p : points_to_remove) {
            //     selected_points_.insert(p);
            //     unselected_points_.erase(p);
            // }
        }
    }

    void MCNN() {

        //MeasureTime mt("MCNN");

        srand(time(NULL));

        // Start with the empty set C `selected_points_`
        unselected_points_ = selected_points_;
        selected_points_.clear();

        vector< vector<Point&> > classes_values(g_max_label);
        vector< bool > already(g_max_label);

        // Separate points by classes
        for (auto p : selected_points_) {
            classes_values[p.ClassLabel()].push_back(p);
        }

        // Pick a random represent from each class
        for (auto cv : classes_values) {
            if (cv.size() > 0) {
                Point p = cv[rand() % cv.size()];
                selected_points_.insert(p);
                unselected_points_.erase(p);
            }
        }

        bool changed = true;
        while (changed) {
            multiset<Point> points_to_move;
            changed = false;

            for (Point p : unselected_points_) {
                if (p.ClassLabel() != classify_(p, selected_points_)) {
                    points_to_move.insert(p);
                    changed = true;
                }
            }
            for (Point p : points_to_move) {
                if (!already[p.ClassLabel()]) {
                    already[p.ClassLabel] = true;
                    selected_points_.insert(p);
                    unselected_points_.erase(p);
                }
            }

            for (int i; i < g_max_label; ++i) {
                already[i] = false;
            }
        }
    }

    void RNN() {

        //MeasureTime mt("RNN");

        CNN();

        multiset<Point> all(selected_points_);
        all.insert(unselected_points_.begin(), unselected_points_.end());

        multiset<Point> selected_points_copy = selected_points_;

        for (Point p : selected_points_copy) {
            bool out = true;
            selected_points_.erase(p);
            for (Point i : all) {
                if (p.ClassLabel() != classify_(p, selected_points_)) {
                    selected_points_.insert(p);
                    out = false;
                    break;
                }
            }

            if (out) {
            }
        }
    }

    void RandomSolution() {

        // Decorator to measure time
        //MeasureTime mt("RandomSolution");

        srand(time(NULL));
        multiset<Point> data(selected_points_);
        data.insert(unselected_points_.begin(), unselected_points_.end());

        // Clear both sets
        selected_points_.clear();
        unselected_points_.clear();
        // First we randomize the selected_points
        for (auto itr = data.begin(); itr != data.end(); ++itr) {
            int use_in_solution = rand() % 2;

            if (use_in_solution == 1) {
                selected_points_.insert(*itr);
            } else {
                unselected_points_.insert(*itr);
            }
        }
    }


        // Function that modifies the map to generate a new neighbor solution map
    void NeighborhoodOperator(void) {

        //MeasureTime mt("NeighborhoodOperator");
        // This function may toggle the same point more than once
        repeat(points_to_toggle_) {

            // We choose either selected_points_ or unselected_points_ (random)
            // If any of them is empty, the we take the other one
            int random_to_pick_set = rand() % 2;

            if (selected_points_.empty()) {
                random_to_pick_set = 0;
            } else if (unselected_points_.empty()) {
                random_to_pick_set = 1;
            }


            Point random_point = GetRandomPoint(random_to_pick_set); 
            //Point random_point = GetBestPoint(random_to_pick_set); 
            toggle(random_point, random_to_pick_set);
        }
    }


    Point GetRandomPoint(int random_to_pick_set) {
        const multiset<Point>& set_to_use = (random_to_pick_set == 1 ? selected_points_
                                                                     : unselected_points_);
        auto random_point_iterator =
            std::next(std::begin(set_to_use),
                      std::rand() % set_to_use.size());

        return *random_point_iterator;
    }


    // Function that evaluates the current map's quality
    float EvaluateQuality(void) const {

        // Decorator to measure time
        //MeasureTime mt("EvaluateQuality");

        float classification_correctness = RunClassifier(selected_points_, unselected_points_);
        float reduction_percentage       = GetReductionPercentage();

        return evaluate_(classification_correctness, reduction_percentage, correctness_weight_);
    }


    pair<float,float> SolutionStatistics() {
        return make_pair(RunClassifier(selected_points_, unselected_points_), GetReductionPercentage()); 
    }

    multiset<Point> selected_points() const { return selected_points_; }
    multiset<Point> unselected_points() const { return unselected_points_; }

    int TotalSize() const { return selected_points_.size() + unselected_points_.size(); }
    int SelectedPointsSize() const { return selected_points_.size(); }
    int UnselectedPointsSize() const { return unselected_points_.size(); }




private:
    // Toggles points between selected and unselected points sets.
    void toggle(Point p, int set_to_use) {
        if (set_to_use == 1) {
            selected_points_.erase(p);
            unselected_points_.insert(p);
        } else {
            unselected_points_.erase(p);
            selected_points_.insert(p);
        }
    }

    // Thread function to be used in parallel
    void ClassifyPoint(int thread, const Point& p, const multiset<Point>& training_set) const {

        if (p.ClassLabel() == classify_(p, training_set)) {
            ++good_classifications_[thread];
        }
    }

    // Returns the percentage of correct classified points (from 0 to 1)
    // TODO: Consider to multiply by 100 the percentage
    float RunClassifier(const multiset<Point>& training_set,
                                    const multiset<Point>& testing_set) const {

        //MeasureTime mt("RunClassifier");
        if (testing_set.empty()) {
            return 0.0;
        }

        // XXX: PARALLELISM IS SLOWER
        //memset(good_classifications_, 0, sizeof(good_classifications_));
        //thread classifiers[N_THREADS];

        //auto fun_ptr = &PopulationMap<Point, Class, classify, fitness>::ClassifyPoint;

        //// TODO: Parallelize this for
        //auto p = testing_set.begin();
        //for (int t = 0; p != testing_set.end(); ++p, t = (t + 1) % N_THREADS) {

            //classifiers[t] = thread(fun_ptr, this, t, *p, training_set);

            //// Wait all threads N_THREAD threads are used
            //if (t == N_THREADS - 1) {
                //for (int w = 0; w < N_THREADS; ++w) {
                    //classifiers[w].join();
                //}
            //}
        //}

        //// Collect the well classified points
        //int correct = 0;
        //for (int w = 0; w < N_THREADS; ++w) {
            //if (classifiers[w].joinable()) {
                //classifiers[w].join();
            //}
            //correct += good_classifications_[w];
        /*}*/

        int correct = 0;

        // TODO: Parallelize this for
        for (const Point& p: testing_set) {
            if (p.ClassLabel() == classify_(p, training_set)) {
                ++correct;
            }

        }

        return float(correct) / testing_set.size();
    }

    float GetReductionPercentage() const {
        return float(unselected_points_.size()) / (unselected_points_.size() +
                                                        selected_points_.size());
    }


    friend class BallonPointTest;
#ifndef TEST_PRIVATE_ATTRIBUTES
    FRIEND_TEST(BallonPointTest, TogglingPoints);
    FRIEND_TEST(BallonPointTest, FitnessFunction);
#endif

    // Class private members
    // XXX: IF NEW ATTRIBUTES ARE ADDED, ADD TO COPY CONSTRUCTOR
    int points_to_toggle_;
    multiset<Point> selected_points_;
    multiset<Point> unselected_points_;
    float correctness_weight_;
    float error_rate_; 
    Classifier classify_; 
    Evaluator evaluate_; 
    Metaheuristic resolve_; 
    mutable int good_classifications_[N_THREADS];

    static MetaHeuristicMap mhm; 
};


// Performs a local search on the current map
template <typename Point, typename Class>
PopulationMap<Point,Class> LocalSearchFirstFound(const PopulationMap<Point,Class>& orig_map, int iterations) {
    // At least 1 iteration
    assert(iterations > 0);
    int curr_iterations = 0;

    PopulationMap<Point, Class> map(orig_map); 

    float curr_quality  = map.EvaluateQuality();

    while (curr_iterations < iterations) {
        PopulationMap<Point, Class> copy_map(map);

        // Get the quality of the modified map
        copy_map.NeighborhoodOperator();
        float copy_quality = copy_map.EvaluateQuality();

        // If the quality is better than the previous map, we found a new map
        if (curr_quality < copy_quality) {
            map             = copy_map;
            curr_iterations = 0;
            curr_quality    = map.EvaluateQuality();
        } else {
            ++curr_iterations;
        }
    }

    return map;
}

template <typename Point, typename Class>
    PopulationMap<Point,Class>
        LocalSearchFirstFoundRec(const PopulationMap<Point,Class>& map,
                                 float map_quality,
                                 int curr_iterations,
                                 int max_iterations) {
    PopulationMap<Point,Class> copy_map(map);

    if (curr_iterations == 0) {
        return map;
    }

    copy_map.NeighborhoodOperator();
    float copy_quality = copy_map.EvaluateQuality();

    return copy_quality > map_quality ?
    LocalSearchFirstFoundRec<Point,Class>(copy_map, copy_quality, 
                                          max_iterations, max_iterations) :
    LocalSearchFirstFoundRec<Point,Class>(map, map_quality,
                                          curr_iterations - 1, max_iterations);
}

// PopulationMap LocalSearchBestOfAll(const PopulationMap&);
// PopulationMap LocalSearchBestOfPartial(const PopulationMap&, int); // The argument is the percentage from 1 to 100
// PopulationMap LocalSearchTopSolutions(const PopulationMap&, int);  // The argument is the number of best solutions to keep

template <typename Point, typename Class>
PopulationMap<Point,Class> IteratedLocalSearch(const PopulationMap<Point,Class>& map, int iterations) {

    const int local_search_its = 20;

    PopulationMap<Point,Class> initial_solution(map);
    initial_solution.RandomSolution();

    PopulationMap<Point,Class> best_solution = LocalSearchFirstFound(initial_solution, local_search_its);

    float curr_quality = best_solution.EvaluateQuality();

    for (int it = 0; it < iterations; ++it) {

        PopulationMap<Point,Class>& perturbated_solution(best_solution);
        perturbated_solution.RandomSolution();

        const PopulationMap<Point,Class>& candidate_solution = LocalSearchFirstFound<Point, Class>(perturbated_solution, local_search_its);
        float candidate_quality = candidate_solution.EvaluateQuality();

        // If the quality is better than the previous map, we found a new map
        if (curr_quality < candidate_quality) {
            best_solution = candidate_solution;
            curr_quality  = candidate_quality;
        }
    }

    return best_solution;
}
#endif
