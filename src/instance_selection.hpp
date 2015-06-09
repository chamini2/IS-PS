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
        result_        = NULL;
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

template <typename Point>
bool AscendingCentroidComparetor(const pair<Point, float>& lhs, const pair<Point, float>& rhs) {
        return lhs.second < rhs.second;
}

template <typename Point>
bool DescendingCentroidComparetor(const pair<Point, float>& lhs, const pair<Point, float>& rhs) {
        return lhs.second > rhs.second;
}

// Template class to handle IS-pS solution representation
// Template arguments:
// * Point: Representation of the classification problem points
// * Class: Representation of the classification problem classes
template <typename Point, typename Class>
class PopulationMap {
public:
    // Map <flag, metaheuristic function>
    typedef PopulationMap<Point, Class> (*Metaheuristic)(const PopulationMap<Point,Class>&, int); 
    typedef unordered_map<int,  Metaheuristic> MetaHeuristicMap; 
    // Function pointers typedefs
    typedef Class (*Classifier)(Point, const multiset<Point>&); 
    typedef float (*Evaluator)(float, float, float); 

    // Just for styling
    typedef int MetaheuristicType; 

    // Empty constructor
    PopulationMap() { }

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
                                                                              resolve_ (LOCAL_SEARCH),
                                                                              set_to_perturb_ ( -1 ) {
    }

    // Constructor without:
    // * weight specification : default 0.5
    PopulationMap(multiset<Point> data, int points_to_toggle, 
                  Classifier cls, Evaluator eval, MetaheuristicType mht) : points_to_toggle_ ( points_to_toggle ), 
                                                                           selected_points_ ( data ), 
                                                                           correctness_weight_ ( 0.5 ), 
                                                                           classify_ (cls), 
                                                                           evaluate_ (eval),
                                                                           resolve_ (mhm[mht]),
                                                                           set_to_perturb_ ( -1 ) {
    }

    // Constructor with all arguments
    PopulationMap(multiset<Point> data, int points_to_toggle, 
                  float correctness_weight, Classifier cls, 
                  Evaluator eval, MetaheuristicType mht) : points_to_toggle_ ( points_to_toggle ), 
                                                           selected_points_ ( data ), 
                                                           correctness_weight_ ( correctness_weight ), 
                                                           classify_ (cls), 
                                                           evaluate_ (eval),
                                                           resolve_ (mhm[mht]),
                                                           set_to_perturb_ ( -1 ) {
    }

    // Initial solution to the Map
    void InitialSolution() {
        // Greedy
        MCNN();
        // XXX: This should not be executing always, only when needed
        // but for now it's done always
        ComputeCentroidsAndTotals(); 
    }

    void reset() {
        set_to_perturb_ = -1; 
        unused_point_to_toggle_.clear(); 
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
        }
    }

    void MCNN() {

        // MeasureTime mt("MCNN");
        srand(time(NULL));

        // Start with the empty set C `selected_points_`
        unselected_points_ = selected_points_;
        selected_points_.clear();

        int classes_n = g_max_label + 1;
        vector< vector<Point> > class_values(classes_n);
        vector< bool > class_represented(classes_n);

        // Separate points by classes
        for (auto p : unselected_points_) {
            class_values[p.ClassLabel()].push_back(p);
        }

        // Pick a random representative from each class
        for (auto cv : class_values) {
            // Maybe the class didn't exist
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

            // Classify all the unselected points
            for (Point p : unselected_points_) {
                // Store the ones wrongly classified
                if (p.ClassLabel() != classify_(p, selected_points_)) {
                    points_to_move.insert(p);
                    changed = true;
                }
            }

            for (int i = 0; i < classes_n; ++i) {
                class_represented[i] = false;
            }

            // Get a representative of each class of all the wrongly classified points
            for (Point p : points_to_move) {
                if (!class_represented[p.ClassLabel()]) {
                    class_represented[p.ClassLabel()] = true;
                    selected_points_.insert(p);
                    unselected_points_.erase(p);
                }
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
            selected_points_.erase(p);
            for (Point i : all) {
                // If it classifies wrongly, insert it again
                if (i.ClassLabel() != classify_(i, selected_points_)) {
                    selected_points_.insert(p);
                    break;
                }
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
    void NeighborhoodOperator(bool intelligent) {

        //MeasureTime mt("NeighborhoodOperator");
        // This function may toggle the same point more than once
        repeat(points_to_toggle_) {

            // We choose either selected_points_ or unselected_points_ (random)
            // If any of them is empty, the we take the other one
            int set_to_use = rand() % 2;

            if (selected_points_.empty()) {
                set_to_use = 0;
                // Rebuild the list of candidates
                if (intelligent && set_to_perturb_ == 1) {
                    set_to_perturb_ = -1; 
                }
            } else if (unselected_points_.empty()) {
                set_to_use = 1;
                // Rebuild the list of candidates
                if (intelligent && set_to_perturb_ == 0) {
                    set_to_perturb_ = -1; 
                }
            }

            // Need to rebuild the list of point if we have no points left
            if (intelligent && set_to_perturb_ != -1 && unused_point_to_toggle_.empty() ) {
                // Change set (so we can toggle the other one)
                if (set_to_perturb_ == set_to_use) {
                    set_to_use = (set_to_perturb_ + 1) % 2; 
                }
                set_to_perturb_ = -1; 
            }


            // If we pick to have an intelligent perturbation, get the best point
            if (intelligent) {
                toggle(GetBestPoint(set_to_use).first, set_to_use);
                //toggle(GetBestPointBF(set_to_perturb), set_to_perturb);
            } else {
                toggle(GetRandomPoint(set_to_use), set_to_use);
            }

        }
    }


    Point GetRandomPoint(int selected_points_set) {

        const multiset<Point>& set_to_use = (selected_points_set == 1 ? selected_points_
                                                                      : unselected_points_);
        auto random_point_iterator =
            std::next(std::begin(set_to_use),
                      std::rand() % set_to_use.size());

        return *random_point_iterator;
    }

    pair<Point,float> GetBestPoint(int selected_points_set) {

        int tmp_selected_set = (set_to_perturb_ != -1 ? set_to_perturb_ : selected_points_set); 
        const multiset<Point>& set_to_use = (tmp_selected_set == 1 ? selected_points_
                                                                   : unselected_points_);


        // Only compute the best point if it wasn't compute
        // in a previous attempt
        if (set_to_perturb_ == -1) {

            // Pairs <Point, Centroid distance if removed/added>
            vector<pair<Point, float> > centroid_distances; 

            centroid_distances.reserve(set_to_use.size()); 

            for (Point p : set_to_use) {
                float distance = ComputeCentroidDistance(p,selected_points_set);
                centroid_distances.push_back(make_pair(p, distance)); 
            }

            // If we intend to take a point from selected points,
            // sort the points ASC to get the minimum first
            if (selected_points_set == 1)  {
                sort(centroid_distances.begin(), centroid_distances.end(), AscendingCentroidComparetor<Point>); 
            } else { 
                sort(centroid_distances.begin(), centroid_distances.end(), DescendingCentroidComparetor<Point>); 
            }


            unused_point_to_toggle_ = centroid_distances; 

            // Update set_to_perturb in the object
            set_to_perturb_ = selected_points_set; 
        }

        pair<Point,float> picked_point = unused_point_to_toggle_.back(); 
        unused_point_to_toggle_.pop_back(); 
        return picked_point; 
    }

    pair<Point, float> GetBestPointBF(int selected_points_set) {
        const multiset<Point>& set_to_use = (selected_points_set == 1 ? selected_points_
                                                                     : unselected_points_);

        if (set_to_perturb_ == -1) {
            vector<pair<Point, float> > evaluations; 
            for (Point p : set_to_use) {
                PopulationMap<Point, Class> copy_map(*this);
                copy_map.toggle(p, selected_points_set); 
                float score = copy_map.EvaluateQuality(); 
                evaluations.push_back(make_pair(p, score)); 
            }
            // If we intend to take a point from selected points,
            // sort the points ASC to get the minimum first
            if (selected_points_set == 1)  {
                sort(evaluations.begin(), evaluations.end(), AscendingCentroidComparetor<Point>); 
            } else { 
                sort(evaluations.begin(), evaluations.end(), DescendingCentroidComparetor<Point>); 
            }

            // Create vector with points to test
            for (auto pp : evaluations) {
                unused_point_to_toggle_.push_back(pp); 
            }

            // Update set_to_perturb in the object
            set_to_perturb_ = selected_points_set; 

        }

        pair<Point, float> picked_point = unused_point_to_toggle_.back(); 
        unused_point_to_toggle_.pop_back(); 

        return picked_point; 

    }

    // XXX: 1000 iterations get to segfault
    void GreedyRandomAlgorithm(float alpha) {

        // Empty selected points, so we can chose from the ground
        unselected_points_ = selected_points_; 
        selected_points_.clear(); 
        ComputeCentroidsAndTotals(); 

        // Get the best point and put it back in the candidates vector
        // so all costs are computed
        set_to_perturb_ = -1; 
        pair<Point, float> p = GetBestPoint(0); 
        unused_point_to_toggle_.push_back(p); 

        // Take random subvector of candidates
        srand(time(NULL)); 
        int candidates_start = rand() % (unused_point_to_toggle_.size() / 2); 
        int candidates_end   = rand() % unused_point_to_toggle_.size(); 

        // Inverted range should be switched
        if (candidates_start > candidates_end) {
            int tmp          = candidates_start;
            candidates_start = candidates_end;
            candidates_end   = tmp;
        }

        // At least 5 point in the candidates
        if (candidates_end < candidates_start + 5) {
            candidates_end = candidates_start + 5; 
        }

        // Get iterators to create new vector
        auto first = unused_point_to_toggle_.begin() + candidates_start; 
        auto last  = unused_point_to_toggle_.begin() + candidates_end; 
        unused_point_to_toggle_ = vector<pair<Point,float> >(first, last); 


        // While we still have candidates
        while (!unused_point_to_toggle_.empty()) {

            // Get min and max
            float c_min = unused_point_to_toggle_.back().second;
            float c_max = unused_point_to_toggle_.begin()->second;

            // Build RCL
            vector<Point> RCL;
            pair<Point, float> candidate = unused_point_to_toggle_.back(); 
            float min_cost = c_min + alpha * (c_max - c_min); 

            while (candidate.second <= min_cost) {
                unused_point_to_toggle_.pop_back(); 
                if (unused_point_to_toggle_.empty()) {
                    break; 
                }
                RCL.push_back(candidate.first); 
                candidate = unused_point_to_toggle_.back(); 
            }

            if (RCL.empty()) {
                break; 
            }

            // Pick random point from RCL
            auto random_point_iterator =
                std::next(std::begin(RCL),
                          std::rand() % RCL.size());

            // Insert point into solution
            toggle(*random_point_iterator, 0); 

            // Force data recalculation
            set_to_perturb_ = -1; 
            pair<Point, float> p = GetBestPoint(0); 
            unused_point_to_toggle_.push_back(p); 
        }
    }

    void ComputeCentroidsAndTotals() {

        // empty solution is not an error
        if (selected_points_.empty()) {
            return; 
        }

        int n_point_attributes = selected_points_.begin()->attributes().size(); 

        for (Point p : selected_points_) {
            Class c = p.ClassLabel(); 

            ++class_frequencies_[c]; 
            vector<double>& class_total = class_totals_[c]; 

            // If the class hasn't been seen yet, create new vector with zeros
            if (class_total.empty()) {
                class_total = vector<double>(n_point_attributes, 0.0); 
            }

            const vector<double>& point_attributes = p.attributes(); 

            for (int i = 0; i < n_point_attributes; ++i) {
                class_total[i] +=  point_attributes[i]; 
            }
        }

        for (auto elem : class_totals_) {
            RecalculateCentroid(elem.first); 
        }
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

    multiset<Point> SelectedPoints() const { return selected_points_; }
    multiset<Point> UnselectedPoints() const { return unselected_points_; }
    multiset<Point> data() const {
        multiset<Point> data(selected_points_); 
        data.insert(unselected_points_.begin(), unselected_points_.end());
        return data; 
    }

    int TotalSize() const { return selected_points_.size() + unselected_points_.size(); }
    int SelectedPointsSize() const { return selected_points_.size(); }
    int UnselectedPointsSize() const { return unselected_points_.size(); }

    int SetToPerturb() { return set_to_perturb_; }
    void SetToPerturb(int set) { set_to_perturb_ = set; }

    vector<pair<Point, float> > UnusedPointsToToggle() { return unused_point_to_toggle_; }
    void UnusedPointsToToggle(vector<pair<Point, float> > points) { unused_point_to_toggle_ = points; }




private:
    // Toggles points between selected and unselected points sets.
    void toggle(Point p, int set_to_use) {

        const multiset<Point>& non_empty_set = (selected_points_.empty() ? unselected_points_ 
                                                                         : selected_points_); 
        int n_point_attributes = non_empty_set.begin()->attributes().size(); 
        vector<double>& tmp_totals = class_totals_[p.ClassLabel()]; 


        if (tmp_totals.empty()) {
            tmp_totals = vector<double>(n_point_attributes, 0.0); 
        }


        if (set_to_perturb_ != -1) {
            set_to_use = set_to_perturb_; 
        }

        if (set_to_use == 1) {
            auto p_itr = selected_points_.find(p); 
            assert(p_itr != selected_points_.end()); 
            selected_points_.erase(p_itr);
            unselected_points_.insert(p);

            --class_frequencies_[p.ClassLabel()]; 
            for (int i = 0; i < n_point_attributes; ++i) {
                tmp_totals[i] -= p.attributes()[i]; 
            }
        } else {
            auto p_itr = unselected_points_.find(p); 
            assert(p_itr != unselected_points_.end()); 
            unselected_points_.erase(p_itr);
            selected_points_.insert(p);

            ++class_frequencies_[p.ClassLabel()]; 
            for (int i = 0; i < n_point_attributes; ++i) {
                tmp_totals[i] += p.attributes()[i]; 
            }
        }
        RecalculateCentroid(p.ClassLabel()); 
        //cout << "NEW CENTROIDS -------------------------------" << endl;  
        //for (auto elem : class_centroids_) {
            //cout << elem.first << " -> ";
            //for (auto value : elem.second) {
                //cout << value << ","; 
            //}
            //cout << endl; 
        /*}*/
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

        void RecalculateCentroid(Class c) {
        int n_point_attributes = selected_points_.begin()->attributes().size(); 
        vector <double> centroid(n_point_attributes);
        vector<double> class_total = class_totals_[c]; 
        int class_frequency = class_frequencies_[c]; 

        for (int i = 0; i < n_point_attributes; ++i) {
            centroid[i] = class_total[i] / class_frequency; 
        }

        class_centroids_[c] = centroid; 
    }

    float ComputeCentroidDistance(const Point& p, int selected_points_set) {
        const multiset<Point>& set_to_use = (selected_points_set == 1 ? selected_points_
                                                                      : unselected_points_);

        int n_point_attributes      = set_to_use.begin()->attributes().size();
        int tmp_frequency           = class_frequencies_[p.ClassLabel()] + (selected_points_set == 1 ?  -1 : 1);
        vector<double> tmp_centroid = class_totals_[p.ClassLabel()];

        // If there isnt any point of class p in the total 
        if (tmp_centroid.empty()){
            tmp_centroid = vector<double>(n_point_attributes, 0.0); 
        }

        for (int i = 0; i < n_point_attributes; ++i) {
            // Removing from selected_points
            if (selected_points_set == 1) {
                tmp_centroid[i] -= p.attributes()[i]; 
            } else {
                tmp_centroid[i] += p.attributes()[i]; 
            }

            tmp_centroid[i] /= tmp_frequency; 
        }

        vector<double>& current_centroid = class_centroids_[p.ClassLabel()];

        // Empty centroid init on origin
        if (current_centroid.empty()) {
            current_centroid = vector<double>(n_point_attributes, 0.0); 
        }

        return EuclideanDistance(tmp_centroid, class_centroids_[p.ClassLabel()]); 
    }

    
    friend class BallonPointTest;
#if TEST_PRIVATE_ATTRIBUTES == 1
    FRIEND_TEST(GenericPointTest, FitnessFunction);
    FRIEND_TEST(GenericPointTest, TogglingPoints);
    FRIEND_TEST(GenericPointTest, ComputingCentroids);
    FRIEND_TEST(GenericPointTest, ComputingCentroidDistance);
    FRIEND_TEST(GenericPointTest, GettingBestPoint);
    FRIEND_TEST(GenericPointTest, NeighborhoodOperatorWithIntelligentPerturbation); 
    FRIEND_TEST(GenericPointTest, NeighborhoodOperatorIntelligentTwice); 
#endif

    // Class private members
    int points_to_toggle_;
    multiset<Point> selected_points_;                       // Set of points in the solution
    multiset<Point> unselected_points_;                     // Remaining points
    float correctness_weight_;                              // Weight of clasification and reduction in Weight objectif function
    Classifier classify_;                                   // Function that classifies the points
    Evaluator evaluate_;                                    // Function that evaluates a solution
    Metaheuristic resolve_;                                 // Metaheuristic to be used 
    int set_to_perturb_;                                    // (==1) selected_points_, (!=1) unselected_points_ 

    float error_rate_;                                      // Error rate of the final solution 
    vector<pair<Point, float> > unused_point_to_toggle_;                  // Points to toggle sorted by how do they improve the current solution
    unordered_map<Class, int> class_frequencies_;           // Number of elements per class
    unordered_map<Class, vector<double> > class_totals_;    // Sum of vector per class
    unordered_map<Class, vector<double> > class_centroids_; // Current centroid per class
    mutable int good_classifications_[N_THREADS];           // Good classifications per thread

    static MetaHeuristicMap mhm;                           // Map string -> metaheuristic function
};


#endif
