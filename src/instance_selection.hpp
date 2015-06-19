#ifndef __INSTANCE_SELECTION_HPP__
#define __INSTANCE_SELECTION_HPP__

#define repeat(N) for(int i = 0; i < N; ++i)
#define TEST_PRIVATE_ATTRIBUTES 0 // 1 => enable testing in private members of the classes
#define N_THREADS 100

#define TOGGLE_OUT 1
#define TOGGLE_IN 0

#define CLASSIFICATION_WEIGHT 0.9


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
using std::set;

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
using std::set_difference;


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

#include <iterator>
using std::iterator;

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


#include "fitness.hpp"

// Just for styling
typedef int MetaheuristicType;

const MetaheuristicType LOCAL_SEARCH          = 0;
const MetaheuristicType ITERATED_LOCAL_SEARCH = 1;
const MetaheuristicType GRASP                 = 2;
const MetaheuristicType GGA                   = 3;
const MetaheuristicType SGA                   = 4;

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
    typedef Class (*Classifier)(Point, const set<Point>&);
    typedef float (*Evaluator)(float, float, float);


    // Empty constructor
    PopulationMap() { }


    // Constructor without:
    // * weight specification : default 0.5
    PopulationMap(set<Point> selected, set<Point> unselected, int points_to_toggle,
                  Classifier cls, Evaluator eval, MetaheuristicType mht) : points_to_toggle_ ( points_to_toggle ),
                                                                           selected_points_ ( selected ),
                                                                           unselected_points_ ( unselected ),
                                                                           correctness_weight_ ( 0.5 ),
                                                                           classify_ (cls),
                                                                           evaluate_ (eval),
                                                                           mht_ ( mht ),
                                                                           resolve_ (mhm[mht]),
                                                                           set_to_perturb_ ( -1 ) {
        n_point_attributes_ = selected.begin()->attributes().size();
    }


    // Constructor with only the data
    PopulationMap(set<Point> data) : points_to_toggle_ ( 1 ),
                                     selected_points_ ( data ),
                                     correctness_weight_ ( CLASSIFICATION_WEIGHT ),
                                     classify_ (&OneNN<Point,Class>),
                                     evaluate_ (&EulerQuality),
                                     mht_ ( LOCAL_SEARCH ),
                                     resolve_ (mhm[LOCAL_SEARCH]),
                                     set_to_perturb_ ( -1 ) {

        n_point_attributes_ = data.begin()->attributes().size();
    }
    // Constructor without:
    // * weight specification : default 0.5
    // * resolver function : default LocalSearch
    PopulationMap(set<Point> data, int points_to_toggle,
                  Classifier cls, Evaluator eval) : points_to_toggle_ ( points_to_toggle ),
                                                    selected_points_ ( data ),
                                                    correctness_weight_ ( CLASSIFICATION_WEIGHT ),
                                                    classify_ (cls),
                                                    evaluate_ (eval),
                                                    mht_ ( LOCAL_SEARCH ),
                                                    resolve_ (mhm[LOCAL_SEARCH]) {

        n_point_attributes_ = data.begin()->attributes().size();
    }

    // Constructor without:
    // * resolver function : default LocalSearch
    PopulationMap(set<Point> data, int points_to_toggle,
                  float correctness_weight, Classifier cls, Evaluator eval) : points_to_toggle_ ( points_to_toggle ),
                                                                              selected_points_ ( data ),
                                                                              correctness_weight_ ( correctness_weight ),
                                                                              classify_ (cls),
                                                                              evaluate_ (eval),
                                                                              mht_ ( LOCAL_SEARCH ),
                                                                              resolve_ (LOCAL_SEARCH),
                                                                              set_to_perturb_ ( -1 ) {
        n_point_attributes_ = data.begin()->attributes().size();
    }

    // Constructor without:
    // * weight specification : default 0.5
    PopulationMap(set<Point> data, int points_to_toggle,
                  Classifier cls, Evaluator eval, MetaheuristicType mht) : points_to_toggle_ ( points_to_toggle ),
                                                                           selected_points_ ( data ),
                                                                           correctness_weight_ ( CLASSIFICATION_WEIGHT ),
                                                                           classify_ (cls),
                                                                           evaluate_ (eval),
                                                                           mht_ ( mht ),
                                                                           resolve_ (mhm[mht]),
                                                                           set_to_perturb_ ( -1 ) {
        n_point_attributes_ = data.begin()->attributes().size();
    }

    // Constructor with all arguments
    PopulationMap(set<Point> data, int points_to_toggle,
                  float correctness_weight, Classifier cls,
                  Evaluator eval, MetaheuristicType mht) : points_to_toggle_ ( points_to_toggle ),
                                                           selected_points_ ( data ),
                                                           correctness_weight_ ( correctness_weight ),
                                                           classify_ (cls),
                                                           evaluate_ (eval),
                                                           resolve_ (mhm[mht]),
                                                           mht_ ( mht ),
                                                           set_to_perturb_ ( -1 ) {
    }

    // Initial solution to the Map
    void InitialSolution() {
        // Greedy
        MCNN();
        // Deteriorate solution so the metaheuristic can improve more
         const float insertion_percentage = 0.1;
         DeteriorateSolution(insertion_percentage);
        // XXX: This should not be executing always, only when needed
        // but for now it's done always
        ComputeCentroidsAndTotals();
    }

    // Function to deteriorate a solution randomly toggling points
    void DeteriorateSolution(float set_percentage) {

        int elements = unselected_points_.size() * set_percentage;

        srand(time(NULL));
        repeat(elements) {
            auto random_point_iterator =
                std::next(std::begin(unselected_points_),
                          std::rand() % unselected_points_.size());

            toggle(*random_point_iterator, TOGGLE_IN);
        }
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
            set<Point> points_to_move;
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

        set<Point> all(selected_points_);
        all.insert(unselected_points_.begin(), unselected_points_.end());

        set<Point> selected_points_copy = selected_points_;

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

        set<Point> data(selected_points_);
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

        const set<Point>& set_to_use = (selected_points_set == 1 ? selected_points_
                                                                      : unselected_points_);
        auto random_point_iterator =
            std::next(std::begin(set_to_use),
                      std::rand() % set_to_use.size());

        return *random_point_iterator;
    }

    static vector<pair<Point, double> > EvaluateIncrementalCost(const set<Point>& source,
                                                                unordered_map<Class, vector<double> >& centroids,
                                                                unordered_map<Class, vector<double> >& totals,
                                                                unordered_map<Class, int>& frequencies,
                                                                int remove_from_solution) {
            // Pairs <Point, Centroid distance if removed/added>
            vector<pair<Point, double> > centroid_distances;

            centroid_distances.reserve(source.size());

            for (Point p : source) {
                float distance = ComputeCentroidDistance(p,
                                                         centroids[p.ClassLabel()],
                                                         totals[p.ClassLabel()],
                                                         frequencies[p.ClassLabel()],
                                                         remove_from_solution);
                centroid_distances.push_back(make_pair(p, distance));
            }

            // If we intend to take a point from selected points,
            // sort the points DESC to get the minimum last (back() member function)
            if (remove_from_solution == 1)  {
                sort(centroid_distances.begin(), centroid_distances.end(),
                                                 DescendingCentroidComparetor<Point>);
            } else {
                sort(centroid_distances.begin(), centroid_distances.end(),
                                                 AscendingCentroidComparetor<Point>);
            }

            return centroid_distances;

    }

    pair<Point,float> GetBestPoint(int remove_from_solution) {

        int tmp_selected_set = (set_to_perturb_ != -1 ? set_to_perturb_ : remove_from_solution);
        const set<Point>& set_to_use = (tmp_selected_set == 1 ? selected_points_
                                                                   : unselected_points_);


        // Only compute the best point if it wasn't compute
        // in a previous attempt
        if (set_to_perturb_ == -1) {

            // Pairs <Point, Centroid distance if removed/added>
            unused_point_to_toggle_ = EvaluateIncrementalCost(set_to_use,
                                                              class_centroids_,
                                                              class_totals_,
                                                              class_frequencies_,
                                                              remove_from_solution);

            // Update set_to_perturb in the object
            set_to_perturb_ = remove_from_solution;
        }

        pair<Point,double> picked_point = unused_point_to_toggle_.back();
        unused_point_to_toggle_.pop_back();
        return picked_point;
    }

    pair<Point, float> GetBestPointBF(int remove_from_solution) {
        const set<Point>& set_to_use = (remove_from_solution == 1 ? selected_points_
                                                                     : unselected_points_);

        if (set_to_perturb_ == -1) {
            vector<pair<Point, float> > evaluations;
            for (Point p : set_to_use) {
                PopulationMap<Point, Class> copy_map(*this);
                copy_map.toggle(p, remove_from_solution);
                float score = copy_map.EvaluateQuality();
                evaluations.push_back(make_pair(p, score));
            }
            // If we intend to take a point from selected points,
            // sort the points ASC to get the minimum first
            if (remove_from_solution == 1)  {
                sort(evaluations.begin(), evaluations.end(), AscendingCentroidComparetor<Point>);
            } else {
                sort(evaluations.begin(), evaluations.end(), DescendingCentroidComparetor<Point>);
            }

            // Create vector with points to test
            for (auto pp : evaluations) {
                unused_point_to_toggle_.push_back(pp);
            }

            // Update set_to_perturb in the object
            set_to_perturb_ = remove_from_solution;

        }

        pair<Point, double> picked_point = unused_point_to_toggle_.back();
        unused_point_to_toggle_.pop_back();

        return picked_point;

    }

    // FIXME: INCONSISTENT DATA. It returns one more element in the data (sometimes)
    static PopulationMap<Point, Class> GreedyRandomAlgorithm(const set<Point>& data,
                                                             float alpha, Classifier cls,
                                                             Evaluator eval, MetaheuristicType mht) {

        // Need to calculate the centroids, totals and frequencies of the random set of candidates
        unordered_map<Class, vector<double> > centroids;
        unordered_map<Class, vector<double> > totals;
        unordered_map<Class, int> frequencies;

        auto sets = PickRandomSet(data);

        set<Point> candidates(sets.first);  // candidates to insert into the solution

        // Empty set of selected points to be fill with candidates
        set<Point> selected;

        // Compute centroid and totals to get the incremental cost
        ComputeCentroidsAndTotals(selected, centroids, totals, frequencies);
        vector<pair<Point, double> > inc_costs = EvaluateIncrementalCost(candidates, centroids,
                                                                         totals, frequencies, 0);

        while (!candidates.empty()) {
            // Min and max costs to get the cost threshold
            double c_max = inc_costs.back().second;
            double c_min = inc_costs.begin()->second;

            // Cost threshold. Alpha defines the greediness of the boundary
            double min_cost  = c_max - alpha * (c_max - c_min);

            // Pick only candidates under the threshold
            set<Point> RCL;
            int n_candidates = inc_costs.size();
            for (int i = n_candidates - 1; i >= 0; --i) {
                double curr_cost = inc_costs[i].second;
                // If the point is not in the RCL, then is not selected to the
                // solution
                if (curr_cost < min_cost) {
                    break;
                }

                RCL.insert(inc_costs[i].first);
            }

            // RCL shouldn't be empty
            assert(!RCL.empty());

            // Pick a random point from RCL
            auto random_point_iterator = std::next(std::begin(RCL),
                                                   std::rand() % RCL.size());

            // Insert random point into solution and remove from RCL
            selected.insert(Point(*random_point_iterator));
            RCL.erase(random_point_iterator);

            // Compute centroid and totals to get the incremental cost
            ComputeCentroidsAndTotals(selected, centroids, totals, frequencies);
            inc_costs = EvaluateIncrementalCost(candidates, centroids,
                                                totals, frequencies, 0);

            // XXX: Update candidates is ambiguous so we take the RCL as new
            // candidates
            candidates = RCL;
        }

        set<Point> unselected; // Points not selected to be considered

        // set_difference to get the remaining points outside the solution
        set_difference(data.begin(),
                       data.end(),
                       selected.begin(), selected.end(),
                       std::inserter(unselected, unselected.begin()));

        //cout << selected.size() << " + " << unselected.size() << " = " << data.size() << endl << flush;


        return PopulationMap<Point,Class>(selected, unselected, 1, cls, eval, mht);
    }


    static void ComputeCentroidsAndTotals(const set<Point>& source,
                                          unordered_map<Class, vector<double> >& centroids,
                                          unordered_map<Class, vector<double> >& totals,
                                          unordered_map<Class, int>& frequencies) {

        // Empty all maps
        centroids.clear();
        totals.clear();
        frequencies.clear();

        // empty solution is not an error
        if (source.empty()) {
            return;
        }

        for (Point p : source) {
            Class c = p.ClassLabel();

            ++frequencies[c];
            vector<double>& class_total = totals[c];

            // If the class hasn't been seen yet, create new vector with zeros
            if (class_total.empty()) {
                class_total = vector<double>(n_point_attributes_, 0.0);
            }

            const vector<double>& point_attributes = p.attributes();

            for (int i = 0; i < n_point_attributes_; ++i) {
                class_total[i] +=  point_attributes[i];
            }
        }

        for (auto elem : totals) {
            Class c = elem.first;
            vector<double> class_total = elem.second;
            vector <double> centroid(n_point_attributes_);
            int class_frequency = frequencies[c];

            for (int i = 0; i < n_point_attributes_; ++i) {
                centroid[i] = class_total[i] / class_frequency;
            }
            centroids[c] = centroid;
        }
    }

    void ComputeCentroidsAndTotals() {

        // empty solution is not an error
        if (selected_points_.empty()) {
            return;
        }


        for (Point p : selected_points_) {
            Class c = p.ClassLabel();

            ++class_frequencies_[c];
            vector<double>& class_total = class_totals_[c];

            // If the class hasn't been seen yet, create new vector with zeros
            if (class_total.empty()) {
                class_total = vector<double>(n_point_attributes_, 0.0);
            }

            const vector<double>& point_attributes = p.attributes();

            for (int i = 0; i < n_point_attributes_; ++i) {
                class_total[i] +=  point_attributes[i];
            }
        }

        for (auto elem : class_totals_) {
            RecalculateCentroid(elem.first);
        }
   }

    // Function that evaluates the current map's quality
    float EvaluateQuality(void) const {
        return EvaluateQuality(unselected_points_);
    }

    float EvaluateQuality(const set<Point>& testing_set) const {
        float classification_correctness = RunClassifier(selected_points_, testing_set);
        float reduction_percentage       = GetReductionPercentage();

        return evaluate_(classification_correctness, reduction_percentage, correctness_weight_);
    }


    // Get current solution statistics
    pair<float,float> SolutionStatistics(void) const {
        return SolutionStatistics(unselected_points_);
    }

    pair<float,float> SolutionStatistics(const set<Point>& testing_set) const {
        return make_pair(RunClassifier(selected_points_, testing_set), GetReductionPercentage());
    }

    // Genetic algorithm functions

    // Generates a random populatio of PopulatioMaps
    static set<PopulationMap<Point,Class> > GenerateRandomPopulation(int population_size,
                                                                     const set<Point>& data) {

        set<PopulationMap<Point,Class> > population;

        srand(time(NULL));
        repeat(population_size) {


            // Generate random solution
            PopulationMap<Point,Class> solution(data);
            solution.RandomSolution();

            // Insert into population
            population.insert(solution);
        }

        return population;
    }

    // Gets the PopulationMap with best quality
    static PopulationMap<Point,Class> GetBestSolution(const set<PopulationMap<Point,Class> >& population) {
        double best_score = -1;
        PopulationMap<Point,Class> best_solution;

        for (auto pm : population) {

            double curr_score = pm.EvaluateQuality();
            if (curr_score > best_score) {

                best_score    = curr_score;
                best_solution = pm;
            }
        }

        return best_solution;
    }

    // Modifies current map in order to diversify the solution
    void mutate(int perturbations, float mutation_percentage) {

        repeat(perturbations) {
            double prob = double(rand()) / RAND_MAX;
            // mutation_percentage = 0.6, then if 0 <= prob <= mutation_percentage is
            // 60% of chances since prob is between 0 and 1
            if (prob <= mutation_percentage) {
                NeighborhoodOperator(false);
            }
        }
    }

    // Selects two population maps from a population
    static vector<PopulationMap<Point,Class> > select(const set<PopulationMap<Point,Class> >& population) {

        srand(time(NULL));
        // Parent's random selection
        auto parent1_itr = std::next(std::begin(population),
                                      std::rand() % population.size());

        auto parent2_itr = std::next(std::begin(population),
                                      std::rand() % population.size());

        vector<PopulationMap<Point,Class> > parents;
        parents.reserve(2);
        parents.push_back(*parent1_itr);
        parents.push_back(*parent2_itr);

        return parents;
    }

    // Replaces two points of the population for the two childrens (according
    // to some criteria)
    static void replace(pair<PopulationMap<Point,Class>, PopulationMap<Point,Class> > childrens,
                        set<PopulationMap<Point,Class> >& population) {
    }
    // Combines two population maps to create two childrens
    static pair<PopulationMap<Point,Class>, PopulationMap<Point,Class> >
                                crossover(const PopulationMap<Point,Class>& fstp,
                                          const PopulationMap<Point,Class>& sndp) {
        int fstp_break_point, sndp_break_point;
        PopulationMap<Point,Class> fstc(fstp), sndc(sndp);
        auto fstp_it = fstc.selected_points_.begin();
        auto sndp_it = sndc.selected_points_.begin();

        // Randomply determine how many instances from each we're going to change
        srand(time(NULL));
        fstp_break_point = rand() % fstc.selected_points_.size();
        sndp_break_point = rand() % sndc.selected_points_.size();

        // Take some instances from fst and put them in snd
        repeat(fstp_break_point) {
            auto random_point_iterator =
                std::next(std::begin(fstp.selected_points_),
                          std::rand() % fstp.selected_points_.size());

            // Out of fstc
            fstc.toggle(*random_point_iterator, TOGGLE_OUT);
            if (sndc.unselected_points_.find(*random_point_iterator) ==
                                            sndc.unselected_points_.end()) {
                // In sndc, if it wasn't already there
                sndc.toggle(*random_point_iterator, TOGGLE_IN);
            }
        }

        // Take some instances from snd and put them in fst
        repeat(sndp_break_point) {
            auto random_point_iterator =
                std::next(std::begin(sndp.selected_points_),
                          std::rand() % sndp.selected_points_.size());

            // Out of sndc
            sndc.toggle(*random_point_iterator, TOGGLE_OUT);
            if (fstc.unselected_points_.find(*random_point_iterator) ==
                                            fstc.unselected_points_.end()) {
                // In fstc, if it wasn't already there
                fstc.toggle(*random_point_iterator, TOGGLE_IN);
            }
        }

        return make_pair(fstc, sndc);
    }

    set<Point> SelectedPoints() const { return selected_points_; }
    set<Point> UnselectedPoints() const { return unselected_points_; }
    set<Point> data() const {
        set<Point> data(selected_points_);
        data.insert(unselected_points_.begin(), unselected_points_.end());
        return data;
    }

    int TotalSize() const { return selected_points_.size() + unselected_points_.size(); }
    int SelectedPointsSize() const { return selected_points_.size(); }
    int UnselectedPointsSize() const { return unselected_points_.size(); }

    int SetToPerturb() const { return set_to_perturb_; }
    void SetToPerturb(int set) { set_to_perturb_ = set; }

    vector<pair<Point, double> > UnusedPointsToToggle() const { return unused_point_to_toggle_; }
    void UnusedPointsToToggle(vector<pair<Point, double> > points) { unused_point_to_toggle_ = points; }


    Classifier classifier() const { return classify_; }
    Evaluator evaluator() const { return evaluate_; }
    MetaheuristicType mht() const { return mht_; }

    int PointsToToggle() const { return points_to_toggle_; }

private:
    // Toggles points between selected and unselected points sets.
    void toggle(Point p, int remove_from_solution) {

        const set<Point>& non_empty_set = (selected_points_.empty() ? unselected_points_
                                                                         : selected_points_);
        int n_point_attributes = non_empty_set.begin()->attributes().size();
        vector<double>& tmp_totals = class_totals_[p.ClassLabel()];


        if (tmp_totals.empty()) {
            tmp_totals = vector<double>(n_point_attributes, 0.0);
        }


        if (set_to_perturb_ != -1) {
            remove_from_solution = set_to_perturb_;
        }

        if (remove_from_solution) {
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
        // cout << "NEW CENTROIDS -------------------------------" << endl;
        // for (auto elem : class_centroids_) {
        //     cout << elem.first << " -> ";
        //     for (auto value : elem.second) {
        //         cout << value << ",";
        //     }
        //     cout << endl;
        // }
    }

    // Thread function to be used in parallel
    void ClassifyPoint(int thread, const Point& p, const set<Point>& training_set) const {

        if (p.ClassLabel() == classify_(p, training_set)) {
            ++good_classifications_[thread];
        }
    }

    // Returns the percentage of correct classified points (from 0 to 1)
    // TODO: Consider to multiply by 100 the percentage
    float RunClassifier(const set<Point>& training_set,
                                    const set<Point>& testing_set) const {

        //MeasureTime mt("RunClassifier");
        if (testing_set.empty()) {
            return 0.0;
        }

        // XXX: PARALLELISM IS SLOWER
        /*memset(good_classifications_, 0, sizeof(good_classifications_));
        thread classifiers[N_THREADS];

        auto fun_ptr = &PopulationMap<Point, Class, classify, fitness>::ClassifyPoint;

        // TODO: Parallelize this for
        auto p = testing_set.begin();
        for (int t = 0; p != testing_set.end(); ++p, t = (t + 1) % N_THREADS) {

            classifiers[t] = thread(fun_ptr, this, t, *p, training_set);

            // Wait all threads N_THREAD threads are used
            if (t == N_THREADS - 1) {
                for (int w = 0; w < N_THREADS; ++w) {
                    classifiers[w].join();
                }
            }
        }

        // Collect the well classified points
        int correct = 0;
        for (int w = 0; w < N_THREADS; ++w) {
            if (classifiers[w].joinable()) {
                classifiers[w].join();
            }
            correct += good_classifications_[w];
        }*/

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
        vector <double> centroid(n_point_attributes_);
        vector<double> class_total = class_totals_[c];
        int class_frequency = class_frequencies_[c];

        for (int i = 0; i < n_point_attributes_; ++i) {
            centroid[i] = class_total[i] / class_frequency;
        }

        class_centroids_[c] = centroid;
    }

    static float ComputeCentroidDistance(const Point& p, vector<double>& centroid,
                                         const vector<double>& total,
                                         int frenquency, int remove_from_solution) {

        vector<double> tmp_centroid(total);

        int tmp_frequency = frenquency + (remove_from_solution == 1 ?  -1 : 1);

        // If there isnt any point of class p in the total
        if (tmp_centroid.empty()){
            tmp_centroid = vector<double>(n_point_attributes_, 0.0);
        }

        for (int i = 0; i < n_point_attributes_; ++i) {
            // Removing from selected_points
            if (remove_from_solution == 1) {
                tmp_centroid[i] -= p.attributes()[i];
            } else {
                tmp_centroid[i] += p.attributes()[i];
            }

            tmp_centroid[i] /= tmp_frequency;
        }

        // Empty centroid init on origin
        if (centroid.empty()) {
            centroid = vector<double>(n_point_attributes_, 0.0);
        }

        return EuclideanDistance(tmp_centroid, centroid);
    }


    // Function to pick a random set of points from the map's data
    static pair<set<Point>, set<Point> > PickRandomSet(const set<Point>& data) {
        set<Point> random_set;
        set<Point> rest;

        srand(time(NULL));

        // XXX: Maybe here we should make the % of selected points a variable
        for (auto itr = data.begin(); itr != data.end(); ++itr) {
            int pick_point = rand() % 3;

            // High probability to be in the set (to avoid empty sets)
            if (pick_point > 0) {
                random_set.insert(*itr);
            } else {
                rest.insert(*itr);
            }
        }


        // Avoid empty set
        if (random_set.empty()) {
            random_set.insert(*data.begin());
        }

        return make_pair(random_set, rest);
    }



#if TEST_PRIVATE_ATTRIBUTES == 1
    friend class GenericPointTest;
    FRIEND_TEST(GenericPointTest, FitnessFunction);
    FRIEND_TEST(GenericPointTest, TogglingPoints);
    FRIEND_TEST(GenericPointTest, ComputingCentroids);
    FRIEND_TEST(GenericPointTest, ComputingCentroidDistance);
    FRIEND_TEST(GenericPointTest, GettingBestPoint);
    FRIEND_TEST(GenericPointTest, NeighborhoodOperatorWithIntelligentPerturbation);
    FRIEND_TEST(GenericPointTest, NeighborhoodOperatorIntelligentTwice);
    FRIEND_TEST(GenericPointTest, PickRandomSet);
    FRIEND_TEST(GenericPointTest, EvaluateIncrementalCostEmpty);
#endif

    // Class private members
    int points_to_toggle_;
    set<Point> selected_points_;                       // Set of points in the solution
    set<Point> unselected_points_;                     // Remaining points
    float correctness_weight_;                              // Weight of clasification and reduction in Weight objectif function
    Classifier classify_;                                   // Function that classifies the points
    Evaluator evaluate_;                                    // Function that evaluates a solution
    Metaheuristic resolve_;                                 // Metaheuristic to be used
    int set_to_perturb_;                                    // (==1) selected_points_, (!=1) unselected_points_

    float error_rate_;                                      // Error rate of the final solution
    vector<pair<Point, double> > unused_point_to_toggle_;   // Points to toggle sorted by how do they improve the current solution
    unordered_map<Class, int> class_frequencies_;           // Number of elements per class
    unordered_map<Class, vector<double> > class_totals_;    // Sum of vector per class
    unordered_map<Class, vector<double> > class_centroids_; // Current centroid per class
    mutable int good_classifications_[N_THREADS];           // Good classifications per thread

    MetaheuristicType mht_;
    static MetaHeuristicMap mhm;                           // Map string -> metaheuristic function
    static int n_point_attributes_;
};

inline bool operator<(const PopulationMap<GenericPoint,int>& lhs,
                      const PopulationMap<GenericPoint,int>& rhs) {

    return lhs.SelectedPoints() < rhs.SelectedPoints();
}
#endif
