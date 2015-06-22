#include "instance_selection.hpp"
#include "measure_time.hpp"

// Performs a local search on the current map
template <typename Point, typename Class>
    PopulationMap<Point,Class>
        LocalSearchFirstFound(const PopulationMap<Point,Class>& orig_map,
                              int iterations) {
    // At least 1 iteration
    assert(iterations > 0);
    int curr_iterations = 0;

    PopulationMap<Point, Class> map(orig_map);

    float curr_quality  = map.EvaluateQuality();

    while (curr_iterations < iterations) {
        PopulationMap<Point, Class> copy_map(map);

        // Get the quality of the modified map
        copy_map.NeighborhoodOperator(true);
        float copy_quality = copy_map.EvaluateQuality();

        // If the quality is better than the previous map, we found a new map
        if (curr_quality < copy_quality) {
            map             = copy_map;
            map.reset();
            curr_iterations = 0;
            curr_quality    = map.EvaluateQuality();
        } else {
            map.SetToPerturb(copy_map.SetToPerturb());
            map.UnusedPointsToToggle(copy_map.UnusedPointsToToggle());
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

    copy_map.NeighborhoodOperator(false);
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
    PopulationMap<Point,Class>
        IteratedLocalSearch(const PopulationMap<Point,Class>& map,
                            int iterations) {

    const int local_search_its = 100;

    PopulationMap<Point,Class> initial_solution(map);

    PopulationMap<Point,Class> best_solution = LocalSearchFirstFound(initial_solution, local_search_its);
    float curr_quality = best_solution.EvaluateQuality();

    repeat (iterations) {

        PopulationMap<Point,Class> perturbated_solution(best_solution);
        perturbated_solution.reset();

        // TODO: add diversification if the solution is not improving a lot (define 'a lot')
        perturbated_solution.NeighborhoodOperator(false);

        const PopulationMap<Point,Class>& candidate_solution = LocalSearchFirstFound<Point, Class>(perturbated_solution, local_search_its);
        float candidate_quality = candidate_solution.EvaluateQuality();


        // If the quality is better than the previous map, we found a new map
        if (candidate_quality > curr_quality) {
            best_solution = candidate_solution;
            curr_quality  = candidate_quality;
        }
    }

    return best_solution;
}

#include "fitness.hpp"

template<typename Point, typename Class>
    PopulationMap<Point,Class>
        GreedyRandomizedAdaptiveSearch(const PopulationMap<Point,Class>& map,
                                       int iterations) {

    // LS iterations
    const int local_search_its = 100;

    // Alpha to determine greediness
    const float alpha = 0.5;

    // Original data set
    set<Point> data(map.data());

    PopulationMap<Point,Class> best_map = map;
    double best_quality                 = best_map.EvaluateQuality();

    repeat (iterations) {
        // Build virgin map
        PopulationMap<Point, Class> curr_map =
            PopulationMap<Point, Class>::GreedyRandomAlgorithm(data, alpha, map.classifier(), map.evaluator(), map.mht());

        if (curr_map.SelectedPoints().empty()) continue;

        //assert(curr_map.data().size() == data.size());

        // Perform local search to improve result
        PopulationMap<Point,Class> candidate = LocalSearchFirstFound<Point, Class>(curr_map, local_search_its);

        if (candidate.SelectedPoints().empty()) continue;


        // Evaluate and keep if better
        double curr_quality = candidate.EvaluateQuality();
        if (curr_quality > best_quality) {
            best_quality = curr_quality;
            best_map     = candidate;
        }
    }

    return best_map;
}

template<typename Point, typename Class>
    PopulationMap<Point,Class>
        GenerationalGeneticAlgorithm(const PopulationMap<Point,Class>& map,
                                       int iterations) {

    // XXX: Need to decide the size
    const int population_size = 50;
    // We need to decide the mutation params (for now is 50% of probability
    // and 10% of the data size perturbations)
    const float mutation_percentage = 0.5;
    const int mutation_perturbations = map.data().size() * 0.1;

    set<PopulationMap<Point,Class> > population =
        PopulationMap<Point,Class>::GenerateRandomPopulation(population_size, map.data());

    PopulationMap<Point,Class> best_solution(PopulationMap<Point,Class>::GetBestSolution(population));

    repeat(iterations) {
        set<PopulationMap<Point,Class> > descendants;

        while (descendants.size() < population_size) {
            auto parents   = PopulationMap<Point,Class>::select(population);
            auto childrens = PopulationMap<Point,Class>::crossover(parents[0], parents[1]);

            childrens.first.mutate(mutation_perturbations, mutation_percentage);
            childrens.second.mutate(mutation_perturbations, mutation_percentage);

            descendants.insert(childrens.first);
            descendants.insert(childrens.second);
        }

        population = descendants;

        PopulationMap<Point,Class> new_best(PopulationMap<Point,Class>::GetBestSolution(population));
        if (new_best.EvaluateQuality() > best_solution.EvaluateQuality()) {
            best_solution = new_best;
        }
    }

    return best_solution;
}

template<typename Point, typename Class>
    PopulationMap<Point,Class>
        SteadyStateGeneticAlgorithm(const PopulationMap<Point,Class>& map,
                                       int iterations) {
    // XXX: Need to decide the size
    const int population_size = 100;
    // Need to decide the mutation params (for now is 50% of probability
    // and 10% of the data size perturbations
    const float mutation_percentage = 0.5;
    const int mutation_perturbations = map.data().size() * 0.1;

    set<PopulationMap<Point,Class> > population =
        PopulationMap<Point,Class>::GenerateRandomPopulation(population_size, map.data());

    PopulationMap<Point,Class> best_solution(PopulationMap<Point,Class>::GetBestSolution(population));

    repeat(iterations) {
        auto parents   = PopulationMap<Point,Class>::select(population);
        auto childrens = PopulationMap<Point,Class>::crossover(parents[0], parents[1]);

        childrens.first.mutate(mutation_perturbations, mutation_percentage);
        childrens.second.mutate(mutation_perturbations, mutation_percentage);
        PopulationMap<Point,Class>::replace(childrens, population);

        PopulationMap<Point,Class> new_best(PopulationMap<Point,Class>::GetBestSolution(population));
        if (new_best.EvaluateQuality() > best_solution.EvaluateQuality()) {
            best_solution = new_best;
        }

    }

    return best_solution;
}
int MeasureTime::deepness = 0;

template<>
PopulationMap<GenericPoint,int>::MetaHeuristicMap
                PopulationMap<GenericPoint,int>::mhm = {
                    { LOCAL_SEARCH, &LocalSearchFirstFound<GenericPoint,int> },
                    { ITERATED_LOCAL_SEARCH, &IteratedLocalSearch<GenericPoint,int> },
                    { GRASP, &GreedyRandomizedAdaptiveSearch<GenericPoint,int> },
                    { GGA, &GenerationalGeneticAlgorithm<GenericPoint,int> },
                    { SGA, &SteadyStateGeneticAlgorithm<GenericPoint,int> }
                };

template<>
int PopulationMap<GenericPoint, int>::n_point_attributes_ = -1;
