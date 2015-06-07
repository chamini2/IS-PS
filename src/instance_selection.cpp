#include "instance_selection.hpp"


int MeasureTime::deepness = 0;
template<>
PopulationMap<GenericPoint,int>::MetaHeuristicMap PopulationMap<GenericPoint,int>::mhm = {
    { LOCAL_SEARCH, &LocalSearchFirstFound<GenericPoint,int> },
    { ITERATED_LOCAL_SEARCH, &IteratedLocalSearch<GenericPoint,int> }
    //{ GRASP, &GRASP<GenericPoint,int> }
}; 
