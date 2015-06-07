#include "instance_selection.hpp"

int MeasureTime::deepness = 0;

template<>
PopulationMap<GenericPoint, int>::MetaHeuristicMap PopulationMap<GenericPoint, int>::mhm = {
    { LOCAL_SEARCH, &LocalSearchFirstFound },
    { ITERATED_LOCAL_SEARCH, &IteratedLocalSearch }
    //{ GRASP, &GRASP }
}; 

// Operator << for HammingDistance and EuclideanDistance
std::ostream& operator<<(std::ostream& os, const GenericPoint& obj) {

    for (float f : obj.attributes()) {
        os << f << ", ";
    }

    os << obj.ClassLabel();
    return os;
}

std::ostream& operator<<(std::ostream& os, const PopulationMap<GenericPoint, int>& obj) {
    os << "Number of points selected " << obj.SelectedPointsSize() << endl;
    os << "Points: " << endl;
    for (GenericPoint p : obj.selected_points()) {
        os << p << endl;
    }

    return os;
}
