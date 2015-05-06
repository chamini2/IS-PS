// Class to define specific problem point classes
// These classes need to extend point_interface.
#ifndef __POINT_INSTANCES_HPP__
#define __POINT_INSTANCES_HPP__

#include <sstream>
using std::ostream; 
#include "point_interface.hpp"

// Point instance class for Ballons problem
// inside data directory of the repository
class BalloonPoint : public PointInterface<bool> {
public:

    BalloonPoint(bool class_label, vector<float> attributes) : 
                        PointInterface<bool> ( class_label, attributes ) {
    }

    // attributes_[0] == YELLOW/PURPLE (0/1)
    // attributes_[1] == SMALL/LARGE   (0/1)
    // attributes_[2] == DIP/STRETCH   (0/1)
    // attributes_[3] == ADULT/CHILD   (0/1)
    float distance(const PointInterface<bool>& obj) {
        int size = attributes_.size();
        float distance = 0;

        for (int i = 0; i < size; ++i) {
            if (attributes_[i] != obj.attributes()[i]) {
                ++distance;
            }
        }

        return distance;
    }

};

std::ostream& operator<<(std::ostream& os, const BalloonPoint& obj) {
    for (float f : obj.attributes()) {
        os << f << ", ";
    }
    return os; 
}

inline bool operator<(const BalloonPoint& lhs, const BalloonPoint& rhs) {
    int size = lhs.attributes().size();

    for (int i = 0; i < size; ++i) {
        if (lhs.attributes()[i] != rhs.attributes()[i]) {
            return lhs.attributes()[i] < rhs.attributes()[i];
        }
    }

    return false;
}

#endif
