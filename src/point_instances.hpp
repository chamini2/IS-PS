// Class to define specific problem point classes
// These classes need to extend point_interface.
#ifndef __POINT_INSTANCES_HPP__
#define __POINT_INSTANCES_HPP__

#include <sstream>
using std::ostream; 
#include "point_interface.hpp"
#include "classifiers.hpp"

// Generic point class.
// Template argument: Distance function to use
template <float (*distance_fun)(const vector<float>&, const vector<float>&)>
class GenericPoint : public PointInterface<int> {
public:
    GenericPoint(int class_label, vector<float> attributes) : 
                    PointInterface<int> (class_label, attributes) {
    }

    float distance(const PointInterface<int>& obj) {
        return distance_fun(attributes_, obj.attributes()); 
    }
};

// Operator << for HammingDistance and EuclideanDistance
std::ostream& operator<<(std::ostream& os, const GenericPoint<HammingDistance>& obj) {
    for (float f : obj.attributes()) {
        os << f << ", ";
    }
    return os; 
}

std::ostream& operator<<(std::ostream& os, const GenericPoint<EuclideanDistance>& obj) {
    for (float f : obj.attributes()) {
        os << f << ", ";
    }
    return os; 
}

// Operator << for HammingDistance and EuclideanDistance
inline bool operator<(const GenericPoint<HammingDistance>& lhs, 
                      const GenericPoint<HammingDistance>& rhs) {
    int size = lhs.attributes().size();

    for (int i = 0; i < size; ++i) {
        if (lhs.attributes()[i] != rhs.attributes()[i]) {
            return lhs.attributes()[i] < rhs.attributes()[i];
        }
    }

    return false;
}

inline bool operator<(const GenericPoint<EuclideanDistance>& lhs, 
                      const GenericPoint<EuclideanDistance>& rhs) {
    int size = lhs.attributes().size();

    for (int i = 0; i < size; ++i) {
        if (lhs.attributes()[i] != rhs.attributes()[i]) {
            return lhs.attributes()[i] < rhs.attributes()[i];
        }
    }

    return false;
}

// ------------------------------------------------------------
// ------------------------------------------------------------
// ------------------------------------------------------------

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
        return HammingDistance(attributes_, obj.attributes()); 
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

// Point instance class for Abalone problem
// inside data directory of the repository
class AbalonePoint : public PointInterface<int> {
public:

    AbalonePoint(int class_label, vector<float> attributes) : 
                        PointInterface<int> ( class_label, attributes ) {
    }

    // attributes_[0] Sex       == M/F/I (0/1/2)
    // attributes_[1] Length    == float 
    // attributes_[2] Diameter  == float
    // attributes_[3] Height    == float
    // attributes_[4] Whole w   == float
    // attributes_[5] Shucked w == float
    // attributes_[6] Viscera w == float
    // attributes_[7] Shell w   == float

    // class          Rings     == int
    float distance(const PointInterface<int>& obj) {
        return EuclideanDistance(attributes_, obj.attributes()); 
    }

};

std::ostream& operator<<(std::ostream& os, const AbalonePoint& obj) {
    for (float f : obj.attributes()) {
        os << f << ", ";
    }
    return os; 
}

inline bool operator<(const AbalonePoint& lhs, const AbalonePoint& rhs) {
    int size = lhs.attributes().size();

    for (int i = 0; i < size; ++i) {
        if (lhs.attributes()[i] != rhs.attributes()[i]) {
            return lhs.attributes()[i] < rhs.attributes()[i];
        }
    }

    return false;
}

class GlassPoint : public PointInterface<int> {
public:
    GlassPoint(int class_label, vector<float> attributes) : 
                    PointInterface<int> (class_label, attributes) {
    }

    float distance(const PointInterface<int>& obj) {
        return EuclideanDistance(attributes_, obj.attributes()); 
    }
};

std::ostream& operator<<(std::ostream& os, const GlassPoint& obj) {
    for (float f : obj.attributes()) {
        os << f << ", ";
    }
    return os; 
}

inline bool operator<(const AbalonePoint& lhs, const GlassPoint& rhs) {
    int size = lhs.attributes().size();

    for (int i = 0; i < size; ++i) {
        if (lhs.attributes()[i] != rhs.attributes()[i]) {
            return lhs.attributes()[i] < rhs.attributes()[i];
        }
    }

    return false;
}

class IrisPoint : public PointInterface<int> {
public:
    IrisPoint(int class_label, vector<float> attributes) : 
                    PointInterface<int> (class_label, attributes) {
    }

    float distance(const PointInterface<int>& obj) {
        return EuclideanDistance(attributes_, obj.attributes()); 
    }
};

std::ostream& operator<<(std::ostream& os, const IrisPoint& obj) {
    for (float f : obj.attributes()) {
        os << f << ", ";
    }
    return os; 
}

inline bool operator<(const AbalonePoint& lhs, const IrisPoint& rhs) {
    int size = lhs.attributes().size();

    for (int i = 0; i < size; ++i) {
        if (lhs.attributes()[i] != rhs.attributes()[i]) {
            return lhs.attributes()[i] < rhs.attributes()[i];
        }
    }

    return false;
}


class BloggerPoint : public PointInterface<int> {
public:
    BloggerPoint(int class_label, vector<float> attributes) : 
                    PointInterface<int> (class_label, attributes) {
    }

    float distance(const PointInterface<int>& obj) {
        return HammingDistance(attributes_, obj.attributes()); 
    }
};

std::ostream& operator<<(std::ostream& os, const BloggerPoint& obj) {
    for (float f : obj.attributes()) {
        os << f << ", ";
    }
    return os; 
}
inline bool operator<(const AbalonePoint& lhs, const BloggerPoint& rhs) {
    int size = lhs.attributes().size();

    for (int i = 0; i < size; ++i) {
        if (lhs.attributes()[i] != rhs.attributes()[i]) {
            return lhs.attributes()[i] < rhs.attributes()[i];
        }
    }

    return false;
}

#endif
