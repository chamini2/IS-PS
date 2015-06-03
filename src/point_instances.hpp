// Class to define specific problem point classes
// These classes need to extend point_interface.
#ifndef __POINT_INSTANCES_HPP__
#define __POINT_INSTANCES_HPP__

// #include <algorithm>
#include <cstring>

#include <iostream>
using std::pair;

#include <sstream>
using std::ostream;

#include <set>
using std::multiset;

#include <cassert>
#include "point_interface.hpp"
#include "classifiers.hpp"

int g_max_label = 0;

// Generic point class.
// Template argument: Distance function to use
template <float (*distance_fun)(const vector<float>&, const vector<float>&)>
class GenericPoint : public PointInterface<int> {
public:
    GenericPoint(const GenericPoint<distance_fun>& obj) {
        class_label_      = obj.class_label_;
        attributes_       = obj.attributes_;
    }

    GenericPoint() : PointInterface<int>(0, vector<float>()) {
    }

    GenericPoint(int class_label, vector<float> attributes) :
                    PointInterface<int> (class_label, attributes) {
        g_max_label = std::max(g_max_label, class_label);
    }

    ~GenericPoint() {}

    float distance(const PointInterface<int>& obj) {
        return distance_fun(attributes_, obj.attributes());
    }

    static multiset<GenericPoint<distance_fun> > load(const char* filename) {

        FILE *fp;
        char *line = NULL;
        size_t len = 0;

        multiset<GenericPoint<distance_fun> > points;

        fp = fopen(filename, "r");
        assert(fp != NULL);

        while (getline(&line, &len, fp) != -1) {

            auto inst_pair = ParseCSV(line);
            points.insert(GenericPoint<distance_fun>(inst_pair.first, inst_pair.second));
        }

        return points;
    }

private:

    static pair<int, vector<float> > ParseCSV(char* line) {
        char *next, *field;
        vector<float> attributes;

        field = strtok(line, ",");
        next = strtok(NULL, ",");
        while (next != NULL) {

            attributes.push_back(atof(field));

            field = next;
            next = strtok(NULL, ",");
        }

        field[strlen(field)-1] = '\0';
        int classLabel = atoi(field);
        return make_pair(classLabel, attributes);

    }
};

// Operator << for HammingDistance and EuclideanDistance
std::ostream& operator<<(std::ostream& os, const GenericPoint<HammingDistance>& obj) {

    for (float f : obj.attributes()) {
        os << f << ", ";
    }

    os << obj.ClassLabel();
    return os;
}

std::ostream& operator<<(std::ostream& os, const GenericPoint<EuclideanDistance>& obj) {
    for (float f : obj.attributes()) {
        os << f << ", ";
    }

    os << obj.ClassLabel();
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


#endif
