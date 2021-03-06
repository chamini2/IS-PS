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
using std::set;

#include <cassert>
#include "point_interface.hpp"
#include "classifiers.hpp"

extern int g_max_label;

// Generic point class.
// Template argument: Distance function to use
class GenericPoint : public PointInterface<int> {
public:
    GenericPoint(const GenericPoint& obj) {
        class_label_      = obj.class_label_;
        attributes_       = obj.attributes_;
    }

    GenericPoint() : PointInterface<int>(0, vector<double>()) {
    }

    GenericPoint(int class_label, vector<double> attributes) :
                    PointInterface<int> (class_label, attributes) {
        g_max_label = std::max(g_max_label, class_label);
    }

    ~GenericPoint() {}

    float distance(const PointInterface<int>& obj) {
        return EuclideanDistance(attributes_, obj.attributes());
    }

    static set<GenericPoint> load(const char* filename) {

        FILE *fp;
        char *line = NULL;
        size_t len = 0;
        int read_chars;

        set<GenericPoint> points;

        fp = fopen(filename, "r");
        assert(fp != NULL);

        while ((read_chars = getline(&line, &len, fp)) != -1) {
            // '^ *@' is the regular expression for comment lines
            int pos = 0;
            while (line[pos] == ' ') { ++pos; }

            // If it's a comment or an empty line
            if (line[pos] == '@' || pos + 1 == read_chars) { continue; }

            auto inst_pair = ParseCSV(line);
            points.insert(GenericPoint(inst_pair.first, inst_pair.second));
        }

        fclose(fp);
        fp = NULL;
        free(line);
        line = NULL;

        return points;
    }

private:

    static pair<int, vector<double> > ParseCSV(char* line) {
        char *next, *field;
        vector<double> attributes;

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


inline bool operator<(const GenericPoint& lhs,
                      const GenericPoint& rhs) {
    int size = lhs.attributes().size();

    for (int i = 0; i < size; ++i) {
        if (lhs.attributes()[i] != rhs.attributes()[i]) {
            return lhs.attributes()[i] < rhs.attributes()[i];
        }
    }

    return lhs.ClassLabel() < rhs.ClassLabel();
}

inline bool operator==(const GenericPoint& lhs,
                       const GenericPoint& rhs) {

    if (lhs.attributes().size() != rhs.attributes().size()) return false; 

    int size = lhs.attributes().size();

    for (int i = 0; i < size; ++i) {
        if (lhs.attributes()[i] != rhs.attributes()[i]) {
            return false;
        }
    }

    return lhs.ClassLabel() == rhs.ClassLabel();
}
#endif
