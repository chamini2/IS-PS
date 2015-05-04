#ifndef __CLASSIFIERS_HPP__
#define __CLASSIFIERS_HPP__

#include <set>
using std::multiset; 

// One Nearest Neighbors classifier function
// Arguments: p of class Point which is the point representation of the problem
// Return value: it returns a type Class which is the classifies Point
template <typename Class, class Point>
Class OneNN(Point point, const multiset<Point>& data) {
    Point min = *data.begin();
    float min_distance = point.distance(min);

    for (Point elem : data) {
        float elem_distance = point.distance(elem);
        if (elem_distance < min_distance) {
            min = elem;
            min_distance = elem_distance;
        }
    }

    return min.ClassLabel();
}

// TODO
template <typename Class, class Point>
Class KNN(Point point, const multiset<Point>& data);

#endif
