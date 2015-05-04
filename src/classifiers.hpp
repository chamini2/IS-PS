#ifndef __CLASSIFIERS_HPP__
#define __CLASSIFIERS_HPP__

#include <unordered_multiset>
using std::unordered_multiset; 

// One Nearest Neighbors classifier function
// Arguments: p of class Point which is the point representation of the problem
// Return value: it returns a type Class which is the classifies Point
template <typename Class, class Point>
Class OneNN(Point point, const unordered_multiset<Point>& data) {
    Point min = unordered_multiset[0];
    float min_distance = Point::distance(point, min);

    for (Point elem : data) {
        float elem_distance = Point::distance(point, elem);
        if (elem_distance < min_distance) {
            min = elem;
            min_distance = elem_distance;
        }
    }

    return min.ClassLabel();
}

// TODO
template <typename Class, class Point>
Class KNN(Point point, const unordered_multiset<Point>& data);

#endif
