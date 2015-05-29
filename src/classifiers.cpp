#include "classifiers.hpp"
#include <cmath>
#include <cassert>


float EuclideanDistance(const vector<float>& lhs, const vector<float>& rhs) {
    assert(lhs.size() == rhs.size()); 

    int N = lhs.size(); 
    float accum = 0; 
    for (int i = 0; i < N; ++i) {
        float diff = lhs[i] - rhs[i]; 
        accum += diff * diff; 
    }

    return sqrt(accum); 
}

float HammingDistance(const vector<float>& lhs, const vector<float>& rhs) {
    assert(lhs.size() == rhs.size()); 

    int N = lhs.size(); 
    int distance = 0; 

    for (int i = 0; i < N; ++i) {
        int diff = (int) fabs(lhs[i] - rhs[i]);

        if ( diff != 0 ) {
            ++distance; 
        }
    }

    return distance; 
}
