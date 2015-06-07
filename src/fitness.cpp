#include <cmath>
#define UNUSED(expr) do { (void)(expr); } while (0) 

#define EULER_MAX 2.71828
#define EULER_MIN 1 
#define EULER_DIFF 1.71828

#include "fitness.hpp"

float WeightedQuality(float classification_correctness, float reduction_percentage, float alpha) {
    return alpha * classification_correctness +
           (1 - alpha) * reduction_percentage;
}

float EulerQuality(float classification_correctness, float reduction_percentage, float alpha) {
    UNUSED(alpha); 
    // Normalization to range {0,1}
    return (exp(classification_correctness * reduction_percentage) - EULER_MIN) / EULER_DIFF; 
}

float SquaredQuality(float classification_correctness, float reduction_percentage, float alpha) {
    UNUSED(alpha); 
    return (classification_correctness * classification_correctness) *
           (reduction_percentage       * reduction_percentage);
}
