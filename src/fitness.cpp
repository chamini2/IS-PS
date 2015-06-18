#include <cmath>

#define EULER_MAX 6.72253
#define EULER_MIN 0.82436
#define EULER_DIFF 5.90094

#define SQRD_MAX 5.0625
#define SQRD_MIN 0
#define SQRD_DIFF 5.0625

#include "fitness.hpp"

float WeightedQuality(float classification_correctness, float reduction_percentage, float alpha) {
    return alpha * classification_correctness + (1 - alpha) * reduction_percentage;
}

float EulerQuality(float classification_correctness, float reduction_percentage, float alpha) {
    // Normalization to range {0,1}
    return (exp((classification_correctness + (1 - alpha)) * 
                           (reduction_percentage + alpha)) - EULER_MIN) / EULER_DIFF; 
}

float SquaredQuality(float classification_correctness, float reduction_percentage, float alpha) {
    float weighted_classification = classification_correctness + (1 - alpha); 
    float weighted_reduction = reduction_percentage + alpha; 
    // Normalization to range {0,1}
    return ((weighted_classification * weighted_classification) * 
                      (weighted_reduction * weighted_reduction) - SQRD_MIN) / SQRD_DIFF;
}
