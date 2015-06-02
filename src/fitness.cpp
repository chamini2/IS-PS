#include <cmath>

#include "fitness.hpp"

float WeightedQuality(float classification_correctness, float reduction_percentage, float alpha) {
    return alpha * classification_correctness +
           (1 - alpha) * reduction_percentage;
}

float EulerQuality(float classification_correctness, float reduction_percentage, float alpha) {
    return exp(classification_correctness * reduction_percentage * 10000);
}

float SquaredQuality(float classification_correctness, float reduction_percentage, float alpha) {
    return (classification_correctness * classification_correctness) *
           (reduction_percentage       * reduction_percentage);
}
