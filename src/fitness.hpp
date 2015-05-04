#ifndef __FITNESS_HPP__
#define __FITNESS_HPP__

float WeightedQuality(float classification_correctness, float reduction_percentage, float alpha);
float EulerQuality(float classification_correctness, float reduction_percentage, float alpha);
float SquaredQuality(float classification_correctness, float reduction_percentage, float alpha);

#endif
