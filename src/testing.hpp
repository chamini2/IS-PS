#ifndef __TESTING_HPP__
#define __TESTING_HPP__

#include <iostream>
using std::cout;
using std::endl;
#include <sstream>
using std::ostream;

#include <vector>
using std::vector;

#include<string>
using std::string;

using std::pair;

class Result {
public:
    Result() {}
    Result(pair<int, int> n_points, pair<float, float> correct,
           pair<float, float> reducti,   pair<float, float> quality) : src_points(n_points.first),
                                                                 dst_points(n_points.second),
                                                                 src_correct(correct.first),
                                                                 dst_correct(correct.second),
                                                                 src_reducti(reducti.first),
                                                                 dst_reducti(reducti.second),
                                                                 src_quality(quality.first),
                                                                 dst_quality(quality.second) {
    }
    void addTime(double t) { time = t; }

    void addPoints(pair<int,int>);
    void addReduction(pair<float,float>);
    void addClassification(pair<float,float>);
    void addQuality(pair<float,float>);

    int src_points, dst_points;
    float src_correct, dst_correct;
    float src_reducti, dst_reducti;
    float src_quality, dst_quality;
    double time;
};

class Test {
public:
    Test(vector<string> attributes) : attributes_ (attributes) {}

    Result run(int iterations);
    vector<string> attributes_;
};


std::ostream& operator<<(std::ostream& os, const Test& obj);
std::ostream& operator<<(std::ostream& os, const Result& obj);
#endif
