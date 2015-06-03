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

struct Result {
    public:
        Result(pair<int, int> n_points, pair<float, float> cc, 
               pair<float, float> rp,   pair<float, float> sq) : src_points(n_points.first),
                                                                 dst_points(n_points.second),
                                                                 src_cc(cc.first),
                                                                 dst_cc(cc.second),
                                                                 src_rp(rp.first),
                                                                 dst_rp(rp.second),
                                                                 src_sq(sq.first),
                                                                 dst_sq(sq.second) {
    } 
    int src_points, dst_points;
    float src_cc, dst_cc;
    float src_rp, dst_rp;
    float src_sq, dst_sq; 
};

class Test {
public:
    Test(vector<string> attributes) : attributes_ (attributes) {

        cout << "Creating test with: \n"; 

        for (auto attr : attributes) {
            cout << " - " << attr << endl; 
        }
    }

    Result run();
    vector<string> attributes_; 
};


std::ostream& operator<<(std::ostream& os, const Test& obj);
std::ostream& operator<<(std::ostream& os, const Result& obj);
#endif
