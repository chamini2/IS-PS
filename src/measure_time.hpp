#ifndef __MEASURE_TIME_HPP__
#define __MEASURE_TIME_HPP__

#define repeat(N) for(int i = 0; i < N; ++i)

#include <iostream>
using std::cout;
using std::endl;
using std::flush;

#include "testing.hpp"

// Class to measure time. The objects will serve as "function decorators"
class MeasureTime {
public:
    MeasureTime(string fn) {
        function_name_ = fn;
        begin_         = clock();
        result_        = NULL;
        print_         = true;
        ++deepness;
    }

    MeasureTime(string fn, Result *result) {
        function_name_ = fn;
        begin_         = clock();
        result_        = result;
        print_         = true;
        ++deepness;
    }

    MeasureTime(string fn, Result *result, bool print) {
        function_name_ = fn;
        begin_         = clock();
        result_        = result;
        print_         = print;
        ++deepness;
    }


    ~MeasureTime() {
        double elapsed_time = double(clock() - begin_) / CLOCKS_PER_SEC;
        if (print_) {
            repeat(deepness) { cout << "-"; }
            cout << function_name_ << " : "
                 << elapsed_time << " seconds\n" << flush;
        }

        if (result_ != NULL) {
            result_->addTime(elapsed_time);
        }

        --deepness;
    }

private:
    clock_t begin_;
    string function_name_;
    static int deepness;
    Result *result_;
    bool print_;
};

#endif
