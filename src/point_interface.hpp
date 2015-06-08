#ifndef __POINT_INTERFACE_HPP__
#define __POINT_INTERFACE_HPP__

#include <vector>
using std::vector;

// Interface to be implemented by real problem instances
// Template arguments:
// * Class: Type of the class of each instance
template <typename Class>
class PointInterface {
public:
    PointInterface() {}
    PointInterface(Class class_label, vector<double> attributes) :
                            class_label_ ( class_label ),
                            attributes_ ( attributes ) {
    }
    ~PointInterface() {}
    // Returns the class label of type Class
    Class ClassLabel() const { return class_label_; }
    virtual float distance(const PointInterface<Class>&)=0;
    vector<double> attributes() const { return attributes_; }
protected:
    Class class_label_;
    vector<double> attributes_;
};
#endif
