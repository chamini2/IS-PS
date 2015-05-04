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
    PointInterface(Class class_label, vector<float> attributes) : 
                            class_label_ ( class_label ), 
                            attributes_ ( attributes ) {
    }
    ~PointInterface() {}
    // Returns the class label of type Class
    Class ClassLabel() { return class_label_; }
    virtual float distance(PointInterface<Class>) = 0;
    vector<float> attributes() const { return attributes_; }
protected:
    Class class_label_;
    vector<float> attributes_;
};
#endif
