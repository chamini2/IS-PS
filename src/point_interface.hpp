#ifndef __POINT_INTERFACE_HPP__
#define __POINT_INTERFACE_HPP__

// Interface to be implemented by real problem instances
// Template arguments:
// * Class: Type of the class of each instance
template <typename Class>
class PointInterface {
    PointInterface() {}
    ~PointInterface() {}
    // Returns the class label of type Class
    virtual Class ClassLabel()=0; 
};
#endif
