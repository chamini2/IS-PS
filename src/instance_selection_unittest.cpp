#include "instance_selection.hpp"
#include "point_instances.hpp"
#include "fitness.hpp"
#include "classifiers.hpp"
#include "gtest-1.7.0/include/gtest/gtest.h"

using std::pair; 

// attributes_[1] == SMALL/LARGE   (0/1)
// attributes_[2] == DIP/STRETCH   (0/1)
// attributes_[3] == ADULT/CHILD   (0/1)
class BallonPointTest : public ::testing::Test {
    protected:
        static void SetUpTestCase()
        {
            vector<pair<bool, vector<float> > > points = {
                                                { true, { 0.0,0.0,1.0,0.0 } }, 
                                                { true, { 0.0,0.0,1.0,0.0 } },
                                                { false,{ 0.0,0.0,1.0,1.0 } },
                                                { false,{ 0.0,0.0,0.0,0.0 } },
                                                { false,{ 0.0,0.0,0.0,1.0 } },
                                                { true ,{ 0.0,1.0,1.0,0.0 } },
                                                { true ,{ 0.0,1.0,1.0,0.0 } },
                                                { false,{ 0.0,1.0,1.0,1.0 } },
                                                { false,{ 0.0,1.0,0.0,0.0 } },
                                                { false,{ 0.0,1.0,0.0,1.0 } },
                                                { true ,{ 1.0,0.0,1.0,0.0 } },
                                                { true ,{ 1.0,0.0,1.0,0.0 } },
                                                { false,{ 1.0,0.0,1.0,1.0 } },
                                                { false,{ 1.0,0.0,0.0,0.0 } },
                                                { false,{ 1.0,0.0,0.0,1.0 } },
                                                { true ,{ 1.0,1.0,1.0,0.0 } },
                                                { true ,{ 1.0,1.0,1.0,0.0 } },
                                                { false,{ 1.0,1.0,1.0,1.0 } },
                                                { false,{ 1.0,1.0,0.0,0.0 } },
                                                { false,{ 1.0,1.0,0.0,1.0 } }
                                                }; 

            multiset<BalloonPoint> point_set;

            for (auto elem: points) {
                point_set.insert(BalloonPoint(elem.first, elem.second)); 
            
            }

            pop_map = PopulationMap<BalloonPoint, bool, OneNN, EulerQuality>(point_set, 1); 
        }

        static void TearDownTestCase() {
        }

        static PopulationMap<BalloonPoint, bool, OneNN, EulerQuality> pop_map; 

};

// Declare static variable
PopulationMap<BalloonPoint, bool, OneNN, EulerQuality> BallonPointTest::pop_map; 

TEST_F(BallonPointTest, PopulationMapConsistency) {

    EXPECT_EQ(pop_map.TotalSize(), 20); 
    EXPECT_EQ(pop_map.SelectedPointsSize() +  pop_map.UnselectedPointsSize(), 
                                                            pop_map.TotalSize()); 
}

#if TEST_PRIVATE_ATTRIBUTES 
TEST_F(BallonPointTest, FitnessFunction) {
    PopulationMap<BalloonPoint, bool, OneNN, EulerQuality> copy_map(pop_map); 
    float classification_correctness = copy_map.RunClassifier(copy_map.selected_points_, copy_map.selected_points_); 

    EXPECT_EQ(classification_correctness, 1); 

    BalloonPoint p = *pop_map.selected_points().begin(); 
    copy_map.toggle(p); 
    float reduction_percentage = (float) copy_map.GetReductionPercentage(); 
    EXPECT_LT(0.049, reduction_percentage); 
    EXPECT_GT(0.051, reduction_percentage); 

    float euler_score = EulerQuality(classification_correctness, reduction_percentage, 0.5); 

    EXPECT_LT(1.050, euler_score); 
    EXPECT_GT(1.052, euler_score); 
}

TEST_F(BallonPointTest, TogglingPoints) {
    PopulationMap<BalloonPoint, bool, OneNN, EulerQuality> copy_map(pop_map); 

    BalloonPoint p = *copy_map.selected_points_.begin(); 

    // The point is not in the unselected_points set
    EXPECT_EQ(copy_map.unselected_points_.find(p), copy_map.unselected_points_.end()); 

    // Toggle point, the point is now in the unselected point set and not in the
    // selected point set
    copy_map.toggle(p); 
    EXPECT_EQ(copy_map.selected_points_.find(p), copy_map.selected_points_.end()); 
    EXPECT_NE(copy_map.unselected_points_.find(p), copy_map.unselected_points_.end()); 
}
#endif

TEST_F(BallonPointTest, Classifier) {
    BalloonPoint p = *pop_map.selected_points().begin(); 
    bool resulting_class = OneNN<bool, BalloonPoint>(p, pop_map.selected_points()); 
    EXPECT_EQ(p.ClassLabel(), resulting_class); 
}

TEST_F(BallonPointTest, NeighborhoodOperator) {
    PopulationMap<BalloonPoint, bool, OneNN, EulerQuality> copy_map(pop_map); 

    int number_of_selected_points   = copy_map.SelectedPointsSize();
    int number_of_unselected_points = copy_map.UnselectedPointsSize();

    copy_map.NeighborhoodOperator(); 

    // Toggled one point from random sets
    EXPECT_TRUE(number_of_selected_points == copy_map.SelectedPointsSize() + 1 ||
                number_of_unselected_points == copy_map.UnselectedPointsSize() + 1);
}

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}
