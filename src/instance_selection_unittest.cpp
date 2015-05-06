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

TEST_F(BallonPointTest, NeighborhoodOperator) {
    PopulationMap<BalloonPoint, bool, OneNN, EulerQuality> copy_map(pop_map); 

    int number_of_selected_points   = copy_map.SelectedPointsSize();
    int number_of_unselected_points = copy_map.UnselectedPointsSize();

    copy_map.NeighborhoodOperator(); 

    EXPECT_TRUE(number_of_selected_points == copy_map.SelectedPointsSize() + 1 ||
                number_of_unselected_points == copy_map.UnselectedPointsSize() + 1);
}

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}
