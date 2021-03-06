#include "instance_selection.hpp"
#include "point_instances.hpp"
#include "fitness.hpp"
#include "classifiers.hpp"
#include "gtest-1.7.0/include/gtest/gtest.h"

using std::pair;
int g_max_label = 0;

// attributes_[1] == SMALL/LARGE   (0/1)
// attributes_[2] == DIP/STRETCH   (0/1)
// attributes_[3] == ADULT/CHILD   (0/1)
class GenericPointTest : public ::testing::Test {
    protected:
        static void SetUpTestCase()
        {
            vector<pair<int, vector<double> > > points = {
                                                { 1,{ 0.0,0.0,1.0,0.0 } },
                                                { 0,{ 0.0,0.0,1.0,1.0 } },
                                                { 0,{ 0.0,0.0,0.0,0.0 } },
                                                { 0,{ 0.0,0.0,0.0,1.0 } },
                                                { 1,{ 0.0,1.0,1.0,0.0 } },
                                                { 0,{ 0.0,1.0,1.0,1.0 } },
                                                { 0,{ 0.0,1.0,0.0,0.0 } },
                                                { 0,{ 0.0,1.0,0.0,1.0 } },
                                                { 1,{ 1.0,0.0,1.0,0.0 } },
                                                { 0,{ 1.0,0.0,1.0,1.0 } },
                                                { 0,{ 1.0,0.0,0.0,0.0 } },
                                                { 0,{ 1.0,0.0,0.0,1.0 } },
                                                { 1,{ 1.0,1.0,1.0,0.0 } },
                                                { 0,{ 1.0,1.0,1.0,1.0 } },
                                                { 0,{ 1.0,1.0,0.0,0.0 } },
                                                { 0,{ 1.0,1.0,0.0,1.0 } }
                                                };

            set<GenericPoint> point_set;

            for (auto elem: points) {
                point_set.insert(GenericPoint(elem.first, elem.second));

            }

            pop_map = PopulationMap<GenericPoint, int>(point_set, 1, &OneNN<GenericPoint, int>, &WeightedQuality, LOCAL_SEARCH); 
        }

        static void TearDownTestCase() {
        }

        static PopulationMap<GenericPoint, int> pop_map;

};

// Declare static variable
PopulationMap<GenericPoint, int> GenericPointTest::pop_map;

TEST_F(GenericPointTest, PopulationMapConsistency) {

    EXPECT_EQ(pop_map.TotalSize(), 16);
    EXPECT_EQ(pop_map.SelectedPointsSize() +  pop_map.UnselectedPointsSize(),
                                                            pop_map.TotalSize());
}

#if TEST_PRIVATE_ATTRIBUTES
TEST_F(GenericPointTest, FitnessFunction) {
    PopulationMap<GenericPoint, int> copy_map(pop_map);
    float classification_correctness = copy_map.RunClassifier(copy_map.selected_points_, copy_map.selected_points_);

    EXPECT_EQ(classification_correctness, 1);

    GenericPoint p = *pop_map.SelectedPoints().begin();
    copy_map.ComputeCentroidsAndTotals(); 
    copy_map.toggle(p, 1);
    float reduction_percentage = (float) copy_map.GetReductionPercentage();
    EXPECT_LT(0.049, reduction_percentage);
    EXPECT_GT(0.051, reduction_percentage);

    float euler_score = EulerQuality(classification_correctness, reduction_percentage, 0.5);

    EXPECT_LE(0, euler_score);
    EXPECT_GE(1, euler_score);
}

TEST_F(GenericPointTest, TogglingPoints) {
    PopulationMap<GenericPoint, int> copy_map(pop_map);

    GenericPoint p = *copy_map.selected_points_.begin();

    // The point is not in the unselected_points set
    EXPECT_EQ(copy_map.unselected_points_.find(p), copy_map.unselected_points_.end());

    // Toggle point, the point is now in the unselected point set and not in the
    // selected point set
    copy_map.toggle(p, 1);
    EXPECT_EQ(copy_map.selected_points_.find(p), copy_map.selected_points_.end());
    EXPECT_NE(copy_map.unselected_points_.find(p), copy_map.unselected_points_.end());
}

TEST_F(GenericPointTest, ComputingCentroids) {
    PopulationMap<GenericPoint, int> copy_map(pop_map);
    copy_map.ComputeCentroidsAndTotals(); 

    EXPECT_EQ(copy_map.class_frequencies_[1], 8); 
    EXPECT_EQ(copy_map.class_frequencies_[0], 12); 

    vector<double> total_one = { 4.0, 4.0, 8.0, 0.0 }; 
    EXPECT_EQ(copy_map.class_totals_[1], total_one); 

    vector<double> total_zero = { 6.0, 6.0, 4.0, 8.0 }; 
    EXPECT_EQ(copy_map.class_totals_[0], total_zero); 

    vector<double> centroid_one = { 0.5, 0.5, 1.0, 0.0 };
    EXPECT_EQ(copy_map.class_centroids_[1][0], centroid_one[0]); 
    EXPECT_EQ(copy_map.class_centroids_[1][1], centroid_one[1]); 
    EXPECT_EQ(copy_map.class_centroids_[1][2], centroid_one[2]); 
    EXPECT_EQ(copy_map.class_centroids_[1][3], centroid_one[3]); 

    vector<double> centroid_zero = { 0.5, 0.5, 0.333333, 0.666667 };
    EXPECT_EQ(copy_map.class_centroids_[0][0], centroid_zero[0]); 
    EXPECT_EQ(copy_map.class_centroids_[0][1], centroid_zero[1]); 
    EXPECT_LT(copy_map.class_centroids_[0][2], centroid_zero[2] + 0.000001); 
    EXPECT_GT(copy_map.class_centroids_[0][2], centroid_zero[2] - 0.000001); 
    EXPECT_LT(copy_map.class_centroids_[0][3], centroid_zero[3] + 0.000001); 
    EXPECT_GT(copy_map.class_centroids_[0][3], centroid_zero[3] - 0.000001); 
}

TEST_F(GenericPointTest, ComputingCentroidDistance) { 
    PopulationMap<GenericPoint, int> copy_map(pop_map);
    copy_map.ComputeCentroidsAndTotals(); 

    GenericPoint p = *copy_map.selected_points_.begin(); 
    float distance = copy_map.ComputeCentroidDistance(p, 
                                                      copy_map.class_centroids_[p.ClassLabel()], 
                                                      copy_map.class_totals_[p.ClassLabel()],
                                                      copy_map.class_frequencies_[p.ClassLabel()],
                                                      1);

    EXPECT_GT(0.0935,distance);
    EXPECT_LT(0.0933, distance);
}

TEST_F(GenericPointTest, GettingBestPoint) {
    PopulationMap<GenericPoint, int> copy_map(pop_map);
    copy_map.ComputeCentroidsAndTotals(); 

    GenericPoint p = copy_map.GetBestPoint(1).first; 
    copy_map.set_to_perturb_ = -1; 
    float distance = copy_map.ComputeCentroidDistance(p,  
                                                      copy_map.class_centroids_[p.ClassLabel()], 
                                                      copy_map.class_totals_[p.ClassLabel()],
                                                      copy_map.class_frequencies_[p.ClassLabel()],
                                                      1);

    for (GenericPoint elem : copy_map.selected_points_) {
        float elem_distance = copy_map.ComputeCentroidDistance(elem, 
                                                               copy_map.class_centroids_[elem.ClassLabel()], 
                                                               copy_map.class_totals_[elem.ClassLabel()],
                                                               copy_map.class_frequencies_[elem.ClassLabel()],
                                                               1);
        EXPECT_GE(elem_distance, distance);  
    }

    EXPECT_LT(0, p.attributes().size()); 

    //EXPECT_NE(copy_map.selected_points_.find(p), copy_map.selected_points_.end()); 
}
TEST_F(GenericPointTest, NeighborhoodOperatorWithIntelligentPerturbation) {
    PopulationMap<GenericPoint, int> copy_map(pop_map);
    int number_of_selected_points   = copy_map.SelectedPointsSize();
    int number_of_unselected_points = copy_map.UnselectedPointsSize();

    copy_map.ComputeCentroidsAndTotals(); 
    copy_map.NeighborhoodOperator(true);

    // Toggled one point from random sets
    EXPECT_TRUE(number_of_selected_points == copy_map.SelectedPointsSize() + 1 ||
                number_of_unselected_points == copy_map.UnselectedPointsSize() + 1);
}

TEST_F(GenericPointTest, NeighborhoodOperatorIntelligentTwice) {
    PopulationMap<GenericPoint, int> copy_map(pop_map);

    copy_map.ComputeCentroidsAndTotals(); 

    GenericPoint p = copy_map.GetBestPoint(1).first; 
    EXPECT_EQ(copy_map.unselected_points_.find(p), copy_map.unselected_points_.end());

    copy_map.toggle(p, 1);
    GenericPoint np = copy_map.GetBestPoint(1).first; 
    EXPECT_NE(copy_map.unselected_points_.find(p), copy_map.unselected_points_.end());
    EXPECT_EQ(copy_map.unselected_points_.find(np), copy_map.unselected_points_.end());

    EXPECT_NE(p.attributes(), np.attributes()); 
}

TEST_F(GenericPointTest, PickRandomSet) {
    PopulationMap<GenericPoint, int> copy_map(pop_map);
    set<GenericPoint> data(copy_map.data()); 
    auto sets = PopulationMap<GenericPoint,int>::PickRandomSet(data);
    set<GenericPoint> random_set = sets.first;
    set<GenericPoint> rest = sets.second;


    // Expecting at least one element
    EXPECT_LT(0, random_set.size()); 
    EXPECT_EQ(random_set.size() + rest.size(), data.size()); 

    for (GenericPoint p : random_set) {
        EXPECT_TRUE(data.find(p) != data.end()); 
    }
}

TEST_F(GenericPointTest, EvaluateIncrementalCostEmpty) {
    PopulationMap<GenericPoint, int> copy_map(pop_map);
    copy_map.ComputeCentroidsAndTotals(); 

    vector<pair<GenericPoint, double> > inc_costs = PopulationMap<GenericPoint,int>::EvaluateIncrementalCost(copy_map.UnselectedPoints(),
                                                                                                             copy_map.class_centroids_,
                                                                                                             copy_map.class_totals_,
                                                                                                             copy_map.class_frequencies_,
                                                                                                             0); 
    for (auto p : inc_costs) {
        double distance = PopulationMap<GenericPoint,int>::ComputeCentroidDistance(p.first, 
                                                                                  copy_map.class_centroids_[p.first.ClassLabel()], 
                                                                                  copy_map.class_totals_[p.first.ClassLabel()],
                                                                                  copy_map.class_frequencies_[p.first.ClassLabel()],
                                                                                  0);
        EXPECT_LT(p.second, distance + 0.0001); 
        EXPECT_GT(p.second, distance - 0.0001); 
    }
}

#endif

TEST_F(GenericPointTest, Classifier) {
    GenericPoint p = *pop_map.SelectedPoints().begin();
    bool resulting_class = OneNN<GenericPoint, int>(p, pop_map.SelectedPoints());
    EXPECT_EQ(p.ClassLabel(), resulting_class);
}

TEST_F(GenericPointTest, NeighborhoodOperator) {
    PopulationMap<GenericPoint, int> copy_map(pop_map);

    int number_of_selected_points   = copy_map.SelectedPointsSize();
    int number_of_unselected_points = copy_map.UnselectedPointsSize();

    copy_map.NeighborhoodOperator(false);

    // Toggled one point from random sets
    EXPECT_TRUE(number_of_selected_points == copy_map.SelectedPointsSize() + 1 ||
                number_of_unselected_points == copy_map.UnselectedPointsSize() + 1);
}

TEST_F(GenericPointTest, GreedyRandomAlgorithm) {
    set<GenericPoint> data(pop_map.data()); 

    PopulationMap<GenericPoint, int> random_map = 
        PopulationMap<GenericPoint, int>::GreedyRandomAlgorithm(data, 0.5, pop_map.classifier(), pop_map.evaluator(), pop_map.mht()); 

    EXPECT_EQ(random_map.UnselectedPointsSize() + random_map.SelectedPointsSize(), data.size()); 

    for (GenericPoint p : random_map.SelectedPoints()) {
        EXPECT_TRUE(data.find(p) != data.end()); 
    }

    for (GenericPoint p : random_map.UnselectedPoints()) {
        EXPECT_TRUE(data.find(p) != data.end()); 
    }
}

TEST_F(GenericPointTest, GenerateRandomPopulation) {

    set<PopulationMap<GenericPoint,int> > population = 
        PopulationMap<GenericPoint,int>::GenerateRandomPopulation(50, pop_map.data()); 

    EXPECT_EQ(50, population.size()); 
}

TEST_F(GenericPointTest, GetBestSolutionFromRandomPopulation) {
    set<PopulationMap<GenericPoint,int> > population = 
        PopulationMap<GenericPoint,int>::GenerateRandomPopulation(50, pop_map.data()); 

    PopulationMap<GenericPoint,int> best_solution(PopulationMap<GenericPoint,int>::GetBestSolution(population)); 

    double best_score = best_solution.EvaluateQuality(); 
    for (auto pm : population) {
        EXPECT_LE(pm.EvaluateQuality(), best_score); 
    }
}

TEST_F(GenericPointTest, SelectParentsFromPopulation) {
    set<PopulationMap<GenericPoint,int> > population = 
        PopulationMap<GenericPoint,int>::GenerateRandomPopulation(50, pop_map.data()); 

    auto parents = PopulationMap<GenericPoint,int>::select(population); 

    EXPECT_TRUE(population.find(parents[0]) != population.end()); 
    EXPECT_TRUE(population.find(parents[1]) != population.end()); 
}

TEST_F(GenericPointTest, MutateChildren) {
    PopulationMap<GenericPoint,int> copy_map(pop_map); 

    copy_map.mutate(5, 1); 

    EXPECT_NE(copy_map.SelectedPoints(), pop_map.SelectedPoints()); 
}

TEST(ClassifierTest, OneNNTestGood) {
    
    vector<double> attrs = { 0.0,0.0,1.0,0.0 }; 
    int class_label = 1; 
    GenericPoint p(class_label, attrs); 

    set<GenericPoint> trs; 
    trs.insert(p); 

    int result_class = OneNN<GenericPoint,int>(p,trs); 
    EXPECT_EQ(class_label, result_class); 
}

TEST(ClassifierTest, OneNNTestBad) {
    
    vector<double> attrs = { 0.0,0.0,1.0,0.0 }; 
    int class_label = 1; 
    GenericPoint p(class_label, attrs); 

    set<GenericPoint> trs; 
    trs.insert(p); 

    vector<double> attrs_w = { 1.0,1.0,1.0,0.0 }; 
    int class_label_w = 2; 
    GenericPoint p_w(class_label_w, attrs_w); 

    int result_class = OneNN<GenericPoint,int>(p_w,trs); 
    EXPECT_NE(class_label_w, result_class); 
    EXPECT_EQ(class_label, result_class); 
}



int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
