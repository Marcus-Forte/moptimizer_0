#include "generic_optimizator.h"
#include "model_cost.hpp"
#include "duna_log.h"

#include <gtest/gtest.h>

using VectorN = GenericOptimizator<2>::VectorN;

//     model: y(x) = b0*x / (b1 + x)
// i 	1 	2 	3 	4 	5 	6 	7
// [S] 	0.038 	0.194 	0.425 	0.626 	1.253 	2.500 	3.740
// Rate 	0.050 	0.127 	0.094 	0.2122 	0.2729 	0.2665 	0.3317
// */

float x_data[] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
float y_data[] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};
test_dataype1_t input_set = {x_data, y_data};

// Cost Class
ModelCost<2> cost(7, &input_set);

GenericOptimizator<2>::VectorN x0;

// Pass the cost class
GenericOptimizator<2> optimizator(&cost);

// TODO use classes to encapsulate all testing
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    for (int i = 1; i < argc; ++i)
    {
        printf("arg: %2d = %s\n", i, argv[i]);
    }

    optimizator.setMaxIt(10);

    return RUN_ALL_TESTS();
}



TEST(ModelTest, Test1)
{

    // INITIAL CONDITIONS
    x0[0] = 0.9;
    x0[1] = 0.2;
    optimizator.minimize(x0);

    EXPECT_NEAR(x0[0], 0.362, 0.05);
    EXPECT_NEAR(x0[1], 0.556, 0.05);
}

TEST(ModelTest, Test2)
{  
    // INITIAL CONDITIONS
    x0[0] = 0.1;
    x0[1] = 0.1;
    optimizator.minimize(x0);

    EXPECT_NEAR(x0[0], 0.362, 0.05);
    EXPECT_NEAR(x0[1], 0.556, 0.05);
}

TEST(ModelTest, Test3)
{
    
    // INITIAL CONDITIONS
    x0[0] = 0.5;
    x0[1] = 0.5;
    optimizator.minimize(x0);

    EXPECT_NEAR(x0[0], 0.362, 0.05);
    EXPECT_NEAR(x0[1], 0.556, 0.05);
}

TEST(ModelTest, Test4)
{

    // INITIAL CONDITIONS
    x0[0] = 5.5;
    x0[1] = 0.9;
    optimizator.minimize(x0);

    EXPECT_NEAR(x0[0], 0.362, 0.05);
    EXPECT_NEAR(x0[1], 0.556, 0.05);
}