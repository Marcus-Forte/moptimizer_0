#include <gtest/gtest.h>
#include <duna/levenberg_marquadt.h>
#include <duna/cost_function_numerical.h>

// Function to be minimized 
struct Model 
{
    Model(float *x, float *y) : data_x(x), data_y(y) {}
    // API simply has to override this method
    void setup(const float* x) 
    {

    }
    void operator()(const float *x, float *residual, unsigned int index) 
    {
        residual[0] = data_y[index] - (x[0] * data_x[index]) / (x[1] + data_x[index]);
    }

private:
    const float *const data_x;
    const float *const data_y;
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

class SimpleModel : public testing::Test
{
public:
    SimpleModel()
    {
        cost = new duna::CostFunctionNumericalDiff<Model,float, 2, 1>(
            new Model(x_data, y_data),
            7);


        optimizer.setCost(cost);
    }

    ~SimpleModel()
    {
        delete cost;
    }

protected:
    
    duna::LevenbergMarquadt<float, 2> optimizer;
    duna::CostFunctionNumericalDiff<Model,float, 2, 1> *cost;
    float x_data[7] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
    float y_data[7] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};
};

TEST_F(SimpleModel, InitialCondition0)
{
    float x0[] = {0.9, 0.2};

    optimizer.minimize(x0);

    EXPECT_NEAR(x0[0], 0.362, 0.01);
    EXPECT_NEAR(x0[1], 0.556, 0.01);
}

TEST_F(SimpleModel, InitialCondition1)
{
    float x0[] = {1.9, 4.2};
    optimizer.minimize(x0);

    EXPECT_NEAR(x0[0], 0.362, 0.01);
    EXPECT_NEAR(x0[1], 0.556, 0.01);
}