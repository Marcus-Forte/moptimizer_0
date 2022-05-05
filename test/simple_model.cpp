#include <gtest/gtest.h>
#include <duna/levenberg_marquadt.h>

// Function to be minimized
struct Model
{
    Model(double *x, double *y) : data_x(x), data_y(y) {}
    // API simply has to override this method
    void setup(const double* x)
    {

    }
    void operator()(const double *x, double *residual, unsigned int index)
    {
        residual[0] = data_y[index] - (x[0] * data_x[index]) / (x[1] + data_x[index]);
    }

private:
    const double *const data_x;
    const double *const data_y;
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
        cost = new duna::CostFunction<Model, double, 2, 1>(
            new Model(x_data, y_data),
            7);

        optimizer.setCost(cost);
    }

protected:
    Eigen::Vector2d x0;
    duna::LevenbergMarquadt<double, 2, 1> optimizer;
    duna::CostFunction<Model, double, 2, 1> *cost;
    double x_data[7] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
    double y_data[7] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};
};

TEST_F(SimpleModel, InitialCondition0)
{
    x0[0] = 0.9;
    x0[1] = 0.2;

    optimizer.minimize(x0);

    EXPECT_NEAR(x0[0], 0.362, 0.01);
    EXPECT_NEAR(x0[1], 0.556, 0.01);
}

TEST_F(SimpleModel, InitialCondition1)
{
    x0[0] = 1.9;
    x0[1] = 4.2;

    optimizer.minimize(x0);

    EXPECT_NEAR(x0[0], 0.362, 0.01);
    EXPECT_NEAR(x0[1], 0.556, 0.01);
}