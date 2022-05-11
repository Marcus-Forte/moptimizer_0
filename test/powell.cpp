#include <gtest/gtest.h>
#include <duna/levenberg_marquadt.h>
#include <duna/cost_function_numerical.h>
#include <duna/stopwatch.hpp>

//  Powell's singular function.
//
//   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
//
//   f1 = x1 + 10*x2;
//   f2 = sqrt(5) * (x3 - x4)
//   f3 = (x2 - 2*x3)^2
//   f4 = sqrt(10) * (x1 - x4)^2

// The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
// The minimum is 0 at (x1, x2, x3, x4) = 0.

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

struct Model
{

    void setup(const double *x)
    {
    }

    void operator()(const double *x, double *f_x, unsigned int index)
    {
        f_x[0] = x[0] + 10 * x[1];
        f_x[1] = sqrt(5) * (x[2] - x[3]);
        f_x[2] = (x[1] - 2 * x[2]) * (x[1] - 2 * x[2]);
        f_x[3] = sqrt(10) * (x[0] - x[3]) * (x[0] - x[3]);
    }
};

TEST(PowellFunction, InitialCondition0)
{
    //
    utilities::Stopwatch timer(true);
    timer.tick();
    double x0[] = {3, -1, 0, 4};

    duna::LevenbergMarquadt<double, 4, 4> optimizer;
    optimizer.setMaximumIterations(50);

    optimizer.setCost(new duna::CostFunctionNumericalDiff<Model, double, 4, 4>(
        new Model, 4));

    optimizer.minimize(x0);

    timer.tock("Power Function minimzation");

    for(int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(x0[i],0.0, 1e-5);
    }

    
}