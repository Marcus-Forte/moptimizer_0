#include <gtest/gtest.h>

#include <duna/levenberg_marquadt.h>
#include <duna/cost_function_analytical.h>
#include <duna/cost_function_rotation_state.h>

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

TEST(MaximumAPosteriori, StateJacobian)
{
    duna::LevenbergMarquadt<double, 6> optimizator;
    Eigen::Matrix<double, 6, 1> x0;
    Eigen::Matrix<double, 6, 6> hessian;
    Eigen::Matrix<double, 6, 1> b;
    x0.setZero();

    duna::CostFunctionRotationError<double> state_cost;

    state_cost.setup(x0.data());

    // Change parameters when other optimization factors act.
    x0[0] = 0.5;   // x angle
    x0[1] = 0.2;   // y angle
    x0[2] = -0.2;  // z angle
    x0[5] = 0.226; // bias

    double y_error = state_cost.computeCost(x0.data());
    double y_error_lin = state_cost.linearize(x0.data(), hessian.data(), b.data());

    // std::cout << hessian << std::endl;
    // std::cout << b << std::endl;

    
    EXPECT_NEAR(y_error, 2 * x0.dot(x0), 1e-9);
    EXPECT_NEAR(y_error_lin, 2 * x0.dot(x0), 1e-9);

    EXPECT_NEAR(b.dot(b), x0.dot(x0), 1e-9);
}