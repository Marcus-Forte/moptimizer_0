#include <gtest/gtest.h>

#include <duna/cost_function_analytical.h>
#include <duna/cost_function_numerical.h>
#include <unsupported/Eigen/NumericalDiff>

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

template <typename Scalar = double>
struct SimpleModel
{
    SimpleModel(Scalar *x, Scalar *y) : data_x(x), data_y(y) {}

    void setup(const Scalar *x)
    {
    }
    void operator()(const Scalar *x, Scalar *residual, unsigned int index)
    {
        residual[0] = data_y[index] - (x[0] * data_x[index]) / (x[1] + data_x[index]);
    }

    // Jacobian
    void df(const Scalar *x, Scalar *jacobian, unsigned int index)
    {
        Scalar denominator = (x[1] + data_x[index]);

        // Col Major
        jacobian[0] = -data_x[index] / denominator;
        jacobian[1] = (x[0] * data_x[index]) / (denominator * denominator);
    }

private:
    const Scalar *const data_x;
    const Scalar *const data_y;
};

template <typename Scalar>
class TestAnalyticalDifferentiation : public ::testing::Test
{

protected:
};

using ScalarTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(TestAnalyticalDifferentiation, ScalarTypes);

TYPED_TEST(TestAnalyticalDifferentiation, SimpleModel)
{
    TypeParam x_data[] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70, 5, 0};
    TypeParam y_data[] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317, 0.2, 0};
    int m_residuals = sizeof(x_data) / sizeof(TypeParam);

    SimpleModel<TypeParam> model(x_data, y_data);

    duna::CostFunctionAnalytical<SimpleModel<TypeParam>, TypeParam, 2, 1> cost_ana(&model, m_residuals);
    duna::CostFunctionNumericalDiff<SimpleModel<TypeParam>, TypeParam, 2, 1> cost_num(&model, m_residuals);

    Eigen::Matrix<TypeParam, 2, 2> Hessian;
    Eigen::Matrix<TypeParam, 2, 2> HessianNum;
    Eigen::Matrix<TypeParam, 2, 1> Residuals;
    Eigen::Matrix<TypeParam, 2, 1> x0(0.9, 0.2);

    cost_ana.linearize(x0, Hessian, Residuals);
    cost_num.linearize(x0, HessianNum, Residuals);

    for (int i = 0; i < Hessian.size(); ++i)
    {
        EXPECT_NEAR(Hessian(i), HessianNum(i), 1e-3);
    }
}

struct Powell
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

    // Should be 4 x 4 = 16. Eigen stores column major order, so we fill indices accordingly.

    /*      [ df0/dx0 df0/dx1 df0/dx2 df0/dx3 ]
    Jac =   [ df1/dx0 df1/dx1 df2/dx2 df1/dx3 ]
            [ df2/dx0 df2/dx1 df2/dx2 df2/dx3 ]
            [ df3/dx0 df3/dx1 df3/dx2 df3/dx3 ]
            
            jacobian[0] := df0/dx0
            jacobian[1] := df1/dx0
                    .
                    .
                    .
    */
    void df(const double *x, double *jacobian, unsigned int index)
    {

        // Df / dx0
        jacobian[0] = 1;
        jacobian[1] = 0;
        jacobian[2] = 0;
        jacobian[3] = sqrt(10) * 2 * (x[0] - x[3]);

        // Df / dx1
        jacobian[4] = 10;
        jacobian[5] = 0;
        jacobian[6] = 2 * (x[1] + 2 * x[2]);
        jacobian[7] = 0;

        // Df / dx2
        jacobian[8] = 0;
        jacobian[9] = sqrt(5);
        jacobian[10] = 2 * (x[1] + 2 * x[2]) * (-2);
        jacobian[11] = 0;

        // Df / dx3
        jacobian[12] = 0;
        jacobian[13] = -sqrt(5);
        jacobian[14] = 0;
        jacobian[15] = sqrt(10) * 2 * (x[0] - x[3]) * (-1);
    }
};

TEST(TestAnalyticalDifferentiation, PowellModel)
{

    const int m_residuals = 4;

    Powell powell;

    duna::CostFunctionAnalytical<Powell, double, 4, 4> cost_ana(&powell, m_residuals);
    duna::CostFunctionNumericalDiff<Powell, double, 4, 4> cost_num(&powell, m_residuals);

    Eigen::Matrix<double, 4, 4> Hessian;
    Eigen::Matrix<double, 4, 4> HessianNum;
    Eigen::Matrix<double, 4, 1> x0(3, -1, 0, 4);
    Eigen::Matrix<double, 4, 1> residuals;

    cost_ana.linearize(x0, Hessian, residuals);
    cost_num.linearize(x0, HessianNum, residuals);

    for (int i = 0; i < Hessian.size(); ++i)
    {
        EXPECT_NEAR(Hessian(i), HessianNum(i), 1e-3);
    }

    std::cerr << Hessian << std::endl;
    std::cerr << HessianNum << std::endl;
}