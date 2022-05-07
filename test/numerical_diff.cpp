#include <gtest/gtest.h>
#include <duna/cost_function.h>
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

    // NumDIFF
    void df(const Scalar *x, Scalar *jacobian, unsigned int index)
    {
        Scalar denominator = ( x[1] + data_x[index] );

        jacobian[0] = -data_x[index] / denominator;
        jacobian[1] = (x[0] * data_x[index]) /  (denominator*denominator);
    }

private:
    const Scalar *const data_x;
    const Scalar *const data_y;
};

class TestNumericalDifferentiation : public ::testing::Test
{

protected:
};

TEST_F(TestNumericalDifferentiation, SimpleModel)
{
    using SCALAR = double;
    SCALAR x_data[] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70, 5, 0};
    SCALAR y_data[] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317, 0.2, 0};
    int m_residuals = sizeof(x_data) / sizeof(SCALAR);

    SimpleModel<SCALAR> model(x_data, y_data);

    duna::CostFunction<SimpleModel<SCALAR>, SCALAR, 2, 1> cost(&model, m_residuals);

    Eigen::Matrix<SCALAR,2,1> x0(0.9, 0.2);
    Eigen::Matrix<SCALAR,2,2> hessian;
    Eigen::Matrix<SCALAR,2,1> b;

    Eigen::Matrix<SCALAR,-1,-1> jacobian;
    cost.linearize(x0, hessian, b, &jacobian);

    Eigen::Matrix<SCALAR,-1,-1> analitic_jacobian;
    analitic_jacobian.resize(m_residuals,2);
    SCALAR jac_row[2];
    for(int i=0; i < m_residuals; ++i)
    {
        model.df(x0.data(),jac_row,i);
        analitic_jacobian(i,0) = jac_row[0];
        analitic_jacobian(i,1) = jac_row[1];
    }  

    std::cout << analitic_jacobian << std::endl << std::endl;
    std::cout << jacobian << std::endl << std::endl;

    for(int i=0; i < analitic_jacobian.size(); i++)
    {
        EXPECT_NEAR(analitic_jacobian(i),jacobian(i), 1e-4);
    }
}