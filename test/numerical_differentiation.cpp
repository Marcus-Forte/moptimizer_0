#include <gtest/gtest.h>
#include <duna/cost_function.h>
#include <duna/cost_function_numerical.h>


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
        Scalar denominator = (x[1] + data_x[index]);

        jacobian[0] = -data_x[index] / denominator;
        jacobian[1] = (x[0] * data_x[index]) / (denominator * denominator);
    }

private:
    const Scalar *const data_x;
    const Scalar *const data_y;
};

template <typename Scalar>
class TestNumericalDifferentiation : public ::testing::Test
{

protected:
};

using ScalarTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(TestNumericalDifferentiation, ScalarTypes);

TYPED_TEST(TestNumericalDifferentiation, DISABLED_SimpleModel)
{
    TypeParam x_data[] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70, 5, 0};
    TypeParam y_data[] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317, 0.2, 0};
    int m_residuals = sizeof(x_data) / sizeof(TypeParam);

    SimpleModel<TypeParam> model(x_data, y_data);

    duna::CostFunctionNumericalDiff<SimpleModel<TypeParam>, TypeParam, 2, 1> cost(&model, m_residuals);
}