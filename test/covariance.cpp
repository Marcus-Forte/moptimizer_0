#include <gtest/gtest.h>
#include <duna/covariance/covariance.h>

// User Define covariance.
class Covariance3X3 : public duna::covariance::ICovariance<double, 3>
{
    using MatrixType = duna::covariance::ICovariance<double, 3>::MatrixType;

    MatrixType getCovariance(double *input = 0) override
    {
        return 0.02 * MatrixType::Identity();
    }
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

template <typename Scalar>
class Covariance : public ::testing::Test
{
};
using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Covariance, ScalarTypes);

TYPED_TEST(Covariance, getNoCovariance)
{
    auto cov = duna::covariance::NoCovariance<TypeParam>();
    auto val = cov.getCovariance();
    GTEST_ASSERT_EQ(val(0), 1.0);
    GTEST_ASSERT_EQ(val.size(), 1);
}

TEST(Covariance, get3x3Covariance)
{
    duna::covariance::ICovariance<double, 3>::Ptr covariance;
    covariance.reset(new Covariance3X3);
    auto val = covariance->getCovariance();
    GTEST_ASSERT_EQ(val.rows(), 3);
    GTEST_ASSERT_EQ(val.cols(), 3);

    GTEST_ASSERT_EQ(val(0, 0), 0.02);
    GTEST_ASSERT_EQ(val(1, 1), 0.02);
    GTEST_ASSERT_EQ(val(2, 2), 0.02);
}

template <typename T>
class Base
{
public:
    virtual T getCov() = 0;
};

template <typename T, int DIM>
class Derived : public Base<T>
{
    T getCov() override
    {

    }
};

TEST(Covariance, base)
{
    Base<double>* base;
    base = new Derived<double, 3>;



}