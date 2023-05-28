#include <duna/covariance/covariance.h>
#include <gtest/gtest.h>

/* Client covariance */
class Covariance3x3 : public duna::covariance::ICovariance<double> {
  using MatrixType = typename duna::covariance::ICovariance<double>::MatrixType;

 public:
  Covariance3x3() {
    constantCovariance.resize(3, 3);
    constantCovariance.setIdentity();
    constantCovariance *= 10.0;
  }

  MatrixType getCovariance(double *input) override {
    return constantCovariance;
  }

 private:
  MatrixType constantCovariance;
};

/* Client covariance */
class Covariance6x6 : public duna::covariance::ICovariance<double> {
  using MatrixType = typename duna::covariance::ICovariance<double>::MatrixType;

 public:
  Covariance6x6() {
    constantCovariance.resize(6, 6);
    constantCovariance.setIdentity();
    constantCovariance *= 0.02;
  }

  MatrixType getCovariance(double *input) override {
    return constantCovariance;
  }

 private:
  MatrixType constantCovariance;
};

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

template <typename Scalar>
class Covariance : public ::testing::Test {};
using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Covariance, ScalarTypes);

TYPED_TEST(Covariance, getNoCovariance) {
  auto cov_obj = duna::covariance::IdentityCovariance<TypeParam>(1);
  auto covariance = cov_obj.getCovariance();
  GTEST_ASSERT_EQ(covariance(0), 1.0f);
  GTEST_ASSERT_EQ(covariance.size(), 1);
}

TEST(Covariance, getCovariance) {
  duna::covariance::ICovariance<double>::Ptr icov;
  icov = std::make_shared<Covariance3x3>();

  auto covariance = icov->getCovariance();
  GTEST_ASSERT_EQ(covariance(0, 0), 10.0);
  GTEST_ASSERT_EQ(covariance(1, 1), 10.0);
  GTEST_ASSERT_EQ(covariance(2, 2), 10.0);
  GTEST_ASSERT_EQ(covariance.rows(), 3);
  GTEST_ASSERT_EQ(covariance.cols(), 3);

  icov = std::make_shared<Covariance6x6>();

  covariance = icov->getCovariance();
  GTEST_ASSERT_EQ(covariance(0, 0), 0.02);
  GTEST_ASSERT_EQ(covariance(1, 1), 0.02);
  GTEST_ASSERT_EQ(covariance(2, 2), 0.02);
  GTEST_ASSERT_EQ(covariance(3, 3), 0.02);
  GTEST_ASSERT_EQ(covariance(4, 4), 0.02);
  GTEST_ASSERT_EQ(covariance(5, 5), 0.02);
  GTEST_ASSERT_EQ(covariance.rows(), 6);
  GTEST_ASSERT_EQ(covariance.cols(), 6);
}