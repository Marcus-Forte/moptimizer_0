#include <moptimizer/cost_function_numerical_dyn.h>
#include <gtest/gtest.h>

#include <cmath>
#include <moptimizer/stopwatch.hpp>

#include "test_models.h"

float x_data[7] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
float y_data[7] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};

class testCovariance : public ::testing::Test {
 public:
  testCovariance() : cost(Model<float>::Ptr(new Model<float>(x_data, y_data)), 2, 1, 7) {
    covariance = std::make_shared<moptimizer::covariance::Matrix<float>>();
  }

 protected:
  Eigen::Matrix<float, 2, 2> hessian, hessian_with_covariance;
  Eigen::Matrix<float, 2, 1> b, b_with_covariance;
  moptimizer::covariance::MatrixPtr<float> covariance;

  moptimizer::CostFunctionNumericalDynamic<float> cost;
};

TEST_F(testCovariance, setIdentityCovariance) {
  float x0[2] = {1.9, 1.5};
  cost.linearize(x0, hessian.data(), b.data());

  covariance->resize(1, 1);
  covariance->setIdentity();
  cost.setCovariance(covariance);

  cost.linearize(x0, hessian_with_covariance.data(), b_with_covariance.data());

  for (int i = 0; i < hessian_with_covariance.size(); ++i) {
    EXPECT_NEAR(hessian_with_covariance(i), hessian(i), 1e-5);
  }

  for (int i = 0; i < b_with_covariance.size(); ++i) {
    EXPECT_NEAR(b_with_covariance(i), b(i), 1e-5);
  }
}

TEST_F(testCovariance, setLowerCovariance) {
  float x0[2] = {1.9, 1.5};
  cost.linearize(x0, hessian.data(), b.data());
  float cov_val = 0.5;

  covariance->resize(1, 1);
  (*covariance)(0, 0) = cov_val;
  cost.setCovariance(covariance);

  cost.linearize(x0, hessian_with_covariance.data(), b_with_covariance.data());

  for (int i = 0; i < hessian_with_covariance.size(); ++i) {
    EXPECT_NEAR(hessian_with_covariance(i), hessian(i) * cov_val, 1e-5);
  }

  for (int i = 0; i < b_with_covariance.size(); ++i) {
    EXPECT_NEAR(b_with_covariance(i), b(i) * cov_val, 1e-5);
  }
}