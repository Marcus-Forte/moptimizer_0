#include <moptimizer/cost_function_numerical.h>
#include <moptimizer/cost_function_numerical_dyn.h>
#include <moptimizer/levenberg_marquadt_dyn.h>
#include <moptimizer/model.h>
#include <gtest/gtest.h>

#include "test_models.h"
using Scalar = float;

class SimpleModel : public testing::Test {
 public:
  SimpleModel() : optimizer(2) {
    cost = new moptimizer::CostFunctionNumerical<Scalar, 2, 1>(
        Model<Scalar>::Ptr(new Model<Scalar>(x_data, y_data)), 7);

    optimizer.addCost(cost);
  }

  ~SimpleModel() { delete cost; }

 protected:
  moptimizer::LevenbergMarquadtDynamic<Scalar> optimizer;
  moptimizer::CostFunctionNumerical<Scalar, 2, 1> *cost;
  Scalar x_data[7] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
  Scalar y_data[7] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};
};

TEST_F(SimpleModel, InitialCondition0) {
  Scalar x0[] = {0.9, 0.2};

  optimizer.minimize(x0);

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);
}

TEST_F(SimpleModel, InitialCondition1) {
  Scalar x0[] = {1.9, 1.5};
  optimizer.minimize(x0);

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);
}

TEST_F(SimpleModel, InitialCondition0Dynamic) {
  Scalar x0[] = {0.9, 0.2};
  moptimizer::LevenbergMarquadtDynamic<Scalar> dyn_optimizer(2);

  dyn_optimizer.addCost(cost);

  dyn_optimizer.minimize(x0);

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);
}

TEST_F(SimpleModel, InitialCondition1Dynamic) {
  Scalar x0[] = {1.9, 1.5};
  moptimizer::LevenbergMarquadtDynamic<Scalar> dyn_optimizer(2);
  dyn_optimizer.addCost(cost);

  dyn_optimizer.minimize(x0);

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);
}

TEST_F(SimpleModel, InitialCondition1DynamicCost) {
  Scalar x0[] = {1.9, 1.5};
  moptimizer::LevenbergMarquadtDynamic<Scalar> dyn_optimizer(2);

  auto *dyn_cost = new moptimizer::CostFunctionNumericalDynamic<Scalar>(
      Model<Scalar>::Ptr(new Model<Scalar>(x_data, y_data)), 2, 1, 7);

  dyn_optimizer.addCost(dyn_cost);
  dyn_optimizer.minimize(x0);

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);

  delete dyn_cost;
}