#include <duna_optimizer/cost_function_numerical.h>
#include <duna_optimizer/cost_function_numerical_dynamic.h>
#include <duna_optimizer/levenberg_marquadt.h>
#include <duna_optimizer/levenberg_marquadt_dynamic.h>
#include <duna_optimizer/model.h>
#include <gtest/gtest.h>

using Scalar = float;

// Function to be minimized
struct Model : public duna_optimizer::BaseModel<Scalar, Model> {
  Model(Scalar *x, Scalar *y) : data_x(x), data_y(y) {}
  // API simply has to override this method

  bool f(const Scalar *x, Scalar *residual, unsigned int index) const override {
    residual[0] = data_y[index] - (x[0] * data_x[index]) / (x[1] + data_x[index]);
    return true;
  }

 private:
  const Scalar *const data_x;
  const Scalar *const data_y;
};

class SimpleModel : public testing::Test {
 public:
  SimpleModel() {
    cost = new duna_optimizer::CostFunctionNumerical<Scalar, 2, 1>(
        Model::Ptr(new Model(x_data, y_data)), 7);

    optimizer.addCost(cost);
  }

  ~SimpleModel() { delete cost; }

 protected:
  duna_optimizer::LevenbergMarquadt<Scalar, 2> optimizer;
  duna_optimizer::CostFunctionNumerical<Scalar, 2, 1> *cost;
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
  duna_optimizer::LevenbergMarquadtDynamic<Scalar> dyn_optimizer(2);

  dyn_optimizer.addCost(cost);

  dyn_optimizer.minimize(x0);

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);
}

TEST_F(SimpleModel, InitialCondition1Dynamic) {
  Scalar x0[] = {1.9, 1.5};
  duna_optimizer::LevenbergMarquadtDynamic<Scalar> dyn_optimizer(2);
  dyn_optimizer.addCost(cost);

  dyn_optimizer.minimize(x0);

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);
}

TEST_F(SimpleModel, InitialCondition1DynamicCost) {
  Scalar x0[] = {1.9, 1.5};
  duna_optimizer::LevenbergMarquadtDynamic<Scalar> dyn_optimizer(2);

  duna_optimizer::CostFunctionNumericalDynamic<Scalar> *dyn_cost =
      new duna_optimizer::CostFunctionNumericalDynamic<Scalar>(
          Model::Ptr(new Model(x_data, y_data)), 2, 1, 7);
  dyn_optimizer.addCost(dyn_cost);
  dyn_optimizer.minimize(x0);

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);

  delete dyn_cost;
}