#include <moptimizer/cost_function_numerical.h>
#include <moptimizer/levenberg_marquadt_dyn.h>
#include <moptimizer/loss_function/geman_mcclure.h>
#include <moptimizer/model.h>
#include <gtest/gtest.h>

using Scalar = float;

// Function to be minimized
struct Model : public moptimizer::BaseModel<Scalar, Model> {
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
  SimpleModel() : optimizer(2) {
    cost = new moptimizer::CostFunctionNumerical<Scalar, 2, 1>(
        Model::Ptr(new Model(x_data, y_data)), 7);

    // auto loss = new duna::loss::GemmanMCClure<Scalar>(100.0);
    cost->setLossFunction(moptimizer::loss::GemmanMCClure<Scalar>::Ptr(
        new moptimizer::loss::GemmanMCClure<Scalar>(100.0)));
    optimizer.addCost(cost);
  }

  ~SimpleModel() { delete cost; }

 protected:
  moptimizer::LevenbergMarquadtDynamic<Scalar> optimizer;
  moptimizer::CostFunctionNumerical<Scalar, 2, 1> *cost;
  Scalar x_data[7] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
  Scalar y_data[7] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};
};

TEST_F(SimpleModel, InitialCondition00) {
  Scalar x0[] = {0.9, 0.2};

  optimizer.minimize(x0);

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);
}

TEST_F(SimpleModel, InitialCondition10) {
  Scalar x0[] = {1.9, 1.5};
  optimizer.minimize(x0);

  EXPECT_NEAR(x0[0], 0.362, 0.01);
  EXPECT_NEAR(x0[1], 0.556, 0.01);
}