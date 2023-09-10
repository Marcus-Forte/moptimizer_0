#include <duna_optimizer/cost_function_numerical.h>
#include <duna_optimizer/cost_function_numerical_dyn.h>
#include <duna_optimizer/levenberg_marquadt.h>
#include <duna_optimizer/levenberg_marquadt_dyn.h>
#include <gtest/gtest.h>

#include <duna_optimizer/stopwatch.hpp>

//  Powell's singular function.
//
//   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
//
//   f1 = x1 + 10*x2;
//   f2 = sqrt(5) * (x3 - x4)
//   f3 = (x2 - 2*x3)^2
//   f4 = sqrt(10) * (x1 - x4)^2

// The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
// The minimum is 0 at (x1, x2, x3, x4) = 0.

struct PowellModel : public duna_optimizer::BaseModelJacobian<double, PowellModel> {
  bool f(const double *x, double *f_x, unsigned int index) const override {
    f_x[0] = x[0] + 10 * x[1];
    f_x[1] = sqrt(5) * (x[2] - x[3]);
    f_x[2] = (x[1] - 2 * x[2]) * (x[1] - 2 * x[2]);
    f_x[3] = sqrt(10) * (x[0] - x[3]) * (x[0] - x[3]);
    return true;
  }

  /* ROW MAJOR*/
  bool f_df(const double *x, double *f_x, double *jacobian, unsigned int index) const override {
    this->f(x, f_x, index);

    // Df / dx0
    jacobian[0] = 1;
    jacobian[4] = 0;
    jacobian[8] = 0;
    jacobian[12] = sqrt(10) * 2 * (x[0] - x[3]);

    // Df / dx1
    jacobian[1] = 10;
    jacobian[5] = 0;
    jacobian[9] = 2 * (x[1] + 2 * x[2]);
    jacobian[13] = 0;

    // Df / dx2
    jacobian[2] = 0;
    jacobian[6] = sqrt(5);
    jacobian[10] = 2 * (x[1] + 2 * x[2]) * (-2);
    jacobian[14] = 0;

    // Df / dx3
    jacobian[3] = 0;
    jacobian[7] = -sqrt(5);
    jacobian[11] = 0;
    jacobian[15] = sqrt(10) * 2 * (x[0] - x[3]) * (-1);

    return true;
  }
};

TEST(PowellFunction, InitialCondition0) {
  //
  utilities::Stopwatch timer;
  timer.tick();
  double x0[] = {3, -1, 0, 4};

  duna_optimizer::logger::setGlobalVerbosityLevel(duna_optimizer::L_DEBUG);

  duna_optimizer::LevenbergMarquadt<double, 4> optimizer;
  optimizer.setMaximumIterations(25);

  optimizer.addCost(new duna_optimizer::CostFunctionNumerical<double, 4, 4>(
      PowellModel::Ptr(new PowellModel), 1));

  optimizer.minimize(x0);

  timer.tock("Power Function minimzation");

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(x0[i], 0.0, 5e-5);
  }
}

TEST(PowellFunction, InitialCondition0Dynamic) {
  //
  utilities::Stopwatch timer;
  timer.tick();
  double x0[] = {3, -1, 0, 4};

  duna_optimizer::logger::setGlobalVerbosityLevel(duna_optimizer::L_DEBUG);

  duna_optimizer::LevenbergMarquadtDynamic<double> optimizer(4);
  optimizer.setMaximumIterations(25);

  optimizer.addCost(new duna_optimizer::CostFunctionNumericalDynamic<double>(
      PowellModel::Ptr(new PowellModel), 4, 4, 1));

  optimizer.minimize(x0);

  timer.tock("Power Function minimzation");

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(x0[i], 0.0, 5e-5);
  }
}