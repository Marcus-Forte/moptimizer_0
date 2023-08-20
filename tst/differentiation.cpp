#include <duna_optimizer/cost_function_analytical.h>
#include <duna_optimizer/cost_function_numerical.h>
#include <duna_optimizer/levenberg_marquadt.h>
#include <duna_optimizer/logger.h>
#include <duna_optimizer/model.h>
#include <duna_optimizer/models/accelerometer.h>
// #include <duna_optimizer/models/scan_matching.h>
#include <duna_optimizer/so3.h>
#include <gtest/gtest.h>

// extern duna::logger::s_level_;
duna_optimizer::VERBOSITY_LEVEL duna_optimizer::logger::s_level_;

/* We compare with numerical diff for resonable results.
It is very difficult that both yield the same results if something is wrong with
either Numerical or Analytical Diff */

template <typename Scalar = double>
struct SimpleModel : duna_optimizer::BaseModelJacobian<Scalar, SimpleModel<Scalar>> {
  SimpleModel(Scalar *x, Scalar *y) : data_x(x), data_y(y){};

  // Defining operator for comparison.
  bool f(const Scalar *x, Scalar *f_x, unsigned int index) const override {
    f_x[0] = data_y[index] - (x[0] * data_x[index]) / (x[1] + data_x[index]);
    return true;
  }

  // Jacobian
  bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) const override {
    Scalar denominator = (x[1] + data_x[index]);

    f_x[0] = data_y[index] - (x[0] * data_x[index]) / (x[1] + data_x[index]);

    // Row major
    jacobian[0] = -data_x[index] / denominator;
    jacobian[1] = (x[0] * data_x[index]) / (denominator * denominator);
    return true;
  }

 private:
  const Scalar *const data_x;
  const Scalar *const data_y;
};

template <typename Scalar>
class Differentiation : public ::testing::Test {};
using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Differentiation, ScalarTypes);

TYPED_TEST(Differentiation, SimpleModel) {
  TypeParam x_data[] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70, 5, 0};
  TypeParam y_data[] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317, 0.2, 0};
  int m_residuals = sizeof(x_data) / sizeof(TypeParam);  // 9
  // std::cout << "m_residuals = " << m_residuals;

  typename SimpleModel<TypeParam>::Ptr model(new SimpleModel<TypeParam>(x_data, y_data));

  duna_optimizer::CostFunctionAnalytical<TypeParam, 2, 1> cost_ana(model, m_residuals);
  duna_optimizer::CostFunctionNumerical<TypeParam, 2, 1> cost_num(model, m_residuals);

  Eigen::Matrix<TypeParam, 2, 2> Hessian;
  Eigen::Matrix<TypeParam, 2, 2> HessianNum;
  Eigen::Matrix<TypeParam, 2, 1> Residuals;
  Eigen::Matrix<TypeParam, 2, 1> x0(0.9, 0.2);

  auto sum_ana = cost_ana.computeCost(x0.data());
  auto sum_num = cost_num.computeCost(x0.data());

  EXPECT_NEAR(sum_ana, sum_num, 1e-4);

  cost_ana.linearize(x0.data(), Hessian.data(), Residuals.data());
  cost_num.linearize(x0.data(), HessianNum.data(), Residuals.data());

  for (int i = 0; i < Hessian.size(); ++i) {
    // May be close enough
    EXPECT_NEAR(Hessian(i), HessianNum(i), 5e-3);
  }
  std::cerr << "Hessian:\n" << Hessian << std::endl;
  std::cerr << "Hessian Numerical:\n" << HessianNum << std::endl;
}

struct Powell : duna_optimizer::BaseModelJacobian<double, Powell> {
  // Should be 4 x 4 = 16. Eigen stores column major order, so we fill indices
  // accordingly.

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

TEST(Differentiation, PowellModel) {
  const int m_residuals = 1;

  Powell::Ptr powell(new Powell);

  duna_optimizer::CostFunctionAnalytical<double, 4, 4> cost_ana(powell, m_residuals);
  duna_optimizer::CostFunctionNumerical<double, 4, 4> cost_num(powell, m_residuals);

  Eigen::Matrix<double, 4, 4> Hessian;
  Eigen::Matrix<double, 4, 4> HessianNum;
  Eigen::Matrix<double, 4, 1> x0(3, -1, 0, 4);
  Eigen::Matrix<double, 4, 1> residuals;

  auto sum_ana = cost_ana.computeCost(x0.data());
  auto sum_num = cost_num.computeCost(x0.data());

  EXPECT_NEAR(sum_ana, sum_num, 1e-4);

  cost_ana.linearize(x0.data(), Hessian.data(), residuals.data());
  cost_num.linearize(x0.data(), HessianNum.data(), residuals.data());

  for (int i = 0; i < Hessian.size(); ++i) {
    EXPECT_NEAR(Hessian(i), HessianNum(i), 1e-4);
  }

  std::cerr << "Hessian:\n" << Hessian << std::endl;
  std::cerr << "Hessian Numerical:\n" << HessianNum << std::endl;
}

TEST(Differentiation, Accelerometer) {
  double measurement[3];
  measurement[0] = 0;
  measurement[1] = 0;
  measurement[2] = 0;
  typename duna_optimizer::IBaseModel<double>::Ptr acc(
      new duna_optimizer::Accelerometer(measurement));
  double x[3];
  double f_x[3];
  x[0] = 0.1;
  x[1] = 0.0;
  x[2] = 0.0;

  duna_optimizer::CostFunctionNumerical<double, 3, 3> cost(acc);
  duna_optimizer::CostFunctionAnalytical<double, 3, 3> cost_a(acc);
  Eigen::Matrix3d hessian;
  Eigen::Matrix3d hessian_a;
  Eigen::Vector3d b;
  cost.linearize(x, hessian.data(), b.data());
  cost_a.linearize(x, hessian_a.data(), b.data());

  std::cout << "H = \n" << hessian << std::endl;
  std::cout << "H_a = \n" << hessian_a << std::endl;

  std::cout << "f(x) = " << f_x[0] << "," << f_x[1] << "," << f_x[2] << std::endl;
}
