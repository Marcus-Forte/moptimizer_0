#include <gtest/gtest.h>

#include "duna_optimizer/cost_function_analytical.h"
#include "duna_optimizer/cost_function_analytical_dyn.h"
#include "duna_optimizer/cost_function_numerical_dyn.h"
#include "duna_optimizer/levenberg_marquadt_dyn.h"
#include "duna_optimizer/model.h"
#include "duna_optimizer/so3.h"

using Scalar = double;
using StateVector = Eigen::Matrix<double, 12, 1>;       // R15 component
using StateRotation = Eigen::Matrix<double, 3, 3>;      // SO3 component
using StateDeltaVector = Eigen::Matrix<double, 15, 1>;  // Delta component

struct state_composition {
  state_composition(const double *x) {
    rot_ = so3::Exp(Eigen::Vector3d(x[0], x[1], x[2]));
    for (int i = 0; i < 12; ++i) {
      lin_[i] = x[i + 3];
    }
  }
  state_composition() {
    lin_.setZero();
    rot_.setIdentity();
  }

  // Delta operation
  void Plus(const StateDeltaVector &delta) {
    const Eigen::Vector3d ang_delta(delta[0], delta[1], delta[2]);

    const StateRotation rhs_rot = so3::Exp(ang_delta);
    this->rot_ = this->rot_ * rhs_rot;
    this->lin_ += delta.block<12, 1>(3, 0);
  }

  // Fullstate operation
  StateDeltaVector Minus(const state_composition &rhs) {
    StateDeltaVector res;
    res.block<12, 1>(3, 0) = this->lin_ - rhs.lin_;
    StateRotation transposes = rhs.rot_.transpose() * this->rot_;
    Eigen::Matrix<double, 3, 1> delta_angle;
    so3::Log<double>(transposes, delta_angle);
    res.block<3, 1>(0, 0) = delta_angle;

    return res;
  }

  StateVector lin_;
  StateRotation rot_;
};

class StateModel : public duna_optimizer::BaseModelJacobian<Scalar, StateModel> {
 public:
  StateModel() = default;
  StateModel(const state_composition &init_state) : init_state_(init_state) {}
  void setup(const Scalar *x) override {}

  bool f(const Scalar *x, Scalar *f_x, unsigned int index) const override {
    // f(x) = x_k - x_k0
    state_composition x_k(x);
    Eigen::Map<StateDeltaVector> f_x_(f_x);

    f_x_ = x_k.Minus(init_state_);

    std::cout << "fx = " << f_x_ << std::endl;

    return true;
  }
  bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) const override {
    // f(x) = x_k - x_k0
    state_composition x_k(x);
    Eigen::Map<StateDeltaVector> f_x_(f_x);

    f_x_ = x_k.Minus(init_state_);

    return true;
  }

 private:
  state_composition init_state_;
};

TEST(StateModel, Optimize) {
  state_composition init;
  init.rot_ = so3::Exp(Eigen::Vector3d(0.1, 0.2, 0.3));

  double x_init[15] = {0.6, 0.8, 0.3, -0.4, 0.11, -0.9};
  double x[15] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  double f_x[15];

  StateModel::Ptr model(new StateModel(x_init));

  // model->f_df(x, f_x, 0, 0);

  std::cout << Eigen::Map<StateDeltaVector>(x) << std::endl;

  duna_optimizer::LevenbergMarquadtDynamic<Scalar> lm(15);
  duna_optimizer::CostFunctionNumericalDynamic<Scalar> cost(model, 15, 15, 1);

  // cost.setCovariance()
  lm.setLogger(std::make_shared<duna::Logger>(std::cout, duna::Logger::L_DEBUG));

  Eigen::Matrix<double, 15, 15> hessian;
  Eigen::Matrix<double, 15, 1> b;

  // cost.linearize(x,hessian.data(), b.data());
  lm.addCost(&cost);
  lm.minimize(x);
  lm.setMaximumIterations(10);

  std::cout << Eigen::Map<StateDeltaVector>(x) << std::endl;
}