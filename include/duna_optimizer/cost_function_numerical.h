#pragma once

#include <duna_optimizer/cost_function.h>
#include <duna_optimizer/logger.h>
#include <duna_optimizer/model.h>

#include <vector>

namespace duna_optimizer {
/* Numerical Differentiation cost function module. Computes numerical
 * derivatives when computing hessian. */
template <class Scalar = double, int model_parameter_dim = Eigen::Dynamic,
          int model_output_dim = Eigen::Dynamic>
class CostFunctionNumerical : public CostFunctionBase<Scalar> {
 public:
  using ParameterVector = Eigen::Matrix<Scalar, model_parameter_dim, 1>;
  using HessianMatrix = Eigen::Matrix<Scalar, model_parameter_dim, model_parameter_dim>;
  using JacobianMatrix =
      Eigen::Matrix<Scalar, model_output_dim, model_parameter_dim, Eigen::RowMajor>;
  using ResidualVector = Eigen::Matrix<Scalar, model_output_dim, 1>;
  using typename CostFunctionBase<Scalar>::Model;
  using typename CostFunctionBase<Scalar>::ModelPtr;

  CostFunctionNumerical(ModelPtr model, int num_residuals)
      : CostFunctionBase<Scalar>(model, num_residuals),
        hessian_map_(0, 0, 0),
        x_map_(0, 0, 0),
        b_map_(0, 0, 0) {}
  CostFunctionNumerical(ModelPtr model)
      : CostFunctionBase<Scalar>(model, 1),
        hessian_map_(0, 0, 0),
        x_map_(0, 0, 0),
        b_map_(0, 0, 0) {}
  CostFunctionNumerical(const CostFunctionNumerical &) = delete;
  CostFunctionNumerical &operator=(const CostFunctionNumerical &) = delete;

  Scalar computeCost(const Scalar *x) override {
    Scalar sum = 0;

    model_->setup(x);

    for (int i = 0; i < m_num_residuals; ++i) {
      if (model_->f(x, residuals_.data(), i)) {
        sum += residuals_.transpose() * residuals_;
      }
    }
    return sum;
  }

  virtual Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override {
    init(x, hessian, b);

    Scalar sum = 0.0;

    const Scalar min_step_size = std::sqrt(std::numeric_limits<Scalar>::epsilon());

    // Step size
    std::vector<Scalar> h(x_map_.size());
    std::vector<ParameterVector> x_plus(x_map_.size(), x_map_);
    std::vector<ModelPtr> models_plus_(x_map_.size());
    for (int j = 0; j < x_map_.size(); ++j) {
      h[j] = min_step_size * abs(x_map_[j]);

      if (h[j] == 0.0) h[j] = min_step_size;

      x_plus[j][j] += h[j];

      models_plus_[j] = model_->clone();
      models_plus_[j]->setup((x_plus[j]).data());
    }

    model_->setup(x_map_.data());
    // TODO check if at least a few residuals were computed.
    // TODO paralelize.
    for (int i = 0; i < m_num_residuals; ++i) {
      if (model_->f(x, residuals_.data(), i)) {
        for (int j = 0; j < x_map_.size(); ++j) {
          models_plus_[j]->f(x_plus[j].data(), residuals_plus_.data(), i);
          jacobian_.col(j) = (residuals_plus_ - residuals_) / h[j];
        }

        Scalar w = loss_function_->weight(residuals_.squaredNorm());
        // auto covariance = covariance_->getCovariance();  // TODO
        // causing double free! hessian_map.template
        // selfadjointView<Eigen::Lower>().rankUpdate(jacobian_.transpose());
        // // H = J^T * J
        hessian_map_.noalias() += jacobian_.transpose() * w * jacobian_;
        b_map_.noalias() += jacobian_.transpose() * w * residuals_;
        sum += residuals_.transpose() * residuals_;
      }
    }
    // hessian_map_.template triangularView<Eigen::Upper>() =
    // hessian_map_.transpose();
    std::cout << "hessian_map_:\n " << hessian_map_ << std::endl;
    return sum;
  }

 protected:
  using CostFunctionBase<Scalar>::m_num_residuals;
  using CostFunctionBase<Scalar>::model_;
  using CostFunctionBase<Scalar>::loss_function_;
  using CostFunctionBase<Scalar>::covariance_;
  Eigen::Map<const ParameterVector> x_map_;
  Eigen::Map<HessianMatrix> hessian_map_;
  Eigen::Map<ParameterVector> b_map_;

  JacobianMatrix jacobian_;
  ResidualVector residuals_;
  ResidualVector residuals_plus_;

  // Initialize internal cost function states.
  virtual void init(const Scalar *x, Scalar *hessian, Scalar *b) override {
    new (&x_map_) Eigen::Map<const ParameterVector>(x, model_parameter_dim, 1);
    new (&hessian_map_)
        Eigen::Map<HessianMatrix>(hessian, model_parameter_dim, model_parameter_dim);
    new (&b_map_) Eigen::Map<ParameterVector>(b, model_parameter_dim, 1);

    hessian_map_.setZero();
    b_map_.setZero();
    covariance_.reset(new covariance::IdentityCovariance<Scalar>(model_parameter_dim));
  }
};
}  // namespace duna_optimizer