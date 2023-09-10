#pragma once

#include <duna_optimizer/cost_function.h>
#include <duna_optimizer/cost_function_common.h>
#include <duna_optimizer/logger.h>
#include <duna_optimizer/model.h>

namespace duna_optimizer {

/// @brief Analytical cost function. Computes hessian using provided `f_df` model function with
/// explicit jacobian calculation.
/// @tparam Scalar Scalar type (double, float)
/// @tparam model_parameter_dim Dimension of parameters
/// @tparam model_output_dim Dimension of output
template <class Scalar = double, int model_parameter_dim = Eigen::Dynamic,
          int model_output_dim = Eigen::Dynamic>
class CostFunctionAnalytical : public CostFunctionBase<Scalar> {
 public:
  using ParameterVector = Eigen::Matrix<Scalar, model_parameter_dim, 1>;
  using HessianMatrix = Eigen::Matrix<Scalar, model_parameter_dim, model_parameter_dim>;
  using JacobianMatrix =
      Eigen::Matrix<Scalar, model_output_dim, model_parameter_dim, Eigen::RowMajor>;
  using ResidualVector = Eigen::Matrix<Scalar, model_output_dim, 1>;
  using typename CostFunctionBase<Scalar>::Model;
  using typename CostFunctionBase<Scalar>::ModelPtr;

  CostFunctionAnalytical(ModelPtr model, int num_residuals)
      : CostFunctionBase<Scalar>(model, num_residuals),
        hessian_map_(0, 0, 0),
        x_map_(0, 0),
        b_map_(0, 0) {}
  CostFunctionAnalytical(ModelPtr model)
      : CostFunctionBase<Scalar>(model, 1), hessian_map_(0, 0, 0), x_map_(0, 0), b_map_(0, 0) {}

  CostFunctionAnalytical(const CostFunctionAnalytical &) = delete;
  CostFunctionAnalytical &operator=(const CostFunctionAnalytical &) = delete;

  Scalar computeCost(const Scalar *x) override {
    return performParallelComputeCost(x, residuals_, model_, num_residuals_);
  }

  Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override {
    init(x, hessian, b);

    Scalar sum = 0.0;

    model_->setup(x);

    // TODO check if at least a few residuals were computed.
    // TODO paralelize.
    for (int i = 0; i < num_residuals_; ++i) {
      if (model_->f_df(x, residuals_.data(), jacobian_.data(), i)) {
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
    // std::cout << "hessian_map_:\n " << hessian_map_ << std::endl;
    // hessian_map_.template triangularView<Eigen::Upper>() = hessian_map_.transpose();
    return sum;
  }

 protected:
  using CostFunctionBase<Scalar>::num_residuals_;
  using CostFunctionBase<Scalar>::model_;
  using CostFunctionBase<Scalar>::loss_function_;
  using CostFunctionBase<Scalar>::covariance_;
  Eigen::Map<const ParameterVector> x_map_;
  Eigen::Map<HessianMatrix> hessian_map_;
  Eigen::Map<ParameterVector> b_map_;

  JacobianMatrix jacobian_;
  ResidualVector residuals_;

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