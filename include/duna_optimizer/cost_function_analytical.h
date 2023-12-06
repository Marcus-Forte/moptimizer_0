#pragma once

#include <duna_optimizer/cost_function.h>
#include <duna_optimizer/linearization.h>
#include <duna_optimizer/logger.h>
#include <duna_optimizer/model.h>

namespace duna_optimizer {

/// @brief Analytical cost function. Computes hessian using provided `f_df` model function with
/// explicit jacobian calculation.
/// @tparam Scalar Scalar type (double, float)
/// @tparam model_parameter_dim Dimension of parameters
/// @tparam model_output_dim Dimension of output
template <class Scalar = double, int model_parameter_dim = 1, int model_output_dim = 1>
class CostFunctionAnalytical : public CostFunctionBase<Scalar> {
 public:
  using typename CostFunctionBase<Scalar>::Model;
  using typename CostFunctionBase<Scalar>::ModelPtr;

  CostFunctionAnalytical(ModelPtr model, int num_residuals)
      : CostFunctionBase<Scalar>(model, num_residuals) {
    covariance_->resize(model_output_dim, model_output_dim);
    covariance_->setIdentity();
  }
  CostFunctionAnalytical(const CostFunctionAnalytical &) = delete;
  CostFunctionAnalytical &operator=(const CostFunctionAnalytical &) = delete;
  virtual ~CostFunctionAnalytical() = default;

  Scalar computeCost(const Scalar *x) override {
    CostComputation<Scalar, model_parameter_dim, model_output_dim> compute;
    return compute.parallelComputeCost(x, model_, num_residuals_);
  }

  Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override {
    CostComputation<Scalar, model_parameter_dim, model_output_dim> compute;
    return compute.computeHessian(x, covariance_->data(), loss_function_, hessian, b, model_,
                                  num_residuals_);
  }

 protected:
  using CostFunctionBase<Scalar>::num_residuals_;
  using CostFunctionBase<Scalar>::model_;
  using CostFunctionBase<Scalar>::loss_function_;
  using CostFunctionBase<Scalar>::covariance_;
};
}  // namespace duna_optimizer