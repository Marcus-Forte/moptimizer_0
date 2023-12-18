#pragma once

#include <duna_optimizer/cost_function.h>
#include <duna_optimizer/linearization.h>
#include <duna_optimizer/logger.h>
#include <duna_optimizer/model.h>

namespace duna_optimizer {

/// @brief Numerical Differentiation cost function. Computes numerical
/// derivatives when computing hessian.
/// @tparam Scalar Scalar type (double, float)
/// @tparam model_parameter_dim Dimension of parameters
/// @tparam model_output_dim Dimension of output
template <class Scalar = double, int model_parameter_dim = 1, int model_output_dim = 1>
class CostFunctionNumerical : public CostFunctionBase<Scalar> {
 public:
  using typename CostFunctionBase<Scalar>::Model;
  using typename CostFunctionBase<Scalar>::ModelPtr;

  CostFunctionNumerical(ModelPtr model, int num_residuals)
      : CostFunctionBase<Scalar>(model, num_residuals) {
    covariance_->resize(model_output_dim, model_output_dim);
    covariance_->setIdentity();
  }
  CostFunctionNumerical(const CostFunctionNumerical &) = delete;
  CostFunctionNumerical &operator=(const CostFunctionNumerical &) = delete;
  virtual ~CostFunctionNumerical() = default;

  Scalar computeCost(const Scalar *x) override {
    CostComputation<Scalar, model_parameter_dim, model_output_dim> compute;
    return compute.parallelComputeCost(x, model_, num_residuals_);
  }

  virtual Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override {
    CostComputation<Scalar, model_parameter_dim, model_output_dim> compute;
    return compute.computeHessianNumerical(x, covariance_->data(), loss_function_, hessian, b,
                                           model_, num_residuals_);
  }

 protected:
  using CostFunctionBase<Scalar>::num_residuals_;
  using CostFunctionBase<Scalar>::model_;
  using CostFunctionBase<Scalar>::loss_function_;
  using CostFunctionBase<Scalar>::covariance_;
};
}  // namespace duna_optimizer