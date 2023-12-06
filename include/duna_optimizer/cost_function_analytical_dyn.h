#pragma once

#include <duna_optimizer/cost_function.h>
#include <duna_optimizer/logger.h>
#include <duna_optimizer/model.h>

namespace duna_optimizer {
/* Analytical cost function module. Computes hessian using provided `f_df` model
 * function with explicit jacobian calculation. Dynamic size parameters and
 * function outputs. */
template <class Scalar = double>
class CostFunctionAnalyticalDynamic : public CostFunctionBase<Scalar> {
 public:
  using typename CostFunctionBase<Scalar>::Model;
  using typename CostFunctionBase<Scalar>::ModelPtr;

  CostFunctionAnalyticalDynamic(ModelPtr model, int num_parameters, int num_outputs,
                                int num_residuals);

  virtual ~CostFunctionAnalyticalDynamic() = default;

  Scalar computeCost(const Scalar *x) override;
  Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override;

 protected:
  using CostFunctionBase<Scalar>::num_residuals_;
  using CostFunctionBase<Scalar>::model_;
  using CostFunctionBase<Scalar>::loss_function_;
  using CostFunctionBase<Scalar>::covariance_;
  int num_parameters_;
  int num_outputs_;
};
}  // namespace duna_optimizer