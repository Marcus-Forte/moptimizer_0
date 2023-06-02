#pragma once

#include <duna/cost_function_analytical.h>

namespace duna {
/* Analytical cost function module. Computes hessian using provided `f_df` model
 * function with explicit jacobian calculation. Dynamic size parameters and
 * function outputs. */
template <class Scalar = double>
class CostFunctionAnalyticalDynamic : public CostFunctionAnalytical<Scalar> {
 public:
  using typename CostFunctionAnalytical<Scalar>::ParameterVector;
  using typename CostFunctionAnalytical<Scalar>::HessianMatrix;
  using typename CostFunctionAnalytical<Scalar>::JacobianMatrix;
  using typename CostFunctionAnalytical<Scalar>::Model;
  using typename CostFunctionAnalytical<Scalar>::ModelPtr;

  CostFunctionAnalyticalDynamic(ModelPtr model, int num_parameters, int num_outputs,
                                int num_residuals)
      : CostFunctionAnalytical<Scalar>(model, num_residuals),
        num_parameters_(num_parameters),
        num_outputs_(num_outputs) {}

  CostFunctionAnalyticalDynamic(ModelPtr model, int num_parameters, int num_outputs)
      : CostFunctionAnalytical<Scalar>(model),
        num_parameters_(num_parameters),
        num_outputs_(num_outputs) {}

  virtual ~CostFunctionAnalyticalDynamic() {}

 private:
  using CostFunctionAnalytical<Scalar>::x_map_;
  using CostFunctionAnalytical<Scalar>::hessian_map_;
  using CostFunctionAnalytical<Scalar>::b_map_;
  using CostFunctionAnalytical<Scalar>::jacobian_;
  using CostFunctionAnalytical<Scalar>::residuals_;
  using CostFunctionAnalytical<Scalar>::covariance_;
  int num_parameters_;
  int num_outputs_;

  void init(const Scalar *x, Scalar *hessian, Scalar *b) override;
};
}  // namespace duna