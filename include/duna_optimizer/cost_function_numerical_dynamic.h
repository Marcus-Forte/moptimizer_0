#pragma once

#include <duna_optimizer/cost_function_numerical.h>

namespace duna_optimizer {
/// @brief Numerical Differentiation cost function. Computes numerical
/// derivatives when computing hessian. Dimensions of function output and parameter are set in
/// runtime.
/// @tparam Scalar Scalar type (double, float)
template <class Scalar = double>
class CostFunctionNumericalDynamic : public CostFunctionNumerical<Scalar> {
 public:
  using typename CostFunctionNumerical<Scalar>::ParameterVector;
  using typename CostFunctionNumerical<Scalar>::HessianMatrix;
  using typename CostFunctionNumerical<Scalar>::JacobianMatrix;
  using typename CostFunctionNumerical<Scalar>::Model;
  using typename CostFunctionNumerical<Scalar>::ModelPtr;

  CostFunctionNumericalDynamic(ModelPtr model, int num_parameters, int num_outputs,
                               int num_residuals)
      : CostFunctionNumerical<Scalar>(model, num_residuals),
        num_parameters_(num_parameters),
        num_outputs_(num_outputs) {}

  CostFunctionNumericalDynamic(ModelPtr model, int num_parameters, int num_outputs)
      : CostFunctionNumerical<Scalar>(model),
        num_parameters_(num_parameters),
        num_outputs_(num_outputs) {}

  virtual ~CostFunctionNumericalDynamic() {}

 private:
  using CostFunctionNumerical<Scalar>::x_map_;
  using CostFunctionNumerical<Scalar>::hessian_map_;
  using CostFunctionNumerical<Scalar>::b_map_;
  using CostFunctionNumerical<Scalar>::jacobian_;
  using CostFunctionNumerical<Scalar>::residuals_;
  using CostFunctionNumerical<Scalar>::residuals_plus_;
  using CostFunctionNumerical<Scalar>::covariance_;
  int num_parameters_;
  int num_outputs_;

  void init(const Scalar *x, Scalar *hessian, Scalar *b) override;
};
}  // namespace duna_optimizer