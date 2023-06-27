#include <duna_optimizer/cost_function_numerical_dynamic.h>

namespace duna_optimizer {
template <class Scalar>
void CostFunctionNumericalDynamic<Scalar>::init(const Scalar *x, Scalar *hessian, Scalar *b) {
  new (&x_map_) Eigen::Map<const ParameterVector>(x, num_parameters_);
  new (&hessian_map_) Eigen::Map<HessianMatrix>(hessian, num_parameters_, num_parameters_);
  new (&b_map_) Eigen::Map<ParameterVector>(b, num_parameters_);

  hessian_map_.setZero();
  b_map_.setZero();

  jacobian_.resize(num_outputs_, num_parameters_);
  residuals_.resize(num_outputs_);
  residuals_plus_.resize(num_outputs_);
  covariance_.reset(new covariance::IdentityCovariance<Scalar>(num_parameters_));
}

template class CostFunctionNumericalDynamic<float>;
template class CostFunctionNumericalDynamic<double>;
}  // namespace duna_optimizer