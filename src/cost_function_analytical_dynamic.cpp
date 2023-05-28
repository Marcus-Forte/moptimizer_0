#include <duna/cost_function_analytical_dynamic.h>

namespace duna {
template <class Scalar>
void CostFunctionAnalyticalDynamic<Scalar>::init(const Scalar *x,
                                                 Scalar *hessian, Scalar *b) {
  new (&x_map_) Eigen::Map<const ParameterVector>(x, num_parameters_);
  new (&hessian_map_)
      Eigen::Map<HessianMatrix>(hessian, num_parameters_, num_parameters_);
  new (&b_map_) Eigen::Map<ParameterVector>(b, num_parameters_);

  hessian_map_.setZero();
  b_map_.setZero();

  jacobian_.resize(num_outputs_, num_parameters_);
  residuals_.resize(num_outputs_);
  covariance_.reset(
      new covariance::IdentityCovariance<Scalar>(num_parameters_));
}

template class CostFunctionAnalyticalDynamic<float>;
template class CostFunctionAnalyticalDynamic<double>;
}  // namespace duna