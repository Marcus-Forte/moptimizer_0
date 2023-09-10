#include <duna_optimizer/cost_function_analytical_dyn.h>

namespace duna_optimizer {

template <class Scalar>
CostFunctionAnalyticalDynamic<Scalar>::CostFunctionAnalyticalDynamic(ModelPtr model,
                                                                     int num_parameters,
                                                                     int num_outputs,
                                                                     int num_residuals)
    : CostFunctionAnalytical<Scalar>(model, num_residuals),
      num_parameters_(num_parameters),
      num_outputs_(num_outputs) {
  jacobian_.resize(num_outputs_, num_parameters_);
  residuals_.resize(num_outputs_);
  covariance_.reset(new covariance::IdentityCovariance<Scalar>(num_parameters_));
}

template <class Scalar>
CostFunctionAnalyticalDynamic<Scalar>::CostFunctionAnalyticalDynamic(ModelPtr model,
                                                                     int num_parameters,
                                                                     int num_outputs)
    : CostFunctionAnalytical<Scalar>(model),
      num_parameters_(num_parameters),
      num_outputs_(num_outputs) {
  jacobian_.resize(num_outputs_, num_parameters_);
  residuals_.resize(num_outputs_);
  covariance_.reset(new covariance::IdentityCovariance<Scalar>(num_parameters_));
}

template <class Scalar>
void CostFunctionAnalyticalDynamic<Scalar>::init(const Scalar *x, Scalar *hessian, Scalar *b) {
  new (&x_map_) Eigen::Map<const ParameterVector>(x, num_parameters_);
  new (&hessian_map_) Eigen::Map<HessianMatrix>(hessian, num_parameters_, num_parameters_);
  new (&b_map_) Eigen::Map<ParameterVector>(b, num_parameters_);

  hessian_map_.setZero();
  b_map_.setZero();
}

template class CostFunctionAnalyticalDynamic<float>;
template class CostFunctionAnalyticalDynamic<double>;
}  // namespace duna_optimizer