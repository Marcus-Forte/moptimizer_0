#include <moptimizer/cost_function_analytical_dyn.h>

#include "moptimizer/linearization.h"
namespace moptimizer {

template <class Scalar>
CostFunctionAnalyticalDynamic<Scalar>::CostFunctionAnalyticalDynamic(ModelPtr model,
                                                                     int num_parameters,
                                                                     int num_outputs,
                                                                     int num_residuals)
    : CostFunctionBase<Scalar>(model, num_residuals),
      num_parameters_(num_parameters),
      num_outputs_(num_outputs) {
  covariance_->resize(num_outputs_, num_outputs_);
  covariance_->setIdentity();
}

template <class Scalar>
Scalar CostFunctionAnalyticalDynamic<Scalar>::computeCost(const Scalar *x) {
  CostComputation<Scalar> compute(num_parameters_, num_outputs_);
  return compute.parallelComputeCost(x, model_, num_residuals_);
}

template <class Scalar>
Scalar CostFunctionAnalyticalDynamic<Scalar>::linearize(const Scalar *x, Scalar *hessian,
                                                        Scalar *b) {
  CostComputation<Scalar> compute(num_parameters_, num_outputs_);
  return compute.computeHessian(x, covariance_->data(), loss_function_, hessian, b, model_,
                                num_residuals_);
}

template class CostFunctionAnalyticalDynamic<float>;
template class CostFunctionAnalyticalDynamic<double>;
}  // namespace moptimizer