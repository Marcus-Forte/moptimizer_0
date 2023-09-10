#include <duna_optimizer/levenberg_marquadt_dyn.h>

namespace duna_optimizer {

template <class Scalar>

void LevenbergMarquadtDynamic<Scalar>::init(Scalar* x0) {
  LevenbergMarquadt<Scalar, Eigen::Dynamic>::init(x0);

  logger::log_debug("[LM] Dynamic");
  new (&x0_map_) Eigen::Map<ParameterVector>(x0, num_parameters_);
  xi_.resize(num_parameters_);
  b_.resize(num_parameters_);
  hessian_diagonal_.resize(num_parameters_, num_parameters_);
  hessian_.resize(num_parameters_, num_parameters_);
  cost_hessian_.resize(num_parameters_, num_parameters_);
  cost_b_.resize(num_parameters_);
}

template class LevenbergMarquadtDynamic<float>;
template class LevenbergMarquadtDynamic<double>;

}  // namespace duna_optimizer