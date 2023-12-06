#include "duna_optimizer/linearization.h"

namespace duna_optimizer {
template class CostComputation<float, Eigen::Dynamic, Eigen::Dynamic>;
template class CostComputation<double, Eigen::Dynamic, Eigen::Dynamic>;
}  // namespace duna_optimizer