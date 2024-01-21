#include "moptimizer/linearization.h"

namespace moptimizer {
template class CostComputation<float, Eigen::Dynamic, Eigen::Dynamic>;
template class CostComputation<double, Eigen::Dynamic, Eigen::Dynamic>;
}  // namespace moptimizer