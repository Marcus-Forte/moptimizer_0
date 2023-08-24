#pragma once

#include <duna_optimizer/model.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Dense>

/// Common cost function methods.

namespace duna_optimizer {

/// @brief perform compuation of cost. Single threaded.
/// @tparam Scalar
/// @tparam ResidualsType
/// @param x parameter
/// @param residuals residuals object. Type of object that receives f(x).
/// @param model model.
/// @param num_elements number of elements to iterate.
/// @return
template <typename Scalar, class ResidualsType>
Scalar performComputeCost(const Scalar *const x, ResidualsType &residuals,
                          typename IBaseModel<Scalar>::Ptr model, int num_elements) {
  model->setup(x);
  Scalar sum = 0.0;

  for (int i = 0; i < num_elements; ++i) {
    if (model->f(x, residuals.data(), i)) {
      sum += residuals.transpose() * residuals;
    }
  }

  return sum;
}

template <typename Scalar, class ResidualsType>
Scalar performParallelComputeCost(const Scalar *const x, ResidualsType &residuals,
                                  typename IBaseModel<Scalar>::Ptr model, int num_elements) {
  model->setup(x);

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, num_elements), 0.0f,
      [&](tbb::blocked_range<int> &r, Scalar init) -> Scalar {
        for (auto it = r.begin(); it != r.end(); ++it) {
          if (model->f(x, residuals.data(), it)) {
            init += residuals.transpose() * residuals;
          }
        }
        return init;
      },
      std::plus<Scalar>());
}

}  // namespace duna_optimizer