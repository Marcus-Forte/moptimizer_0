#ifndef TYPES_H
#define TYPES_H

namespace duna_optimizer {

enum OptimizationStatus {
  CONVERGED,
  MAXIMUM_ITERATIONS_REACHED,
  SMALL_DELTA,
  NUMERIC_ERROR,
  FATAL_ERROR,
};
}  // namespace duna_optimizer

#endif