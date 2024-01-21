#pragma once

#include "loss_function.h"

namespace moptimizer::loss {
template <typename T>
class GemmanMCClure : public ILossFunction<T> {
 public:
  GemmanMCClure(T threshold) : threshold_(threshold) {}

  inline T weight(T errorSquaredNorm) override {
    return (square(threshold_) / square(errorSquaredNorm + threshold_));
  }

 private:
  T threshold_;

  inline T square(T val) { return val * val; }
};
}  // namespace moptimizer::loss