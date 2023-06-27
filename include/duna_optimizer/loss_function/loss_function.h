#pragma once

#include <memory>

namespace duna_optimizer::loss {
// Loss Function Interface
template <typename T>
class ILossFunction {
 public:
  using Ptr = std::shared_ptr<ILossFunction>;
  using ConstPtr = std::shared_ptr<const ILossFunction>;

  ILossFunction() = default;
  virtual ~ILossFunction() = default;

  virtual T weight(T errorSquaredNorm) = 0;
};

template <typename T>
class NoLoss : public ILossFunction<T> {
 public:
  inline virtual T weight(T errorSquaredNorm) override { return 1.0; }
};
}  // namespace duna_optimizer::loss