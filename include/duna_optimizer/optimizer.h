#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <duna_optimizer/cost_function.h>
#include <duna_optimizer/logger.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace duna_optimizer {
template <class Scalar = double>
class Optimizer {
 public:
  using Ptr = std::shared_ptr<Optimizer>;
  using ConstPtr = std::shared_ptr<const Optimizer>;
  using CostFunctionType = CostFunctionBase<Scalar>;

  Optimizer() : maximum_iterations_(15) {}
  Optimizer(const Optimizer &) = delete;
  Optimizer &operator=(const Optimizer &) = delete;
  virtual ~Optimizer() = default;

  bool isCostSmall(Scalar cost_sum) {
    if (std::abs(cost_sum) < 8 * (std::numeric_limits<Scalar>::epsilon())) return true;
    return false;
  }

  /// @brief set maximum allowed iterations.
  /// @param max_iterations
  inline void setMaximumIterations(int max_iterations) {
    if (max_iterations < 0)
      throw std::invalid_argument("Optimization::max_iterations cannot be less than 0.");
    maximum_iterations_ = max_iterations;
  }
  /// @brief get maximum allowed iterations.
  /// @return
  inline unsigned int getMaximumIterations() const { return maximum_iterations_; }

  /// @brief get executed iterations.
  /// @return
  inline unsigned int getExecutedIterations() const { return executed_iterations_; }

  /// @brief Chdeck cost objects.
  /// @return
  inline bool checkCosts() const {
    if (costs_.size() == 0) {
      std::cerr << "No cost function added!\n";
      throw std::runtime_error("No cost function added!");
    }
    return true;
  }

  /// @brief Add cost function to optimization problem.
  /// @param cost
  inline void addCost(CostFunctionType *cost) { costs_.push_back(cost); }

  /// @brief Clear costs from list. Optionally delete them from memory.
  /// @param delete_costs
  inline void clearCosts(bool delete_costs = false) {
    if (delete_costs) {
      for (int i = 0; i < costs_.size(); ++i) delete costs_[i];
    }

    costs_.clear();
  }
  /// @brief Init optimization parameters.
  /// @param x0
  virtual void init(Scalar *x0) = 0;

  /// @brief Perform optimization step.
  /// @param x0
  /// @return
  virtual OptimizationStatus step(Scalar *x0) = 0;

  /// @brief Perform optimization.
  /// @param x0
  /// @return
  virtual OptimizationStatus minimize(Scalar *x0) = 0;

 protected:
  virtual bool hasConverged() = 0;
  std::vector<CostFunctionType *> costs_;
  unsigned int maximum_iterations_;
  unsigned int executed_iterations_;
};
}  // namespace duna_optimizer

#endif