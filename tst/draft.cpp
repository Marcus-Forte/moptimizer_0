#include <gtest/gtest.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Dense>
#include <memory>

/// General sandbox program.

int main(int argc, char** argv) {
  Eigen::Matrix<double, -1, 1> some_vec;
  int n = 10;
  some_vec.resize(n);
  double init = 0.0;
  std::for_each(some_vec.begin(), some_vec.end(), [&init](auto& el) { el = init++; });

  Eigen::Matrix<double, -1, 1> some_vec_copy(some_vec);

  some_vec[1] = 999;

  std::cout << some_vec;

  std::cout << some_vec_copy;
}