
#include <duna_optimizer/logger.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
/// General sandbox program.

int main(int argc, char** argv) {
  Eigen::Matrix<double, 2, 1> jac_transpose;
  Eigen::Matrix<double, -1, -1> cov;
  Eigen::Matrix<double, 1, 2> jac;

  cov.resize(2, 2);

  auto res = jac_transpose * cov;
  std::cout << res << std::endl;
}