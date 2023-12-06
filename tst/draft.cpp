
#include <duna_optimizer/logger.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
/// General sandbox program.

void fun(const Eigen::Ref<Eigen::Matrix<double, 3,3>>& m) {
  std::cout << m << std::endl;
}

int main(int argc, char** argv) {
  Eigen::Matrix<double, 3, 3> jac_transpose;

  jac_transpose = Eigen::Matrix<double,3,3>::Random(2,2);

  Eigen::Map<Eigen::Matrix<double, 3,3>> some_map(nullptr);
  
  new(&some_map) Eigen::Map<Eigen::Matrix<double,3,3>>(jac_transpose.data(), 3,3);
  some_map(1,1) = 100;

  fun(jac_transpose);
}