
#include <duna_optimizer/logger.h>

#include <Eigen/Dense>
#include <duna_optimizer/stopwatch.hpp>
#include <fstream>
#include <iostream>
/// General sandbox program.

void fun(const Eigen::Ref<Eigen::Matrix<double, 3, 3>>& m) { std::cout << m << std::endl; }

int main(int argc, char** argv) {
  int size = 50;
  // VectorXf is a vector of floats, with dynamic size.
  Eigen::VectorXf u(size), v(size), w(size);
  u = v + w;
  // Eigen::Matrix<double, -1, -1> matrix;
  // Eigen::Matrix<double, -1, -1> matrixb;
  // int dim = 1000;

  // matrix.setRandom(dim, dim);
  // matrixb.setRandom(dim, dim);
  // utilities::Stopwatch watch;
  // watch.enable();
  // watch.tick();
  // auto res = matrix * matrixb;
  // watch.tock("multiplication\n");

  // std::cout << "Maaping mult...\n";
  // Eigen::Map<Eigen::Matrix<double, -1, -1>> mapA(matrix.data(), dim, dim);
  // Eigen::Map<Eigen::Matrix<double, -1, -1>> mapB(matrixb.data(), dim, dim);
  // Eigen::Matrix<double, -1, -1> resmap;
  // resmap = mapA * mapB;

  // std::cout << "Done\n";
  // std::cout << resmap(50,50) << std::endl;

  // std::cout << res(50,50);
}