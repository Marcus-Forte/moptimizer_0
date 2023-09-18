
#include <duna_optimizer/logger.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
/// General sandbox program.

int main(int argc, char** argv) {
  duna::Logger logger(std::cout, duna::Logger::L_INFO, "SCREEN");

  std::ofstream file("FILE.LOG");
  logger.addSink(&file);
  logger.log(duna::Logger::L_INFO, "HEY!!");
}