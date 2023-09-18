
#include <duna_optimizer/logger_.h>

#include <fstream>
#include <iostream>
/// General sandbox program.

int main(int argc, char** argv) {
  duna::Logger logger(std::cout);

  std::ofstream file("log.txt");
  duna::Logger file_logger(file);

  logger.log(duna::Logger::L_INFO, "INFO");
  logger.setLogLevel(duna::Logger::L_INFO);
  logger.log(duna::Logger::L_INFO, "INFO2");
  // file_logger.log(duna::Logger::L_INFO, "hey");
}