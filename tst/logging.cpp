#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "duna_optimizer/logger.h"

// Need GMOCK?
TEST(Drafts, Logging) {
  duna_optimizer::logger logger;
  logger.setVerbosityLevel(duna_optimizer::L_WARN);
  duna_optimizer::logger::setGlobalVerbosityLevel(duna_optimizer::L_DEBUG);
  duna_optimizer::logger::log_info("info");
  duna_optimizer::logger::log_warn("warn");
  duna_optimizer::logger::log_error("error");
  duna_optimizer::logger::log_debug("debug");
  logger.log(duna_optimizer::L_INFO, "INFO");
  // logger.log(duna::L_WARN, "WARN");
  // logger.log(duna::L_ERROR, "ERROR");
  // logger.log(duna::L_DEBUG, "DEBUG");

  Eigen::Matrix4f some_matrix;

  std::stringstream somestream;
  somestream << some_matrix;
  logger.log(duna_optimizer::L_WARN, somestream);

  logger.log(duna_optimizer::L_WARN, "Hello world %d + %d = %d\n", 20, 30, 20 + 30);
}