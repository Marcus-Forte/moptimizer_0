#include <gtest/gtest.h>
#include <duna_optimizer/logger.h>


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  duna_optimizer::logger::setGlobalVerbosityLevel(duna_optimizer::L_DEBUG);

  return RUN_ALL_TESTS();
}