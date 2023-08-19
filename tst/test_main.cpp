#include <duna_optimizer/logger.h>
#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  duna_optimizer::logger::setGlobalVerbosityLevel(duna_optimizer::L_DEBUG);

  return RUN_ALL_TESTS();
}