#include <gtest/gtest.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Dense>
#include <memory>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

TEST(Draft, Draft1) {}