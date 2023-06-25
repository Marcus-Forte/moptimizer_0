#include <duna/parameter.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(Parameters, Euclidean) {
  duna_optimizer::Parameter<3> A;
  duna_optimizer::Parameter<3> B;
  duna_optimizer::Parameter<3> C;
  duna_optimizer::plus(&A, &B, &C);
}