#include <duna/parameter.h>
#include <gtest/gtest.h>


TEST(Parameters, Euclidean) {
  duna_optimizer::Parameter<3> A;
  duna_optimizer::Parameter<3> B;
  duna_optimizer::Parameter<3> C;
  duna_optimizer::plus(&A, &B, &C);
}