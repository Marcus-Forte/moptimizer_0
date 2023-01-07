#include <gtest/gtest.h>
#include <duna/parameter.h>


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}


TEST(Parameters, Euclidean)
{
    duna::Parameter<3> A;
    duna::Parameter<3> B;
    duna::Parameter<3> C;
    duna::plus(&A,&B,&C);
}