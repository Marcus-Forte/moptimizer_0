#include <gtest/gtest.h>

#include <Eigen/Dense>

/* Draft space for testing quick stuff */

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}


TEST(Draft, Draft)
{
Eigen::Matrix<double,-1,1> vector;

vector.resize(10);
vector.setZero();

double* pointer = vector.data();

pointer[0] = 1;
pointer[1] = 2;
std::cout << vector << std::endl;


}
