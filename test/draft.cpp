#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <memory.h>
/* Draft space for testing quick stuff */

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

struct A
{
    A(int *ptr_)
    {
        ptr = ptr_;
    }
    ~A()
    {
        delete ptr;
    }

    int *ptr;
};

void function(A *input)
{

    delete input;
}

TEST(Draft, Draft)
{

    function(new A(new int));
}
