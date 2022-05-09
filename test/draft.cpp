#include <gtest/gtest.h>

#include <Eigen/Dense>

/* Draft space for testing quick stuff */

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

class Obj
{

public:
    Obj(

    )
    {
        std::cout << "Ctr\n";
    }

    Obj(const Obj & rhs)
    {
        std::cout << "Cpy Crt\n";
        member = rhs.member;
    }

public:
    int member;
};

TEST(Draft, Draft)
{
    Obj *a = new Obj;

    a->member = 10;
    std::vector<Obj> obj_vec(10, *a);

    a->member = 1;

    std::cout << obj_vec[5].member << std::endl;

    obj_vec[3].member = 25;

    for (auto& it : obj_vec)
        std::cout << it.member << std::endl;
}
