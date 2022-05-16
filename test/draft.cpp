#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <unordered_map>
#include <unordered_set>

/* Draft space for testing quick stuff */

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}


TEST(Draft, Draft)
{
    std::unordered_set<int> set;

    if(!set.count(55))
        set.insert(55);


}
