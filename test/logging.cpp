#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "duna/logger.h"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}


// Need GMOCK?
TEST(Drafts, Logging)
{
    duna::logger logger;
    logger.setVerbosityLevel(duna::L_WARN);
    duna::logger::setGlobalVerbosityLevel(duna::L_DEBUG);
    duna::logger::log_info("info");
    duna::logger::log_warn("warn");
    duna::logger::log_error("error");
    duna::logger::log_debug("debug");
    logger.log(duna::L_INFO, "INFO");
    // logger.log(duna::L_WARN, "WARN");
    // logger.log(duna::L_ERROR, "ERROR");
    // logger.log(duna::L_DEBUG, "DEBUG");

    Eigen::Matrix4f some_matrix;

    std::stringstream somestream;
    somestream << some_matrix;
    logger.log(duna::L_WARN, somestream);

    logger.log(duna::L_WARN, "Hello world %d + %d = %d\n", 20, 30, 20 + 30);


}