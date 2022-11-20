#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <duna/stopwatch.hpp>
#include <duna/logger.h>
#include <omp.h>

#include <memory>
/* Draft space for testing quick stuff */
int num_param = 100;
int num_res = 10000;
utilities::Stopwatch timer;
#define TOL 1e-6
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    if (argc > 1)
        num_param = atoi(argv[1]);

    if (argc > 2)
        num_res = atoi(argv[2]);

    std::cerr << "Num Param: " << num_param << std::endl;
    std::cerr << "Num Res: " << num_res << std::endl;

    return RUN_ALL_TESTS();
}


TEST(Drafts, Logging)
{
    duna::logger logger;
    logger.setVerbosityLevel(duna::L_WARN);
    // logger.log(duna::L_INFO, "INFO");
    // logger.log(duna::L_WARN, "WARN");
    // logger.log(duna::L_ERROR, "ERROR");
    // logger.log(duna::L_DEBUG, "DEBUG");

    Eigen::Matrix4f some_matrix;

    std::stringstream somestream;
    somestream << some_matrix;
    logger.log(duna::L_WARN, somestream);

    logger.log(duna::L_WARN, "Hello world %d + %d = %d\n", 20, 30, 20+30);
}

TEST(Drafts, Draft1)
{
    Eigen::MatrixXd bigJacobian(num_res, num_param);
    bigJacobian.setRandom();

    Eigen::VectorXd residuals(num_res);
    residuals.setRandom();
    Eigen::MatrixXd Hessian;

    timer.tick();
    Hessian = bigJacobian.transpose() * bigJacobian;
    timer.tock("Hessian");

    // Hessian.resize(num_param,num_param);
    // Hessian.setRandom();

    timer.tick();
    residuals = bigJacobian.transpose() * residuals;
    timer.tock("residuals");

    timer.tick();
    Eigen::LLT<Eigen::MatrixXd> LLT_solver(Hessian);
    Eigen::VectorXd LLT_solution = LLT_solver.solve(-residuals);
    timer.tock("LLT");

    timer.tick();
    Eigen::LDLT<Eigen::MatrixXd> LDLT_solver(Hessian);
    Eigen::VectorXd LDLT_solution = LDLT_solver.solve(-residuals);
    timer.tock("LDLT");

    timer.tick();
    Eigen::PartialPivLU<Eigen::MatrixXd> PartialPivLU_solver(Hessian);
    Eigen::VectorXd PartialPivLU_solution = PartialPivLU_solver.solve(-residuals);
    timer.tock("PartialPivLU");

    timer.tick();
    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> FullPivHouseholderQR_solver(Hessian);
    Eigen::VectorXd FullPivHouseholderQR_solution = FullPivHouseholderQR_solver.solve(-residuals);
    timer.tock("FullPivHouseholderQR");

    timer.tick();
    Eigen::FullPivLU<Eigen::MatrixXd> FullPivLU_solver(Hessian);
    Eigen::VectorXd FullPivLU_solution = FullPivLU_solver.solve(-residuals);
    timer.tock("FullPivLU");

    timer.tick();
    Eigen::MatrixXd inverse_ = Hessian.inverse();
    Eigen::VectorXd INVERSE_solution = -(inverse_ * residuals);
    timer.tock("inverse_");

    // timer.tick();
    // Eigen::JacobiSVD<Eigen::MatrixXd> JacobiSVD_solver(Hessian);
    // Eigen::VectorXd JacobiSVD_solution = JacobiSVD_solver.solve(-residuals);
    // timer.tock("JacobiSVD");

    for (int i = 0; i < residuals.size(); ++i)
    {
        EXPECT_NEAR(FullPivHouseholderQR_solution[i], LDLT_solution[i], TOL);
        EXPECT_NEAR(FullPivHouseholderQR_solution[i], LLT_solution[i], TOL);
        EXPECT_NEAR(FullPivHouseholderQR_solution[i], FullPivLU_solution[i], TOL);
        EXPECT_NEAR(FullPivHouseholderQR_solution[i], INVERSE_solution[i], TOL);
        // EXPECT_NEAR(FullPivHouseholderQR_solution[i], JacobiSVD_solution[i], TOL);
        EXPECT_NEAR(FullPivHouseholderQR_solution[i], PartialPivLU_solution[i], TOL);
    }
}
/* Without Vectorization (1000 params, 50000 res) */
// 'Hessian' took: 2.079559 [s]
// 'residuals' took: 0.024136 [s]
// 'LLT' took: 0.035371 [s]
// 'LDLT' took: 0.050970 [s]
// 'PartialPivLU' took: 0.039047 [s]
// 'FullPivHouseholderQR' took: 0.501205 [s]
// 'FullPivLU' took: 0.445898 [s]
// 'inverse_' took: 0.189802 [s]

/* With Vectorization (1000 params, 50000 res) */
// 'Hessian' took: 0.701434 [s] ++
// 'residuals' took: 0.022840 [s]
// 'LLT' took: 0.016674 [s]
// 'LDLT' took: 0.048059 [s]
// 'PartialPivLU' took: 0.024906 [s]
// 'FullPivHouseholderQR' took: 0.403789 [s]
// 'FullPivLU' took: 0.342347 [s]
// 'inverse_' took: 0.089391 [s]
