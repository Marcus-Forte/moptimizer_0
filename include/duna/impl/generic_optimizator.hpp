#include "duna/generic_optimizator.h"

template <int NPARAM>
GenericOptimizator<NPARAM>::GenericOptimizator(CostFunction<NPARAM> *cost) : Optimizator<NPARAM>(cost)
{
}

template <int NPARAM>
typename GenericOptimizator<NPARAM>::Status GenericOptimizator<NPARAM>::minimize(VectorN &x0)
{
    if (m_cost == 0)
    {
        std::cerr << "no cost object!\n";
        throw std::runtime_error("no cost object.");
    }

    m_cost->checkData();

    // LM Configuration
    double lm_init_lambda_factor_ = 1e-9;
    double lm_lambda_ = -1.0;
    int lm_max_iterations_ = 10;

    MatrixN hessian_diag_;

    MatrixN hessian;
    VectorN xi(NPARAM);
    VectorN b(NPARAM);

    for (int j = 0; j < m_max_it; ++j)
    {
        DUNA_DEBUG_STREAM("## GenericOptimizator Iteration: " << j + 1 << "/" << m_max_it << " ##\n");

        // linearization
        double y0 = m_cost->linearize(x0, hessian, b);

        hessian_diag_ = hessian.diagonal().asDiagonal(); // MatrixN::Identity();
                                                         // diag_ =  MatrixN::Identity();

        // DUNA_DEBUG_STREAM("Hessian:\n"
        //                   << hessian << std::endl);

        if (lm_lambda_ < 0.0)
        {
            lm_lambda_ = lm_init_lambda_factor_ * hessian.diagonal().array().abs().maxCoeff();
            // lm_lambda_ = lm_init_lambda_factor_;
        }

        // // LM Iterations
        double nu = 2.0;
        for (int k = 0; k < lm_max_iterations_; ++k)
        {

            // DUNA_DEBUG_STREAM("A: \n"
            //                   << (hessian + lm_lambda_ * diag_) << "\n");
            // DUNA_DEBUG_STREAM("b: \n"
            //                   << b << "\n");

            // VectorN delta = (hessian + lm_lambda_ * diag_).inverse() * b;
            // Eigen::LDLT<MatrixN> solver(hessian + lm_lambda_ * diag_);

            // DUNA_DEBUG_STREAM("Hessian: " << hessian << std::endl);
            // DUNA_DEBUG_STREAM("b: " << b << std::endl);

            Eigen::LDLT<MatrixN> solver(hessian + lm_lambda_ * MatrixN::Identity());
            VectorN delta = solver.solve(b);

            // VectorN delta = (hessian + lm_lambda_ * hessian_diag_).inverse() * b;

            DUNA_DEBUG("--- Solver delta: ");
            for (int n = 0; n < NPARAM; ++n)
            {
                fprintf(stderr," %f", delta[n]);
            }//
            fprintf(stderr,"\n");

            if (testConvergence(delta) == 0)
            {
                return Status::SMALL_DELTA;
            }

            xi = x0 - delta;

            // DUNA_DEBUG_STREAM("x0: " << x0 << std::endl);
            // DUNA_DEBUG_STREAM("delta: " << delta << std::endl);
            // DUNA_DEBUG_STREAM("xi: " << xi << std::endl);

            // Uncomment below to use Gauss Newton approach
            // x0 = xi;
            // break;

            double yi = m_cost->computeCost(xi);
            double rho = (yi - y0) / delta.dot(lm_lambda_ * delta - b);
            DUNA_DEBUG("--- Internal LM Iteration --- : %d/%d | %f %f %f %f %f\n", k + 1, lm_max_iterations_, y0, yi, rho, lm_lambda_, nu);

            // check if output is worse
            if (rho < 0)
            {
                if (testConvergence(delta) == 0)
                {
                    return Status::SMALL_DELTA;
                }

                lm_lambda_ = nu * lm_lambda_;
                nu = 2 * nu;
                continue;
            }

            x0 = xi;
            lm_lambda_ = lm_lambda_ * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));

            break;
        }
    }
    
    return Status::MAX_IT_REACHED;
}

// TODO improve!!
template <int NPARAM>
int GenericOptimizator<NPARAM>::testConvergence(const VectorN &delta)
{
    double epsilon = delta.array().abs().maxCoeff();
    // DUNA_DEBUG_STREAM("epsilon: " << epsilon << "\n");

    if (epsilon < 5e-6)
        return 0;
    return 1;
}
