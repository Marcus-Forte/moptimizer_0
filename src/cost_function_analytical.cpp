#include "duna/cost_function_analytical.h"

/* This file is simply to show that is would be possible to compile cost function API,
but several template instantations would be needed */

namespace duna
{

    template <typename Scalar, int N_PARAMETERS, int N_MODEL_OUTPUTS>
    Scalar CostFunctionAnalytical<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::computeCost(const Scalar *x)
    {
        Scalar sum = 0;
        residuals_.resize(m_num_residuals * N_MODEL_OUTPUTS);

        for (int i = 0; i < m_num_residuals; ++i)
        {
            (*model_)(x, residuals_.template block<N_MODEL_OUTPUTS, 1>(i * N_MODEL_OUTPUTS, 0).data(), i);
        }
        sum = 2 * residuals_.transpose() * residuals_;
        return sum;
    }

    template class CostFunctionAnalytical<double, 6, 1>;
    template class CostFunctionAnalytical<float, 6, 1>;
    template class CostFunctionAnalytical<float, 2, 1>;
    template class CostFunctionAnalytical<double, 2, 1>;
    template class CostFunctionAnalytical<double, 4, 4>;

} // namespace