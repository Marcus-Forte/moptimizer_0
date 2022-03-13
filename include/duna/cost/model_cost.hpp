#pragma once

#include "duna/cost_function.hpp"
#include <assert.h>
/*

Test data

model: y(x) = b0*x / (b1 + x)
i 	1 	2 	3 	4 	5 	6 	7
[S] 	0.038 	0.194 	0.425 	0.626 	1.253 	2.500 	3.740
Rate 	0.050 	0.127 	0.094 	0.2122 	0.2729 	0.2665 	0.3317

*/
namespace duna
{
    /* Define your dataset type */
    struct test_dataype1_t
    {
        float *x; // model output
        float *y; // model input
        int data_size;
    };

    template <int NPARAM>
    class ModelCost : public CostFunction<NPARAM>
    {
    public:
        using VectorN = typename CostFunction<NPARAM>::VectorN;
        using MatrixN = typename CostFunction<NPARAM>::MatrixN;

        using CostFunction<NPARAM>::m_dataset;

        ModelCost(void *dataset) : CostFunction<NPARAM>(dataset)
        {
            l_dataset = reinterpret_cast<test_dataype1_t *>(m_dataset);
        }
        virtual ~ModelCost() = default;

        virtual void checkData() override
        {
        }

        double computeCost(const VectorN &x) override
        {
            double sum = 0;
            for (int i = 0; i < l_dataset->data_size; ++i)
            {

                // fout
                float xout = l_dataset->y[i] - (x[0] * l_dataset->x[i]) / (x[1] + l_dataset->x[i]);
                sum += xout * xout;
            }

            return sum;
        }

        double linearize(const VectorN &x, MatrixN &hessian, VectorN &b) override
        {
            test_dataype1_t *l_dataset = reinterpret_cast<test_dataype1_t *>(m_dataset);
            double sum = 0;
            hessian.setZero();
            b.setZero();

            Eigen::Matrix<float, 1, NPARAM> jacobian_row;

            for (int i = 0; i < l_dataset->data_size; ++i)
            {

                // fout
                float xout = l_dataset->y[i] - (x[0] * l_dataset->x[i]) / (x[1] + l_dataset->x[i]);

                // jacobian
                const float epsilon = 0.0001;

                for (int j = 0; j < NPARAM; ++j)
                {
                    VectorN x_plus(x);
                    x_plus[j] += epsilon;

                    float xout_plus = l_dataset->y[i] - (x_plus[0] * l_dataset->x[i]) / (x_plus[1] + l_dataset->x[i]);
                    jacobian_row[j] = (xout_plus - xout) / epsilon;
                }

                hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
                b += jacobian_row.transpose() * xout;

                sum += xout * xout;
            }

            hessian.template triangularView<Eigen::Upper>() = hessian.transpose();

            return sum;
        }

    private:
        test_dataype1_t *l_dataset;
    };

}