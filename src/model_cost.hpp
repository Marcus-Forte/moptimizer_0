#pragma once

#include "cost_function.hpp"
#include <assert.h>
/*

Test data

model: y(x) = b0*x / (b1 + x)
i 	1 	2 	3 	4 	5 	6 	7
[S] 	0.038 	0.194 	0.425 	0.626 	1.253 	2.500 	3.740
Rate 	0.050 	0.127 	0.094 	0.2122 	0.2729 	0.2665 	0.3317

*/

/* Define your dataset type */
struct test_dataype1_t
{
    float *x; // model output
    float *y; // model input
};

template <int NPARAM >
class ModelCost : public CostFunction<NPARAM>
{
public:
    using VectorN = typename CostFunction<NPARAM>::VectorN;
    using VectorX = typename CostFunction<NPARAM>::VectorX;
    using MatrixX = typename CostFunction<NPARAM>::MatrixX;
    using CostFunction<NPARAM>::m_dataset;
    using CostFunction<NPARAM>::m_data_size;

    ModelCost(unsigned int data_size, void* dataset) : CostFunction<NPARAM>(data_size, dataset) {}
    virtual ~ModelCost() = default;

    // Computes error
    double f(const VectorN &x, VectorX &xout) override
    {

        test_dataype1_t* l_dataset = reinterpret_cast<test_dataype1_t*>(m_dataset);
        double sum = 0.0;

        for (int i = 0; i < m_data_size; ++i)
        {
            xout[i] = l_dataset->y[i] - (x[0] * l_dataset->x[i]) / (x[1] + l_dataset->x[i]);
            sum += xout[i]*xout[i];
        }

        return sum;
    }

    // Computes jacobian
    void df(const VectorN &x, MatrixX &xout) override
    {

        const float epsilon = 0.0001; 

        VectorX f_(xout.rows());
        f(x,f_);

        VectorX f_plus(xout.rows());
        for (int j = 0; j < NPARAM; ++j)
        {
            VectorN x_plus(x);
            x_plus[j] += epsilon;
            f(x_plus,f_plus);
            
            xout.col(j) = (f_plus - f_) / epsilon;
        }
    }


private:
};