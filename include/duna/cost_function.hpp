#pragma once

#include <Eigen/Dense>

template <int NPARAM>
class CostFunction
{
public:
    // col vec
    using VectorN = Eigen::Matrix<float, NPARAM, 1>;
    using MatrixN = Eigen::Matrix<float, NPARAM, NPARAM>;


    CostFunction(void *dataset) :  m_dataset(dataset)
    {
    }
    virtual ~CostFunction() = default;

    // TODO rethink design
    inline void setDataset(void *dataset)
    {
        m_dataset = dataset;
    }

    inline void* getDataset()
    {
        return m_dataset;
    }  

    virtual void checkData() = 0;

    virtual double computeCost(const VectorN& x) = 0;
    // computes hessian and residual vector
    virtual double linearize(const VectorN &x, MatrixN &hessian, VectorN &b) = 0;


    protected:
    void *m_dataset = 0;

 
};
