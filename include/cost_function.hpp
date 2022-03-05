#ifndef COST_FUNCTION_HPP
#define COST_FUNCTION_HPP

#include <Eigen/Dense>
#include <iostream>


template <int NPARAM>
class CostFunction
{
public:
    // col vec
    using VectorN = Eigen::Matrix<float, NPARAM, 1>;
    using MatrixN = Eigen::Matrix<float, NPARAM, NPARAM>;
    using VectorX = Eigen::Matrix<float, Eigen::Dynamic, 1>;              //
    using MatrixX = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>; //

    CostFunction(unsigned int data_size, void *dataset) : m_data_size(data_size), m_dataset(dataset)
    {
    }
    virtual ~CostFunction() = default;

    // TODO maybe enforce @ constructor ?
    inline virtual void setDataset(void *dataset)
    {
        m_dataset = dataset;
      
    }

    inline unsigned int getDataSize() const
    {
        return m_data_size;
    }

    

    // dataset configuration
    virtual void init(const VectorN &x0)
    {
    }

    // Preprocess dataset before iteration
    virtual void preprocess(const VectorN &x0)
    {
        // No Op
    }

    // Process data after of iteration
    virtual void postprocess(VectorN &x0)
    {
    }

    virtual void finalize(VectorN &x0){
        
    }

    virtual double computeCost(const VectorN& x) = 0;
    // computes hessian and residual vector
    virtual double linearize(const VectorN &x, MatrixN &hessian, VectorN &b) = 0;


    protected:
    unsigned int m_data_size = 0;
    void *m_dataset = 0;

 
};

#endif