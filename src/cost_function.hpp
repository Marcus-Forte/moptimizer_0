#ifndef COST_FUNCTION_HPP
#define COST_FUNCTION_HPP

#include <Eigen/Dense>


template <int NPARAM>
class CostFunction {
    public:

    using VectorN = Eigen::Matrix<float,NPARAM,1>; // col vec
    using VectorN_ = Eigen::Matrix<float,1,NPARAM>; // row vec
    using VectorX = Eigen::Matrix<float,Eigen::Dynamic, 1>; // 
    using MatrixX = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>; //
    
    
    CostFunction(unsigned int data_size, void* dataset) : m_data_size(data_size) , m_dataset(dataset){}
    virtual ~CostFunction() = default;

    
    // TODO maybe enforce @ constructor ?
    inline virtual void setData(void* dataset) {
        m_dataset = dataset;

        // TODO data must be reinterpreted within function
    }

    inline unsigned int getDataSize() {
        return m_data_size;
    }


    // TODO think about templated size | think about returning types

    // Computes error + sum error squared
    virtual double f(const VectorN& xi, VectorX& xout) = 0;

    // Computes jacobian
    virtual void df(const VectorN&x, MatrixX& xout) = 0;

   protected:
   // TODO use better name
   unsigned int m_data_size = 0;
   void* m_dataset = 0;
    
    
};

#endif