#pragma once

#include "cost_function.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>

#include <assert.h>
/*

/* Define your dataset */
struct datatype_t
{
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr source;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr target;
    pcl::search::KdTree<pcl::PointXYZ>::ConstPtr tgt_kdtree;
};

template <int NPARAM>
class CalibrationCost : public CostFunction<NPARAM>
{
public:
    using VectorN = typename CostFunction<NPARAM>::VectorN;
    using VectorN_ = typename CostFunction<NPARAM>::VectorN_;
    using VectorX = typename CostFunction<NPARAM>::VectorX;
    using MatrixX = typename CostFunction<NPARAM>::MatrixX;
    using CostFunction<NPARAM>::m_dataset;
    using CostFunction<NPARAM>::m_data_size;

    CalibrationCost(unsigned int data_size, void * dataset) : CostFunction<NPARAM>(data_size, dataset) {}
    virtual ~CalibrationCost() = default;

    // Computes error
    double f(const VectorN &xi, VectorX &xout) override
    {
        double sum = 0;


        return sum;
    }

    // Computes jacobian
    void df(const VectorN &x, MatrixX &xout) override
    {
    }

private:
};