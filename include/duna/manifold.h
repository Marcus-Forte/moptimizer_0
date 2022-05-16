#ifndef MANIFOLD_H
#define MANIFOLD_H

#include <Eigen/Dense>
#include <iostream>

/* All optimization parameters inherit this class */

// Dim is the tangend space dimension
template <int TangentDim, int LinearDim = TangentDim, typename Scalar = double>
class Manifold
{

public:
    using LinearRepresentation = Eigen::Matrix<Scalar, LinearDim, 1>;
    using TangentRepresentation = Eigen::Matrix<Scalar, TangentDim, 1>;

    Manifold() = default;
    Manifold(const LinearRepresentation &linear_rep) : parameter(linear_rep) {}

    LinearRepresentation &getEunclideanRepresentation()
    {
        return parameter;
    }

    // I tried to think of a way to use operator+ overload.. no success;
    virtual void Plus(const TangentRepresentation& rhs) = 0;

    virtual void Minus(const TangentRepresentation& rhs) = 0;

protected:
    LinearRepresentation parameter;
};


/* Euclidean Manifolds use traditional operators */
template <int Dim>
class EuclideanManifold : public Manifold<Dim>
{
    using typename Manifold<Dim>::LinearRepresentation;
    using typename Manifold<Dim>::TangentRepresentation;
    using Manifold<Dim>::parameter;

    public:

    void Plus(const TangentRepresentation& rhs) override
    {
        parameter += rhs;
    }

    void Minus(const TangentRepresentation& rhs) override
    {
        parameter -= rhs;
    }

};

#endif