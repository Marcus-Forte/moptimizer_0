#ifndef TYPES_H
#define TYPES_H

namespace duna
{
    enum Size
    {
        Dynamic = -1
    };

    enum OptimizationStatus
    {
        MAXIMUM_ITERATIONS_REACHED,
        SMALL_DELTA,
        NUMERIC_ERROR,
        FATAL_ERROR,
    };
} // namespace duna

#endif