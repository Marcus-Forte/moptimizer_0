#pragma once

#include <moptimizer/cost_function.h>
#include <gmock/gmock.h>

template <class Scalar>
class CostMock : public moptimizer::CostFunctionBase<Scalar> {
  MOCK_METHOD(Scalar, computeCost, (const Scalar*), (override));
};