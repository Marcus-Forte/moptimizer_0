#pragma once

#include <duna_optimizer/cost_function.h>
#include <gmock/gmock.h>

template <class Scalar>
class CostMock : public duna_optimizer::CostFunctionBase<Scalar> {
  MOCK_METHOD(Scalar, computeCost, (const Scalar*), (override));
};