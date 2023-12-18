#include <duna_optimizer/cost_function_numerical.h>
#include <duna_optimizer/linearization.h>
#include <duna_optimizer/model.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <duna_optimizer/stopwatch.hpp>

using scalar = double;
using point_set_type = std::vector<Eigen::Vector3d>;

struct Point2PointDist : public duna_optimizer::BaseModel<scalar, Point2PointDist> {
  Point2PointDist(const point_set_type *src, const point_set_type *tgt) {
    src_ = src;
    tgt_ = tgt;
  }
  bool f(const scalar *x, scalar *f_x, unsigned int index) const override {
    const auto &src_pt = (*src_)[index];
    const auto &tgt_pt = (*tgt_)[index];

    auto dist = src_pt - tgt_pt;

    f_x[0] = dist[0];
    f_x[1] = dist[1];
    f_x[2] = dist[2];
    return true;
  }

 private:
  const point_set_type *src_;
  const point_set_type *tgt_;
};

/// @brief Test point set registration optimization with known correspondences.
class ParallelCostTest : public testing::Test {
 public:
  ParallelCostTest() {
    // Generate points;
    double hi_range = 10;
    double lo_range = 0;
    int n_elements = 1000000;

    for (int i = 0; i < n_elements; ++i) {
      Eigen::Vector3d pt = Eigen::Vector3d::Random();
      pt = (pt + Eigen::Vector3d::Constant(3, 1, 1)) * (hi_range - lo_range) * 0.5f;
      src_point_cloud_.emplace_back(pt);
    }

    // Transform point set
    Eigen::Affine3d t = Eigen::Affine3d::Identity();
    t.translate(Eigen::Vector3d(1.0, 2.0, 3.0));
    auto rotation = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()) *
                    Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ());
    // t.rotate(rotation);

    std::cout << "TF Matrix: " << t.matrix() << std::endl;
    std::for_each(src_point_cloud_.cbegin(), src_point_cloud_.cend(), [&](const auto &pt) {
      auto tgt_pt = t * pt;
      tgt_point_cloud_.emplace_back(tgt_pt);
    });
  }
  virtual ~ParallelCostTest() = default;

 protected:
  point_set_type src_point_cloud_;
  point_set_type tgt_point_cloud_;
};

TEST_F(ParallelCostTest, ComputeCost) {
  // Correspondences are the vector indices.
  auto num_points = this->src_point_cloud_.size();
  Point2PointDist::Ptr model(new Point2PointDist(&this->src_point_cloud_, &this->tgt_point_cloud_));
  // duna_optimizer::CostFunctionNumerical<scalar, 6, 3> cost(model, num_points);

  // auto diff = cost.computeCost(nullptr);

  Eigen::Matrix<scalar, 3, 1> residual_type;
  utilities::Stopwatch timer;

  duna_optimizer::CostComputation<scalar, 3, 3> computor;

  timer.tick();
  auto mt_diff = computor.parallelComputeCost(nullptr, model, num_points);

  auto delta = timer.tock();
  std::cerr << "Parallel thread cost compute" << delta << std::endl;
  timer.tick();
  auto st_diff = computor.computeCost(nullptr, model, num_points);
  delta = timer.tock();
  std::cerr << "Single thread cost compute" << delta << std::endl;

  EXPECT_NEAR(mt_diff, st_diff, 1e-8);
}