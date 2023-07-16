#include <gtest/gtest.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Dense>
#include <memory>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

// bool fun(float* x, float* f_x) {
//   f_x[0] = x[0] * x[0];
//   return true;
// }

struct Fun {
  Fun(const std::vector<float>& input, std::vector<float>& output)
      : input_(input), output_(output) {}

  void operator()(tbb::blocked_range<int> r) const {
    for (int i = r.begin(); i != r.end(); ++i) {
      output_[i] = input_[i] * input_[i];
    }
  }

  const std::vector<float>& input_;
  std::vector<float>& output_;
};

TEST(Draft, Draft1) {
  const int n = 100;
  std::vector<float> x_i(n);
  std::vector<float> res(n);
  int i = 0;
  for (auto& it : x_i) {
    it = i++;
  }

  using iterator = std::vector<float>::const_iterator;
  tbb::blocked_range<int> r(0, x_i.size());

  Fun fun(x_i, res);

  std::cout << fun.input_[5];

  // fun()

  tbb::parallel_for(r, fun);

  // tbb::parallel_for(r, [&](const tbb::blocked_range<int>& r) {
  //   // for (int i = r.begin(); i != r.end(); ++i) {
  //     // res[i] = x_i[i] * x_i[i];
  //   // }
  // });

  auto sum = tbb::parallel_reduce(
      r, 0.0,
      [&](tbb::blocked_range<int> r, double running) -> double {
        for (int i = r.begin(); i < r.end(); ++i) {
          running += res[i];
        }

        return running;
      },
      std::plus<double>());

  for (const auto& it : res) {
    std::cout << "Res: " << it << std::endl;
  }

  std::cout << "sum: " << sum << std::endl;
}