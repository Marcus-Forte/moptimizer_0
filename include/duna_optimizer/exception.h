#pragma once

#include <exception>
#include <string>
namespace duna_optimizer {

class Exception : public std::exception {
 public:
  explicit Exception(const char *message) : msg_(message) {}

  explicit Exception(const std::string &message) : msg_(message) {}

  virtual ~Exception() noexcept {}

  virtual const char *what() const noexcept { return msg_.c_str(); }

 protected:
  std::string msg_;
};

}  // namespace duna_optimizer