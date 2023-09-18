#pragma once

#include <ostream>
#include <unordered_map>
#include <vector>

namespace duna {
class Logger {
 public:
  enum VERBOSITY_LEVEL {
    L_ERROR,  // Error logging level
    L_WARN,   // Warn logging level
    L_INFO,   // Info logging level
    L_DEBUG,  // Debug logging level
  };

  Logger(std::ostream& sink, VERBOSITY_LEVEL level = L_ERROR);

  template <typename... Args>
  void log(VERBOSITY_LEVEL level, Args&&... args) {
    if (level > level_) return;

    sink_ << "duna::logger::" << level_prefix_.at(level) << ": ";
    (sink_ << ... << args);
    sink_ << std::endl;
  }

  inline void setLogLevel(VERBOSITY_LEVEL level) { level_ = level; }

 private:
  std::ostream& sink_;
  VERBOSITY_LEVEL level_;

  const std::unordered_map<VERBOSITY_LEVEL, std::string> level_prefix_;
};
}  // namespace duna