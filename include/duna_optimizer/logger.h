#pragma once

#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace duna {
/// @brief basic Logger class.
class Logger {
 public:
  using LoggerPtr = std::shared_ptr<Logger>;

  enum VERBOSITY_LEVEL {
    L_ERROR,  // Error logging level
    L_WARN,   // Warn logging level
    L_INFO,   // Info logging level
    L_DEBUG,  // Debug logging level
  };

  /// @brief
  /// @param sink ostream sink. Can be std::cout, a file, etc.
  /// @param level log level for that logger.
  /// @param name (optional) logger name.
  Logger(std::ostream& sink, VERBOSITY_LEVEL level = L_ERROR, const std::string&& name = "logger");
  duna::Logger& operator=(const duna::Logger& rhs) = delete;
  /// @brief perform loggging. Accepts any object that has the ostream overload.
  template <typename... Args>
  void log(VERBOSITY_LEVEL level, Args&&... args) const {
    if (level > level_) return;

    std::ostringstream content_stream;
    content_stream << "[" << level_prefix_.at(level) << "] duna::" << logger_name_ << "::";
    (content_stream << ... << args) << std::endl;
    auto content_string = content_stream.str();

    sink_ << content_string;

    // Log to additional logs.
    for (const auto& sink : added_sinks_) {
      *sink << content_string;
    }
  }

  /// @brief set log level of the logger.
  inline void setLogLevel(VERBOSITY_LEVEL level) { level_ = level; }

  /// @brief add extra logging sinks (ostream objects) to the logger.
  inline void addSink(std::ostream* sink) { added_sinks_.emplace(sink); }

 private:
  std::ostream& sink_;
  VERBOSITY_LEVEL level_;
  std::string logger_name_;
  std::unordered_set<std::ostream*> added_sinks_;

  const std::unordered_map<VERBOSITY_LEVEL, std::string> level_prefix_ = {
      {L_ERROR, "ERROR"},
      {L_WARN, "WARN"},
      {L_INFO, "INFO"},
      {L_DEBUG, "DEBUG"},
  };
};
}  // namespace duna