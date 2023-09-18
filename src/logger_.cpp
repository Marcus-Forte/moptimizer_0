#include <duna_optimizer/logger_.h>

namespace duna {
Logger::Logger(std::ostream& sink, VERBOSITY_LEVEL level)
    : sink_(sink),
      level_(level),
      level_prefix_({
          {L_ERROR, "ERROR"},
          {L_WARN, "WARN"},
          {L_INFO, "INFO"},
          {L_DEBUG, "DEBUG"},
      }) {}
}  // namespace duna