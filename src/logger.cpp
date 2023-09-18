#include <duna_optimizer/logger.h>

namespace duna {
Logger::Logger(std::ostream& sink, VERBOSITY_LEVEL level, const std::string&& name)
    : sink_(sink), level_(level), logger_name_(name) {}

}  // namespace duna