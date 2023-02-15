#include "duna/logger.h"

namespace duna
{
    // Default setting for global/static log level;
    VERBOSITY_LEVEL logger::s_level_ = L_ERROR;

    void logger::log(VERBOSITY_LEVEL level, const std::stringstream &stream) const
    {
        if (level > level_)
            return;

        default_stream_ << "[" << logger_name_ << "." << levelToString(level) << "]: " << stream.str() << std::endl;
    }

    void logger::log(VERBOSITY_LEVEL level, const char *format, ...) const
    {
        if (level > level_)
            return;

        va_list ap;

        va_start(ap, format);

        fprintf(stdout, "[%s.%s]: ", logger_name_.c_str(), levelToString(level).c_str());
        vfprintf(stdout, format, ap);
        fprintf(stdout, "\n");

        va_end(ap);
    }

    void logger::log(VERBOSITY_LEVEL level, const std::string &message) const
    {
        if (level > level_)
            return;

        default_stream_ << "[" << logger_name_ << "." << levelToString(level) << "]: " << message << std::endl;
    }

    std::string logger::levelToString(VERBOSITY_LEVEL level) const
    {

        switch (level)
        {
        case (L_INFO):
            return "duna::opt::INFO";
        case (L_WARN):
            return "duna::opt::WARN";
        case (L_DEBUG):
            return "duna::opt::DEBUG";
        case (L_ERROR):
            return "duna::opt::ERROR";
        }

        return "";
    }

} // namespace