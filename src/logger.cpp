#include "duna/logger.h"

namespace duna
{

    void logger::log(VERBOSITY_LEVEL level, const std::stringstream &stream) const
    {
        if (level > level_)
            return;

        default_stream << levelToString(level) << stream.str() << std::endl;
        
    }

    void logger::log(VERBOSITY_LEVEL level, const char *format, ...) const
    {
        if (level > level_)
            return;

        va_list ap;

        va_start(ap, format);

        fprintf(stdout, "%s", levelToString(level).c_str());
        vfprintf(stdout, format, ap);
        fprintf(stdout, "\n");

        va_end(ap);
    }

    void logger::log(VERBOSITY_LEVEL level, const std::string &message) const
    {
        if (level > level_)
            return;

        default_stream << levelToString(level) << message << std::endl;
    }

    std::string logger::levelToString(VERBOSITY_LEVEL level) const
    {

        switch (level)
        {
        case (L_INFO):
            return "[INFO]: ";
        case (L_WARN):
            return "[WARN]: ";
        case (L_DEBUG):
            return "[DEBUG]: ";
        case (L_ERROR):
            return "[ERROR]: ";
        }

        return "";
    }

} // namespace