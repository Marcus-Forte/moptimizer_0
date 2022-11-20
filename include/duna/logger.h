#pragma once
#include <cstdarg> // for va_list, va_start, va_end
#include <sstream>
#include <iostream>

namespace duna
{
    enum VERBOSITY_LEVEL
    {
        L_ERROR, // Error logging level
        L_WARN, // Warn logging level
        L_INFO, // Info logging level
        L_DEBUG, // Debug logging level
    };

    class logger
    {
    public:
        logger() : default_stream(std::cout), level_(L_ERROR) {}
        virtual ~logger() = default;

        void log(VERBOSITY_LEVEL level, const char* format, ...) const;
        void log(VERBOSITY_LEVEL level, const std::string &message) const;
        void log(VERBOSITY_LEVEL level, const std::stringstream& stream) const;

        inline void setVerbosityLevel(VERBOSITY_LEVEL level)
        {
            level_ = level;
        }

    private:
        VERBOSITY_LEVEL level_;
        std::ostream& default_stream;

        std::string levelToString(VERBOSITY_LEVEL level) const;
    };
} // namespace