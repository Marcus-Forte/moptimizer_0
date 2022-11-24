#pragma once
#include <cstdarg> // for va_list, va_start, va_end
#include <sstream>
#include <iostream>

#define _PRINT_MESSAGE(LEVEL)     \
    va_list ap;                   \
    va_start(ap, format);         \
    fprintf(stdout, "%s", LEVEL); \
    vfprintf(stdout, format, ap); \
    fprintf(stdout, "\n");        \
    va_end(ap)

namespace duna
{
    enum VERBOSITY_LEVEL
    {
        L_ERROR, // Error logging level
        L_WARN,  // Warn logging level
        L_INFO,  // Info logging level
        L_DEBUG, // Debug logging level
    };

    /* Basic logging class. Also provides static methods for `global` logging. */
    class logger
    {
    public:
        logger() : default_stream_(std::cout), level_(L_ERROR), logger_name_("LOG")
        {
        }

        logger(const std::string &logger_name) : default_stream_(std::cout), level_(L_ERROR), logger_name_(logger_name)
        {
        }

        virtual ~logger() = default;

        inline void setLoggerName(const std::string &name)
        {
            logger_name_ = name;
        }

        static void log_info(const char *format, ...)
        {
            if (L_INFO > s_level_)
                return;

            _PRINT_MESSAGE("[INFO]");
        }

        static void log_warn(const char *format, ...)
        {
            if (L_WARN > s_level_)
                return;

            _PRINT_MESSAGE("[WARN]");
        }
        static void log_error(const char *format, ...)
        {
            if (L_ERROR > s_level_)
                return;

            _PRINT_MESSAGE("[ERROR]");
        }
        static void log_debug(const char *format, ...)
        {
            if (L_DEBUG > s_level_)
                return;

            _PRINT_MESSAGE("[DEBUG]");
        }
        // static void log(const std::string &message);
        // static void log(const std::stringstream &stream);

        void log(VERBOSITY_LEVEL level, const char *format, ...) const;
        void log(VERBOSITY_LEVEL level, const std::string &message) const;
        void log(VERBOSITY_LEVEL level, const std::stringstream &stream) const;

        inline void setVerbosityLevel(VERBOSITY_LEVEL level)
        {
            level_ = level;
        }

        static inline void setGlobalVerbosityLevel(VERBOSITY_LEVEL level)
        {
            s_level_ = level;
        }

    private:
        VERBOSITY_LEVEL level_;
        std::ostream &default_stream_;
        std::string levelToString(VERBOSITY_LEVEL level) const;
        std::string logger_name_;

        static VERBOSITY_LEVEL s_level_;
    };
} // namespace