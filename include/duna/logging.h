#pragma once
#include <iostream>

namespace duna
{
    enum VERBOSITY_LEVEL
    {
        L_ERROR,
        L_WARN,
        L_INFO,
        L_DEBUG,
        L_VERBOSE
    };

    class logger
    {
    public:
    public:
        logger() = default;
        virtual ~logger() = default;

        inline void log(VERBOSITY_LEVEL level, const std::string &message)
        {
            if (level > level_)
                return;

            std::cout << levelToString(level) << message << std::endl;
        }

        inline void
        setVerbosityLevel(VERBOSITY_LEVEL level)
        {
            level_ = level;
        }

    private:
        VERBOSITY_LEVEL level_ = L_VERBOSE;

        std::string levelToString(VERBOSITY_LEVEL level) const
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
    };
} // namespace