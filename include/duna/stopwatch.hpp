#pragma once

#include <chrono>
#include <iostream>
// Utility class for timing function calls

namespace utilities
{
    class Stopwatch
    {
    public:
        Stopwatch() = default;
        ~Stopwatch() = default;

        inline void enable()
        {
            is_enabled = true;
        }

        inline void disable()
        {
            is_enabled = false;
        }

        inline void tick()
        {
            if (!is_enabled)
                return;

            m_tick = std::chrono::high_resolution_clock::now();
        }

        // TODO decouble from <iostream>
        inline double tock(const std::string &message)
        {
            if (!is_enabled)
            {
                fprintf(stderr, "Warning, stopwatch not enabled\n");
                return 0;
            }

            const auto delta_tick = std::chrono::high_resolution_clock::now() - m_tick;
            double duration = std::chrono::duration<double>(delta_tick).count();
            fprintf(stderr, "'%s' took: %f [s]\n", message.c_str(), duration);

            return duration;
        }

    private:
        // Debug timing
        std::chrono::time_point<std::chrono::high_resolution_clock> m_tick;
        bool is_enabled = true;
    };
}
