#pragma once

#include <chrono>


namespace utilities
{
    /// @brief Very simple class for timing function calls.
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
        
        /// @brief Start stopwatch timer.
        inline void tick()
        {
            if (!is_enabled)
                return;

            m_tick = std::chrono::high_resolution_clock::now();
        }
        
        /// @brief  Stop stopwatch timer.
        /// @return Time duration since last tick() in seconds.
        inline double tock() const
        {
            if (!is_enabled)
            {
                return -1.0;
            }

            const auto delta_tick = std::chrono::high_resolution_clock::now() - m_tick;
            double duration = std::chrono::duration<double>(delta_tick).count();

            return duration;
        }

    private:
        // Debug timing
        std::chrono::time_point<std::chrono::high_resolution_clock> m_tick;
        bool is_enabled = true;
    };
}
