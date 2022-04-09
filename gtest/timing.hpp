#pragma once

#include <iostream>
#include <chrono>

// Utility class for timing function calls
class Timing 
{
    public:
    Timing() = default;
    ~Timing() = default;

    inline void enable(bool enable_)
    {
        is_enabled = enable_;
    }

    inline void tick() 
    {
        if(!is_enabled)
            return;


        m_tick = std::chrono::high_resolution_clock::now();
    }

    inline void tock(const std::string& message)
    {
        if(!is_enabled)
            return;

        const auto delta_tick = std::chrono::high_resolution_clock::now() - m_tick;
        fprintf(stderr, "'%s' took: %f [s]\n", message.c_str(), std::chrono::duration<double> (delta_tick).count());
    }


    private:
        // Debug timing
    std::chrono::time_point<std::chrono::high_resolution_clock> m_tick;
    bool is_enabled = false;

};