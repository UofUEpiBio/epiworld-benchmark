#ifndef EPIWORLD_BENCHMARK_HPP
#define EPIWORLD_BENCHMARK_HPP

#include <iostream>
#include <chrono>

typedef std::chrono::time_point<std::chrono::system_clock> clocktime;
#define TOKENPASTE(a,b) a ## b
#define TIME_START(a) clocktime TOKENPASTE(a,_start) = std::chrono::system_clock::now();
#define TIME_END(a) clocktime TOKENPASTE(a,_end) = std::chrono::system_clock::now(); \
    std::chrono::duration< double, std::milli > \
    TOKENPASTE(a,_diff) = TOKENPASTE(a,_end) - TOKENPASTE(a,_start);
#define ELAPSED(a) TOKENPASTE(a,_diff).count()


#define REPLICATE(a) for(size_t i = 0u; i < static_cast<size_t>(a); ++i)

#endif
