#pragma once
#include <iostream>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// TODO improve stream

#ifndef NDEBUG
// Prints Only in Debug compilation setting
#define DUNA_DEBUG(...) fprintf(stderr, "\x1b[32m[DBG] %s,%d: ",__FILENAME__,__LINE__); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\x1b[0m")
#define DUNA_DEBUG_STREAM(a) std::cerr << "\x1b[32m[DBG] " << a << "\x1b[0m"
#else
#define DUNA_DEBUG(...) do {} while(0)
#define DUNA_DEBUG_STREAM(a) do {} while(0)
#endif

// Log
#define DUNA_LOG(...) fprintf(stderr,"[LOG] %s,%d: ",__FILENAME__,__LINE__); fprintf(stderr,__VA_ARGS__)
#define DUNA_LOG_STREAM(a) std::cerr << a