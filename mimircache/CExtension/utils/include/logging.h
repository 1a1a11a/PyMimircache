#ifndef LOGGING_H
#define LOGGING_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <execinfo.h>


#define NORMAL  "\x1B[0m"
#define RED     "\x1B[31m"
#define GREEN   "\x1B[32m"
#define YELLOW  "\x1B[33m"
#define BLUE    "\x1B[34m"
#define MAGENTA "\x1B[35m"
#define CYAN    "\x1B[36m"

#define VERBOSE_LEVEL 1
#define DEBUG_LEVEL   2
#define INFO_LEVEL    3
#define WARNING_LEVEL 4
#define SEVERE_LEVEL  5

#ifndef LOGLEVEL
    #define LOGLEVEL VERBOSE_LEVEL
#endif // LOGLEVEL


int log_header(int level, const char *file, int line);
void log_lock(int);


#define LOGGING(level, FMT, ...) { \
    log_lock(1); \
    if(log_header(level, __FILE__, __LINE__)) { \
        printf(FMT, ##__VA_ARGS__); printf("%s", NORMAL); fflush(stdout); \
    } \
    log_lock(0); \
}

#if LOGLEVEL <= VERBOSE_LEVEL
    #define verbose(FMT, ...) LOGGING(VERBOSE_LEVEL, FMT, ##__VA_ARGS__)
#else
    #define verbose(FMT, ...)
#endif

#if LOGLEVEL <= DEBUG_LEVEL
    #define debug(FMT, ...) LOGGING(DEBUG_LEVEL, FMT, ##__VA_ARGS__)
#else
    #define debug(FMT, ...)
#endif

#if LOGLEVEL <= INFO_LEVEL
    #define info(FMT, ...) LOGGING(INFO_LEVEL, FMT, ##__VA_ARGS__)
#else
    #define info(FMT, ...)
#endif

#if LOGLEVEL <= WARNING_LEVEL
    #define warning(FMT, ...) LOGGING(WARNING_LEVEL, FMT, ##__VA_ARGS__)
#else
    #define warning(FMT, ...)
#endif

#if LOGLEVEL <= SEVERE_LEVEL
    #define error_msg(FMT, ...) LOGGING(SEVERE_LEVEL, FMT, ##__VA_ARGS__)
#else
    #define error_msg(FMT, ...)
#endif



void print_stack_trace(); 

#endif
