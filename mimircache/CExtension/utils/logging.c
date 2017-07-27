
#include "logging.h"


#ifdef __cplusplus
extern "C"
{
#endif


void log_lock(int lock)
{
    // static std::mutex m;
    static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    if (lock) {
        pthread_mutex_lock(&mtx);
        // m.lock();
    } else {
        pthread_mutex_unlock(&mtx);
        // m.unlock();
    }
}

int log_header(int level, const char *file, int line)
{
    if(level < LOGLEVEL) {
        return 0;
    }

    switch(level) {
        case VERBOSE_LEVEL:
            printf("%s[VERBOSE] ", BLUE);
            break;
        case DEBUG_LEVEL:
            printf("%s[DEBUG]   ", BLUE);
            break;
        case INFO_LEVEL:
            printf("%s[INFO]    ", GREEN);
            break;
        case WARNING_LEVEL:
            printf("%s[WARNING] ", YELLOW);
            break;
        case SEVERE_LEVEL:
            printf("%s[ERROR] ", RED);
            break;
    }

    char buffer[30];
    struct timeval tv;
    time_t curtime;

    gettimeofday(&tv, NULL);
    curtime = tv.tv_sec;
    strftime(buffer, 30, "%m-%d-%Y %T", localtime(&curtime));

    printf("%s %25s:%-4d ", buffer, strrchr(file, '/')+1, line);
    printf("(tid=%zu): ", (unsigned long) pthread_self());

    return 1;
}



#ifdef __cplusplus
}
#endif
