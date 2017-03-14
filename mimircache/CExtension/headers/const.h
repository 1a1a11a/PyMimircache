

#define cache_line_label_size 1024
#define CACHE_LINE_LABEL_SIZE 1024
#define FILE_LOC_STR_SIZE 1024

#define KRESET  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"




#define hit_rate_start_time_end_time                    1
#define hit_rate_start_time_cache_size                  2
#define avg_rd_start_time_end_time                      3
#define cold_miss_count_start_time_end_time             4
#define rd_distribution                                 5
#define future_rd_distribution                          6
#define rd_distribution_CDF                             7


#define NORMAL_REUSE_DISTANCE                           1
#define FUTURE_REUSE_DISTANCE                           2



#define SANITY_CHECK 1
#define _DEBUG
#define ML
#undef ML 

#undef __DEBUG__
#undef _DEBUG


#if defined(__DEBUG__) || defined(_DEBUG)
#define DEBUG_MSG(...) fprintf(stderr, __VA_ARGS__)
#define DEBUG(...) \
{fprintf(stderr, "[DEBUG]: %s:%d:%s: ", __FILE__, __LINE__, __func__); \
fprintf(stderr, __VA_ARGS__);}
#else
#define DEBUG_MSG(...) do { } while (0)
#define DEBUG(...) do { } while (0)
#endif

// #define INFO(...) fprintf(stderr, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define INFO(...) \
    {fprintf(stderr, "[INFO]: %s%s:%d:%s: ", KYEL, __FILE__, __LINE__, __func__); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "%s", KRESET); }

#define WARNING(...) \
    {fprintf(stderr, "[WARNING]: %s:%d:%s: ", __FILE__, __LINE__, __func__); \
    fprintf(stderr, __VA_ARGS__);}

#define ERROR(...) \
    {fprintf(stderr, "[ERROR]: %s:%d:%s: ", __FILE__, __LINE__, __func__); \
    fprintf(stderr,  __VA_ARGS__);}

