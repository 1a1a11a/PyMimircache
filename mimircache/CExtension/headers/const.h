

#define cache_line_label_size 1024
#define FILE_LOC_STR_SIZE 1024 

#define KNRM  "\x1B[0m"
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

//#define DEBUG 

#ifndef DEBUG
#define DEBUG(stat) ;
#endif 