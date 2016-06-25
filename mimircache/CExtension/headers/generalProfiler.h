//
//  generalProfiler.h
//  generalProfiler
//
//  Created by Juncheng on 5/24/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef generalProfiler_h
#define generalProfiler_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <glib.h>
#include "reader.h"
#include "glib_related.h"
#include "cache.h" 
#include "const.h"

typedef struct{
    long long total_count;
    long long hit_count;
    long long miss_count;
    float miss_rate;
    float hit_rate;
    long long cache_size; 
}return_res;


struct multithreading_params_generalProfiler{
    READER* reader;
    guint64 begin_pos;
    guint64 end_pos;
    struct cache** caches;
    return_res** result;
    long num_of_cache;
    long bin_size;
    int order;
    int num_of_threads;
};

return_res** profiler(READER* reader_in, struct cache* cache_in, int num_of_threads_in, int bin_size_in, gint64 begin_pos, gint64 end_pos);



//long long* get_hit_count_seq(READER* reader, long size, long long begin, long long end);
//float* get_hit_rate_seq(READER* reader, long size, long long begin, long long end);
//float* get_miss_rate_seq(READER* reader, long size, long long begin, long long end);
//long long* get_reuse_dist_seq(READER* reader, long long begin, long long end);
//long long* get_rd_distribution(READER* reader, long long begin, long long end);
//long long* get_reversed_reuse_dist(READER* reader, long long begin, long long end);


#endif /* generalProfiler_h */
