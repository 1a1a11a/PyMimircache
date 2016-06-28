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
    struct cache* cache;
    return_res** result;
//    long num_of_cache;
    guint bin_size;
//    int order;
//    int num_of_threads;
};

return_res** profiler(READER* reader_in, struct cache* cache_in, int num_of_threads_in, int bin_size_in, gint64 begin_pos, gint64 end_pos);


#endif /* generalProfiler_h */
