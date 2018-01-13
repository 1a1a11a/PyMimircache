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
#include <string.h> 
#include "reader.h"
#include "glib_related.h"
#include "cache.h" 
#include "const.h"



#ifdef __cplusplus
extern "C"
{
#endif


typedef struct{
    long long total_count;
    long long hit_count;        // this can be negative!!
    long long miss_count;
    float miss_rate;
    float hit_rate;
    long long cache_size; 
}return_res;


struct multithreading_params_generalProfiler{
    reader_t* reader;
    guint64 begin_pos;
    guint64 end_pos;
    struct cache* cache;
    return_res** result;
    guint bin_size;
    GHashTable *prefetch_hashtable;
    GMutex mtx;             // prevent simultaneous write to progress
    guint64* progress;
};
typedef struct multithreading_params_generalProfiler mt_param_gp_t;


return_res** profiler(reader_t* reader_in, struct cache* cache_in, int num_of_threads_in, int bin_size_in, gint64 begin_pos, gint64 end_pos);

gdouble* LRU_evict_err_statistics(reader_t* reader_in, struct_cache* cache_in, guint64 time_interval);


#ifdef __cplusplus
}
#endif

#endif /* generalProfiler_h */
