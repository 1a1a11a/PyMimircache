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

typedef struct{
    long long total_count;
    long long hit_count;        // this can be negative!!
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
    guint bin_size;
    GHashTable *prefetch_hashtable;
    GMutex mtx;             // prevent simultaneous write to progress
    guint64* progress;
};

struct HR_PE{
    guint64 real_cache_size[80];
    double HR[80];
    double PE[80];
    gint64 prefetch[80];
};

struct HR_PE_params{
    READER* reader;
    struct HR_PE* hrpe;
    struct_cache** caches;
};

struct HR_PE* get_HR_PE(READER* reader_in, guint64 size); 
void get_HR_PE_thread(gpointer data, gpointer user_data);

return_res** profiler(READER* reader_in, struct cache* cache_in, int num_of_threads_in, int bin_size_in, gint64 begin_pos, gint64 end_pos);
return_res** profiler_with_prefetch(READER* reader_in, struct_cache* cache_in, int num_of_threads_in, int bin_size_in, char* prefetch_file_loc, gint64 begin_pos, gint64 end_pos);

gdouble* LRU_evict_err_statistics(READER* reader_in, struct_cache* cache_in, guint64 time_interval);


#endif /* generalProfiler_h */
