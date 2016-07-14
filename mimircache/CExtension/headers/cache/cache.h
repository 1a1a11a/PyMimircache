//
//  cache.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef cache_h
#define cache_h

#include <stdio.h>
#include <glib.h>
#include "reader.h"
#include <stdlib.h> 
#include <string.h> 
#include "glib_related.h"
#include "const.h"




typedef enum{
    e_LRU,
    e_LFU,
    e_Optimal,
    e_FIFO,
    e_LRU_K,
    e_LRU_LFU,
    e_LRU_dataAware,
}cache_type;


struct cache_core{
    cache_type type;
    long size;
    char data_type;     // l, c
    long long hit_count;
    long long miss_count;
    void* cache_init_params;
    struct cache* (*cache_init)(guint64, char, void*);
    void (*destroy)(struct cache* );
    void (*destroy_unique)(struct cache* );
    gboolean (*add_element)(struct cache*, cache_line* cp);
    gboolean (*check_element)(struct cache*, cache_line* cp);
    
    
    int cache_debug_level;  // 0 not debug, 1: prepare oracle, 2: compare to oracle
    void* oracle; 
    void* eviction_array;           // Optimal Eviction Array, either guint64* or char**
    guint64 eviction_array_len;
    guint64 evict_err;      // used for counting
    struct break_point* bp; // break points, same as the one in reader, just one more pointer
    guint64 bp_pos;         // the current location in bp->array
    gdouble* evict_err_array;       // in each time interval, the eviction error array 
};


typedef struct cache{
    struct cache_core *core;
    void* cache_params;
}struct_cache;



//#define core->type type;
//#define core->size size;
//#define data_type core->data_type;
//#define hit_count core->hit_count;
//#define miss_count core->miss_count;
//#define cache_init_params core->cache_init_params;
//#define cache_init core->cache_init;
//#define destroy core->destroy;
//#define destroy_unique core->destroy_unique;
//#define add_element core->add_element;
//#define check_element core->check_element;




struct_cache* cache_init(long long size, char data_type);
void cache_destroy(struct_cache* cache);
void cache_destroy_unique(struct_cache* cache); 





#endif /* cache_h */
