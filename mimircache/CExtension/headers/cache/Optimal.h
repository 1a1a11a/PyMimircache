//
//  optimal.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef optimal_h
#define optimal_h


#include "cache.h"

///** priority data type */
//typedef unsigned long long pqueue_pri_t;

#include "pqueue.h"
#include "heatmap.h"


/* need add support for p and c type of data
 
 */




struct optimal_params{
    GHashTable *hashtable;
    pqueue_t *pq;
    GArray* next_access;
    long long ts;       // virtual time stamp
    READER* reader;
};


struct optimal_init_params{
    READER* reader;
    GArray* next_access;
    long long ts;
};






extern inline void __optimal_insert_element_long(struct_cache* optimal, cache_line* cp);

extern inline gboolean optimal_check_element_long(struct_cache* cache, cache_line* cp);

extern inline void __optimal_update_element_long(struct_cache* optimal, cache_line* cp);

extern inline void __optimal_evict_element(struct_cache* optimal);

extern inline gboolean optimal_add_element_long(struct_cache* cache, cache_line* cp);

extern inline void optimal_destroy(struct_cache* cache);
extern inline void optimal_destroy_unique(struct_cache* cache);

struct_cache* optimal_init(long long size, char data_type, void* params);



#endif