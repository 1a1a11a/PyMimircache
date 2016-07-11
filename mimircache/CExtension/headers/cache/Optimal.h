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
#include "pqueue.h"
#include "heatmap.h"





struct optimal_params{
    GHashTable *hashtable;
    pqueue_t *pq;
    GArray* next_access;
    guint64 ts;       // virtual time stamp
    READER* reader;
};


struct optimal_init_params{
    READER* reader;
    GArray* next_access;
    guint64 ts;
};






extern inline void __optimal_insert_element(struct_cache* optimal, cache_line* cp);

extern inline gboolean optimal_check_element(struct_cache* cache, cache_line* cp);

extern inline void __optimal_update_element(struct_cache* optimal, cache_line* cp);

extern inline void __optimal_evict_element(struct_cache* optimal, cache_line* cp);

extern inline gboolean optimal_add_element(struct_cache* cache, cache_line* cp);

extern inline void optimal_destroy(struct_cache* cache);
extern inline void optimal_destroy_unique(struct_cache* cache);

struct_cache* optimal_init(guint64 size, char data_type, void* params);



#endif