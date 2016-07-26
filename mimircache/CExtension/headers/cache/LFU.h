//
//  LFU.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef LFU_h
#define LFU_h


#include "cache.h"
#include "pqueue.h"



struct LFU_params{
    GHashTable *hashtable;
    pqueue_t *pq;
};





extern inline void __LFU_insert_element(struct_cache* LFU, cache_line* cp);

extern inline gboolean LFU_check_element(struct_cache* cache, cache_line* cp);

extern inline void __LFU_update_element(struct_cache* LFU, cache_line* cp);

extern inline void __LFU_evict_element(struct_cache* LFU, cache_line* cp);

extern inline gboolean LFU_add_element(struct_cache* cache, cache_line* cp);

extern inline void LFU_destroy(struct_cache* cache);
extern inline void LFU_destroy_unique(struct_cache* cache);

struct_cache* LFU_init(guint64 size, char data_type, void* params);



#endif