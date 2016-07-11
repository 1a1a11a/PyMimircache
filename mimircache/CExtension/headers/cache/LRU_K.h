

//
//  LRU_K.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef LRUK_H
#define LRUK_H

#define INITIAL_TS 0 


#include "cache.h"
#include "pqueue.h" 



struct LRU_K_params{
    GHashTable *cache_hashtable;        // label -> pq_node_pointer
    GHashTable *ghost_hashtable;        // label -> gqueue of size K
    pqueue_t *pq;                       
    guint K;
    guint maxK;
    guint64 ts;
};

struct LRU_K_init_params{
    int K;
    int maxK;
};



extern inline void __LRU_K_insert_element(struct_cache* LRU_K, cache_line* cp);

extern inline gboolean LRU_K_check_element(struct_cache* cache, cache_line* cp);

extern inline void __LRU_K_update_element(struct_cache* LRU_K, cache_line* cp);

extern inline void __LRU_K_evict_element(struct_cache* LRU_K);

extern inline gboolean LRU_K_add_element(struct_cache* cache, cache_line* cp);

extern inline void LRU_K_destroy(struct_cache* cache);
extern inline void LRU_K_destroy_unique(struct_cache* cache);


struct_cache* LRU_K_init(guint64 size, char data_type, void* params);


#endif