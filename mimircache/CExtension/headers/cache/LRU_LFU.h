//
//  LRU_LFU.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef LRU_LFU_h
#define LRU_LFU_h


#include "cache.h" 
#include "LFU.h" 
#include "LRU.h"




struct LRU_LFU_params{
    struct_cache* LRU;
    struct_cache* LFU;
//    struct LRU_params* LRU;
//    struct LFU_params* LFU;
    double LRU_percentage;
};

struct LRU_LFU_init_params{
    double LRU_percentage;
};



extern inline void __LRU_LFU_insert_element(struct_cache* LRU_LFU, cache_line* cp);

extern inline gboolean LRU_LFU_check_element(struct_cache* cache, cache_line* cp);

extern inline void __LRU_LFU_update_element(struct_cache* LRU_LFU, cache_line* cp);

extern inline void __LRU_LFU_evict_element(struct_cache* LRU_LFU, cache_line* cp);

extern inline gboolean LRU_LFU_add_element(struct_cache* cache, cache_line* cp);

extern inline void LRU_LFU_destroy(struct_cache* cache);
extern inline void LRU_LFU_destroy_unique(struct_cache* cache);


struct_cache* LRU_LFU_init(guint64 size, char data_type, void* params);


#endif