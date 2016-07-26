//
//  LRU.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef LRU_h
#define LRU_h


#include "cache.h" 




struct LRU_params{
    GHashTable *hashtable;
    GQueue *list;
};



extern inline void __LRU_insert_element(struct_cache* LRU, cache_line* cp);

extern inline gboolean LRU_check_element(struct_cache* cache, cache_line* cp);

extern inline void __LRU_update_element(struct_cache* LRU, cache_line* cp);

extern inline void __LRU_evict_element(struct_cache* LRU, cache_line* cp);

extern inline gboolean LRU_add_element(struct_cache* cache, cache_line* cp);


extern inline void LRU_destroy(struct_cache* cache);
extern inline void LRU_destroy_unique(struct_cache* cache);


struct_cache* LRU_init(guint64 size, char data_type, void* params);



extern inline void __LRU_remove_element(struct_cache* cache, void* data_to_remove);



#endif