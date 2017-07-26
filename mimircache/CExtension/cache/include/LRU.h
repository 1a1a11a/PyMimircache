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


#ifdef __cplusplus
extern "C"
{
#endif


struct LRU_params{
    GHashTable *hashtable;
    GQueue *list;
    gint64 ts;              // this only works when add_element is called 
};

typedef struct LRU_params LRU_params_t; 




extern gboolean LRU_check_element(struct_cache* cache, cache_line* cp);
extern gboolean LRU_add_element(struct_cache* cache, cache_line* cp);


extern void     __LRU_insert_element(struct_cache* LRU, cache_line* cp);
extern void     __LRU_update_element(struct_cache* LRU, cache_line* cp);
extern void     __LRU_evict_element(struct_cache* LRU, cache_line* cp);
extern void*    __LRU__evict_with_return(struct_cache* LRU, cache_line* cp);


extern void     LRU_destroy(struct_cache* cache);
extern void     LRU_destroy_unique(struct_cache* cache);


struct_cache*   LRU_init(guint64 size, char data_type, int block_size, void* params);


extern void     LRU_remove_element(struct_cache* cache, void* data_to_remove);
extern gint64 LRU_get_size(struct_cache* cache);


#ifdef __cplusplus
}
#endif


#endif	/* LRU_H */
