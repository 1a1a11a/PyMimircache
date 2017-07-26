//
//  LFU_fast.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef LFU_FAST_H
#define LFU_FAST_H


#include "cache.h"


#ifdef __cplusplus
extern "C"
{
#endif


typedef struct LFU_fast_params{
    GHashTable *hashtable;      // key -> glist 
    GQueue *main_list;
    gint min_freq; 
}LFU_fast_params_t;


// main list is a doubly linkedlist sort by frequency
typedef struct main_list_node_data{
    gint freq;
    GQueue *queue;   // linked to branch list
}main_list_node_data_t;


// branch list is the list of items with same freq, sorted in LRU
typedef struct branch_list_node_data{
    gpointer key;
    GList *main_list_node;
    
}branch_list_node_data_t;



extern gboolean LFU_fast_check_element(struct_cache* cache, cache_line* cp);
extern gboolean LFU_fast_add_element(struct_cache* cache, cache_line* cp);


extern void     __LFU_fast_insert_element(struct_cache* LFU_fast, cache_line* cp);
extern void     __LFU_fast_update_element(struct_cache* LFU_fast, cache_line* cp);
extern void     __LFU_fast_evict_element(struct_cache* LFU_fast, cache_line* cp);
extern void*    __LFU_fast__evict_with_return(struct_cache* cache, cache_line* cp);


extern void     LFU_fast_destroy(struct_cache* cache);
extern void     LFU_fast_destroy_unique(struct_cache* cache);

struct_cache*   LFU_fast_init(guint64 size, char data_type, int block_size, void* params);


extern void     LFU_fast_remove_element(struct_cache* cache, void* data_to_remove);
extern gint64 LFU_fast_get_size(struct_cache* cache);


#ifdef __cplusplus
}
#endif


#endif	/* LFU_FAST_H */
