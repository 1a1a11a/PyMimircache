//
//  SLRU.h
//  mimircache
//
//  Created by Juncheng on 2/12/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#ifndef SLRU_h
#define SLRU_h


#include "cache.h" 
#include "LRU.h" 


#ifdef __cplusplus
extern "C"
{
#endif


typedef struct SLRU_params{
    cache_t** LRUs;
    int N_segments;
    uint64_t *current_sizes; 
}SLRU_params_t;


typedef struct SLRU_init_params{
    int N_segments;
}SLRU_init_params_t;




extern gboolean SLRU_check_element(struct_cache* cache, cache_line* cp);
extern gboolean SLRU_add_element(struct_cache* cache, cache_line* cp);


extern void     __SLRU_insert_element(struct_cache* SLRU, cache_line* cp);
extern void     __SLRU_update_element(struct_cache* SLRU, cache_line* cp);
extern void     __SLRU_evict_element(struct_cache* SLRU, cache_line* cp);
extern void*    __SLRU__evict_with_return(struct_cache* SLRU, cache_line* cp);


extern void     SLRU_destroy(struct_cache* cache);
extern void     SLRU_destroy_unique(struct_cache* cache);


struct_cache*   SLRU_init(guint64 size, char data_type, int block_size, void* params);


extern void     SLRU_remove_element(struct_cache* cache, void* data_to_remove);
extern gint64 SLRU_get_size(struct_cache* cache);


#ifdef __cplusplus
}
#endif


#endif	/* SLRU_H */ 
