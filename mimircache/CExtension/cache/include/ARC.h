//
//  ARC.h
//  mimircache
//
//  Created by Juncheng on 2/12/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#ifndef ARC_h
#define ARC_h


#include "cache.h"
#include "LRU.h" 
#include "LFU.h"


#ifdef __cplusplus
extern "C"
{
#endif


// by default, the ghost list size is 10 times the orginal cache size

typedef struct ARC_params{
    struct_cache* LRU1;         // normal LRU segment
    struct_cache* LRU1g;        // ghost list for normal LRU segment
    struct_cache* LRU2;         // normal LRU segement for items accessed more than once
    struct_cache* LRU2g;        // ghost list for normal LFU segment
    gint32 ghost_list_factor;  // size(ghost_list)/size(cache),
                                // by default, the ghost list size is
                                // 10 times the orginal cache size
    gint64 size1;             // size for segment 1
    gint64 size2;             // size for segment 2
}ARC_params_t;


typedef struct ARC_init_params{
    gint32 ghost_list_factor;
} ARC_init_params_t;



extern gboolean ARC_check_element(struct_cache* cache, cache_line* cp);
extern gboolean ARC_add_element(struct_cache* cache, cache_line* cp);


extern void     __ARC_insert_element(struct_cache* ARC, cache_line* cp);
extern void     __ARC_update_element(struct_cache* ARC, cache_line* cp);
extern void     __ARC_evict_element(struct_cache* ARC, cache_line* cp);
extern void*    __ARC__evict_with_return(struct_cache* ARC, cache_line* cp);


extern void     ARC_destroy(struct_cache* cache);
extern void     ARC_destroy_unique(struct_cache* cache);


struct_cache*   ARC_init(guint64 size, char data_type, int block_size, void* params);


extern void     ARC_remove_element(struct_cache* cache, void* data_to_remove);
extern gint64 ARC_get_size(struct_cache* cache);


#ifdef __cplusplus
}
#endif


#endif  /* ARC_H */ 
