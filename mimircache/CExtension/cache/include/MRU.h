//
//  MRU.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef MRU_h
#define MRU_h

#include <stdio.h>
#include "cache.h"


#ifdef __cplusplus
extern "C"
{
#endif


struct MRU_params{
    GHashTable *hashtable;
};





extern  void __MRU_insert_element(struct_cache* MRU, cache_line* cp);

extern  gboolean MRU_check_element(struct_cache* cache, cache_line* cp);

extern  void __MRU_update_element(struct_cache* MRU, cache_line* cp);

extern  void __MRU_evict_element(struct_cache* MRU, cache_line* cp);

extern  gboolean MRU_add_element(struct_cache* cache, cache_line* cp);

extern  void MRU_destroy(struct_cache* cache);
extern  void MRU_destroy_unique(struct_cache* cache);

struct_cache* MRU_init(guint64 size, char data_type, int block_size, void* params);


#ifdef __cplusplus
}
#endif

 
#endif /* MRU_h */
