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





extern  void __LFU_insert_element(struct_cache* LFU, cache_line* cp);

extern  gboolean LFU_check_element(struct_cache* cache, cache_line* cp);

extern  void __LFU_update_element(struct_cache* LFU, cache_line* cp);

extern  void __LFU_evict_element(struct_cache* LFU, cache_line* cp);

extern  gboolean LFU_add_element(struct_cache* cache, cache_line* cp);

extern  void LFU_destroy(struct_cache* cache);
extern  void LFU_destroy_unique(struct_cache* cache);

struct_cache* LFU_init(guint64 size, char data_type, void* params);


extern void LFU_remove_element(struct_cache* cache, void* data_to_remove);
extern guint64 LFU_get_size(struct_cache* cache);
extern gpointer __LFU_evict_element_with_return(struct_cache* cache, cache_line* cp);



#endif
