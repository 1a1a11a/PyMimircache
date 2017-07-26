//
//  FIFO.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef FIFO_H
#define FIFO_H


#include "cache.h" 

/* need add support for p and c type of data 
 
 */


#ifdef __cplusplus
extern "C"
{
#endif


struct FIFO_params{
    GHashTable *hashtable;
    GQueue *list;
};




extern gboolean fifo_check_element(struct_cache* cache, cache_line* cp);
extern gboolean fifo_add_element(struct_cache* cache, cache_line* cp);

extern void     __fifo_insert_element(struct_cache* fifo, cache_line* cp);
extern void     __fifo_update_element(struct_cache* fifo, cache_line* cp);
extern void     __fifo_evict_element(struct_cache* fifo, cache_line* cp);
extern void*    __fifo__evict_with_return(struct_cache* fifo, cache_line* cp);


extern void     fifo_destroy(struct_cache* cache);
extern void     fifo_destroy_unique(struct_cache* cache);
extern gint64 fifo_get_size(struct_cache *cache);


struct_cache* fifo_init(guint64 size, char data_type, int block_size, void* params);


#ifdef __cplusplus
}
#endif


#endif	/* FIFO_H */
