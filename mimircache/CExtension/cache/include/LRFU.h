////
////  LRFU.h
////  mimircache
////
////  Created by Juncheng on 2/12/17.
////  Copyright © 2017 Juncheng. All rights reserved.
////
//
//#ifndef LRFU_h
//#define LRFU_h
//
//
//#include "cache.h" 
//#include "pqueue.h"
//
//
///* because priority queue only uses integer as priority, this is
// *  the factor used to change fractional priority into integer 
// */
//#define LRFU_PRI_BASE 10000000
//
//typedef struct LRFU_params{
//    GHashTable *hashtable;
//    pqueue_t *pq;
//}LRFU_params_t;
//
//
//typedef struct LRFU_init_params{
//    int t;
//}LRFU_init_params_t;
//
//
//
//
//
//
//extern gboolean LRFU_check_element(struct_cache* cache, cache_line* cp);
//extern gboolean LRFU_add_element(struct_cache* cache, cache_line* cp);
//
//
//extern void     __LRFU_insert_element(struct_cache* LRFU, cache_line* cp);
//extern void     __LRFU_update_element(struct_cache* LRFU, cache_line* cp);
//extern void     __LRFU_evict_element(struct_cache* LRFU, cache_line* cp);
//extern void*    __LRFU__evict_with_return(struct_cache* LRFU, cache_line* cp);
//
//
//extern void     LRFU_destroy(struct_cache* cache);
//extern void     LRFU_destroy_unique(struct_cache* cache);
//
//
//struct_cache*   LRFU_init(guint64 size, char data_type, void* params);
//
//
//extern void     LRFU_remove_element(struct_cache* cache, void* data_to_remove);
//extern uint64_t LRFU_get_size(struct_cache* cache);
//
//
//
//#endif
