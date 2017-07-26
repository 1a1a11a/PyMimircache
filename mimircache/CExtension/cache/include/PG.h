//
//  PG.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


/** since this is sequence based prefetching, we will use gint64 for block number **/



#ifndef PG_h
#define PG_h


#include "cache.h" 
#include "const.h" 
#include "pqueue.h"

#ifdef __cplusplus
extern "C"
{
#endif


#define PG_getPage(x) ((struct PG_Page*) g_hashtable_lookup())

// #define get_Nth_past_request_c(PG_params, n)         ((char**)((PG_params)->past_requests))[(n)]
#define get_Nth_past_request_c(PG_params, n, des)    strcpy((des), ((char**)((PG_params)->past_requests))[(n)])

#define get_Nth_past_request_l(PG_params, n)         ((guint64*)((PG_params)->past_requests))[(n)]
#define set_Nth_past_request_c(PG_params, n, v)      strcpy(((char**)((PG_params)->past_requests))[(n)], (char*)(v))
#define set_Nth_past_request_l(PG_params, n, v)      ((guint64*)((PG_params)->past_requests))[(n)] = (v)



typedef struct{
    struct_cache    *cache;
    
    GHashTable      *graph;                  // key -> graphNode_t 

    void            *past_requests;          // past requests, using array instead of queue to avoid frequent memory allocation
    uint32_t        past_request_pointer;
    
    double          prefetch_threshold;
    uint8_t         lookahead;
    
    gint64          num_of_prefetch;
    gint64          num_of_hit;
    GHashTable      *prefetched;
    uint64_t        meta_data_size;          // unit byte
    uint64_t        init_size;
    double          max_meta_data;
    uint            block_size;
    gboolean        stop_recording;
}PG_params_t;


typedef struct{
    GHashTable *graph;              // key -> pq_node_t
    pqueue_t *pq;
    uint64_t total_count;
}graphNode_t;


typedef struct{
    double prefetch_threshold;
    uint8_t lookahead;
    char* cache_type;
    double max_meta_data;
    uint block_size; 
}PG_init_params_t;




extern gboolean PG_check_element(struct_cache* cache, cache_line* cp);
extern gboolean PG_add_element(struct_cache* cache, cache_line* cp);


extern void     __PG_insert_element(struct_cache* PG, cache_line* cp);
extern void     __PG_update_element(struct_cache* PG, cache_line* cp);
extern void     __PG_evict_element(struct_cache* PG, cache_line* cp);
extern void*    __PG_evict_with_return(struct_cache* PG, cache_line* cp);


extern void     PG_destroy(struct_cache* cache);
extern void     PG_destroy_unique(struct_cache* cache);


struct_cache*   PG_init(guint64 size, char data_type, int block_size, void* params);


extern void     PG_remove_element(struct_cache* cache, void* data_to_remove);
extern gint64 PG_get_size(struct_cache* cache);


#ifdef __cplusplus
}
#endif


#endif  /* PG_H */ 
