//
//  optimal.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef optimal_h
#define optimal_h


#include "cache.h"
#include "pqueue.h"
#include "heatmap.h"



#ifdef __cplusplus
extern "C"
{
#endif


struct optimal_params{
    GHashTable *hashtable;
    pqueue_t *pq;
    GArray* next_access;
    guint64 ts;       // virtual time stamp
    reader_t* reader;
};


struct optimal_init_params{
    reader_t* reader;
    GArray* next_access;
    guint64 ts;
};

typedef struct optimal_params       optimal_params_t;
typedef struct optimal_init_params  optimal_init_params_t;




extern  void __optimal_insert_element(struct_cache* optimal, cache_line* cp);

extern  gboolean optimal_check_element(struct_cache* cache, cache_line* cp);

extern  void __optimal_update_element(struct_cache* optimal, cache_line* cp);

extern  void __optimal_evict_element(struct_cache* optimal, cache_line* cp);

extern  gboolean optimal_add_element(struct_cache* cache, cache_line* cp);
extern  gboolean optimal_add_element_only(struct_cache* cache, cache_line* cp);

extern  void optimal_destroy(struct_cache* cache);
extern  void optimal_destroy_unique(struct_cache* cache);

struct_cache* optimal_init(guint64 size, char data_type, int block_size, void* params);


#ifdef __cplusplus
}
#endif


#endif
