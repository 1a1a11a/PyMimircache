//
//  ML.h
//  mimircache
//
//  Created by Juncheng on 7/15/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef ML_h
#define ML_h

#include <stdio.h>

#define INITIAL_TS 0
#define LRU_segments 5


#include "cache.h"
#include "LRU.h"
#include "pqueue.h"



struct ML_params{
    GHashTable *cache_hashtable;        // label -> pq_node_pointer
    GHashTable *ghost_hashtable;        // label -> gqueue of size K
    pqueue_t *pq;
    struct_cache* LRU[LRU_segments];

    guint64 ts;
};


typedef struct{
    guint64 freq;
    guint64 freq_clear;
    gdouble avg_freq;
    gint64 rd;
    gint64 rd_2;            // second to last history
    gdouble request_rate;
    gdouble cold_miss_rate;
    
    
    
}ML_hashtable_value_struct;



extern  void __ML_insert_element(struct_cache* cache, cache_line* cp);

extern  gboolean ML_check_element(struct_cache* cache, cache_line* cp);

extern  void __ML_update_element(struct_cache* cache, cache_line* cp);

extern  void __ML_evict_element(struct_cache* cache);

extern  gboolean ML_add_element(struct_cache* cache, cache_line* cp);

extern  void ML_destroy(struct_cache* cache);
extern  void ML_destroy_unique(struct_cache* cache);

extern  ML_hashtable_value_struct* report_feature_add_element(struct_cache* cache, cache_line* cp);


struct_cache* ML_init(guint64 size, char data_type, void* params);






#endif /* ML_h */
