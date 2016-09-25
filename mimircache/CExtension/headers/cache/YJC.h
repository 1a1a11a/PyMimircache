//
//  YJC.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef YJC_h
#define YJC_h


#include "cache.h" 

#define PREDICTION_LIMIT 18



struct YJC_params{
    struct_cache* LRU;
    struct_cache* LFU;
    struct_cache* prefetch;

    guint64 LRU_size;
    guint64 LFU_size;
    guint64 prefetch_size;
    
    gint64 ts;
    
    GHashTable* prefetch_hashtable;
    GHashTable* clustering_hashtable;
    GHashTable* element_hashtable;
    
    guint64 counter;
    
};

struct YJC_init_params{
    double LRU_percentage;
    double LFU_percentage;
    GHashTable* clustering_hashtable;
};


struct element_info{
    gint accessed_times;
    gint effective_accessed_times;
    gint64 last_access_time;
    gboolean evicted;
    gboolean already_prefetched; 
};

struct clustering_group{
    GSList* list;
    gint num_of_prediction;
    GArray* predictions;
    gint num_of_last_predictions;
    gint64 last_predictions[PREDICTION_LIMIT+2];
};


extern  void __YJC_insert_element(struct_cache* YJC, cache_line* cp);

extern  gboolean YJC_check_element(struct_cache* cache, cache_line* cp);

extern  void __YJC_update_element(struct_cache* YJC, cache_line* cp);

extern  void __YJC_evict_element(struct_cache* YJC, cache_line* cp);

extern  gboolean YJC_add_element(struct_cache* cache, cache_line* cp);

extern  void YJC_destroy(struct_cache* cache);
extern  void YJC_destroy_unique(struct_cache* cache);


struct_cache* YJC_init(guint64 size, char data_type, void* params);


#endif
