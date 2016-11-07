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


//#define PG_prev(cache, x) (PG_lookup((cache), ((struct PG_page*)x)->block_numer - 1))
//#define PG_last(cache, x) (PG_lookup((cache), ((struct PG_page*)x)->last_block_numer))
//#define PG_isLast(x) (((struct PG_page*)x)->block_number == ((struct PG_page*)x)->last_block_number) 

#define PG_getPage(x) ((struct PG_Page*) g_hashtable_lookup())



struct PG_params{
    struct_cache *cache;
    
    GHashTable *graph;
    gdouble prefetch_threshold;
    
    gint64 num_of_prefetch;
    gint64 num_of_hit;
    GHashTable *prefetched; 
};


struct PG_init_params{
    gdouble prefetch_threshold;
};



extern void __PG_insert_element(struct_cache* PG, cache_line* cp);

extern  gboolean PG_check_element_int(struct_cache* cache, gint64 block);
extern  gboolean PG_check_element(struct_cache* cache, cache_line* cp);

extern struct PG_page* __PG_update_element_int(struct_cache* PG, gint64 block);
extern void __PG_update_element(struct_cache* PG, cache_line* cp);

extern  void __PG_evict_element(struct_cache* PG, cache_line* cp);
extern gpointer __PG_evict_element_with_return(struct_cache* PG, cache_line* cp);

extern  gboolean PG_add_element(struct_cache* cache, cache_line* cp);
extern  gboolean PG_add_element_no_eviction(struct_cache* cache, cache_line* cp);


extern  void PG_destroy(struct_cache* cache);
extern  void PG_destroy_unique(struct_cache* cache);


struct_cache* PG_init(guint64 size, char data_type, void* params);


extern  void PG_remove_element(struct_cache* cache, void* data_to_remove);
extern gpointer __PG_evict_element_with_return(struct_cache* PG, cache_line* cp);
extern guint64 PG_get_size(struct_cache* cache);



#endif
