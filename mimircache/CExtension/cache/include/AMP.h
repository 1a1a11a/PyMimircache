//
//  AMP.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


/** since this is sequence based prefetching, we will use gint64 for block number **/



#ifndef AMP_h
#define AMP_h


#include "cache.h" 

#ifdef __cplusplus
extern "C"
{
#endif


//#define AMP_prev(cache, x) (AMP_lookup((cache), ((struct AMP_page*)x)->block_numer - 1))
//#define AMP_last(cache, x) (AMP_lookup((cache), ((struct AMP_page*)x)->last_block_numer))
//#define AMP_isLast(x) (((struct AMP_page*)x)->block_number == ((struct AMP_page*)x)->last_block_number) 

#define AMP_getPage(x) ((struct AMP_Page*) g_hashtable_lookup())



struct AMP_params{
    GHashTable *hashtable;
    GQueue *list;
    
    int APT;
    int read_size;
    int p_threshold;
    int K; 
    
    GHashTable *last_hashset;
    
    
    gint64 num_of_prefetch;
    gint64 num_of_hit;
    GHashTable *prefetched; 
};

struct AMP_page{
    gint64 block_number;
    gint64 last_block_number;
    gboolean accessed;
    gboolean tag;
    gboolean old;
    gint16 p;                   // prefetch degree
    gint16 g;                   // trigger distance 

    gint size;
};

struct AMP_init_params{
    int APT;
    int read_size;
    int p_threshold;
    int K;
};



extern gboolean     AMP_check_element_int(struct_cache* cache, gint64 block);
extern gboolean     AMP_check_element(struct_cache* cache, cache_line* cp);

extern gboolean     AMP_add_element(struct_cache* cache, cache_line* cp);
extern gboolean     AMP_add_element_only(struct_cache* cache, cache_line* cp);
extern gboolean     AMP_add_element_only_no_eviction(struct_cache* AMP, cache_line* cp); 
extern gboolean     AMP_add_element_no_eviction(struct_cache* cache, cache_line* cp);

extern struct AMP_page* __AMP_update_element_int(struct_cache* AMP, gint64 block);
extern void         __AMP_update_element(struct_cache* AMP, cache_line* cp);

extern struct AMP_page* __AMP_insert_element_int(struct_cache* AMP, gint64 block);
extern void         __AMP_insert_element(struct_cache* AMP, cache_line* cp);


extern void         __AMP_evict_element(struct_cache* AMP, cache_line* cp);
extern void*        __AMP__evict_with_return(struct_cache* AMP, cache_line* cp);



extern void         AMP_destroy(struct_cache* cache);
extern void         AMP_destroy_unique(struct_cache* cache);


struct_cache*       AMP_init(guint64 size, char data_type, int block_size, void* params);


extern void         AMP_remove_element(struct_cache* cache, void* data_to_remove);
extern gint64     AMP_get_size(struct_cache* cache);


#ifdef __cplusplus
}
#endif


#endif /* AMP_H */ 
