//
//  LFU.c
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h"
#include "pqueue.h"
#include "LFU.h"


/** priority queue structs and def
 */


static inline int
cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
    return (next > curr);
}


static inline pqueue_pri_t
get_pri(void *a)
{
    return ((pq_node_t *) a)->pri;
}


static inline void
set_pri(void *a, pqueue_pri_t pri)
{
    ((pq_node_t *) a)->pri = pri;
}


static inline size_t
get_pos(void *a)
{
    return ((pq_node_t *) a)->pos;
}


static inline void
set_pos(void *a, size_t pos)
{
    ((pq_node_t *) a)->pos = pos;
}


//static void print_cacheline(struct_cache* LFU){
//    struct LFU_params* LFU_params = (struct LFU_params*)(LFU->cache_params);
//    if (!pqueue_peek(LFU_params->pq))
//        printf("pq size: %zu, hashtable size: %u, peek: EMPTY\n", pqueue_size(LFU_params->pq), g_hash_table_size(LFU_params->hashtable));
//    else
//        printf("pq size: %zu, hashtable size: %u, peek: %lld\n", pqueue_size(LFU_params->pq), g_hash_table_size(LFU_params->hashtable), ((pq_node_t*)pqueue_peek(LFU_params->pq))->long_item);
//}



inline void __LFU_insert_element(struct_cache* LFU, cache_line* cp){
    struct LFU_params* LFU_params = (struct LFU_params*)(LFU->cache_params);
    
    pq_node_t *node = g_new(pq_node_t, 1);
    gpointer key;
    if (cp->type == 'l'){
        key = (gpointer)g_new(gint64, 1);
        *(guint64*)key = *(guint64*)(cp->item_p);
    }
    else{
        key = (gpointer)g_strdup((gchar*)(cp->item_p));
    }
    node->data_type = cp->type;
    node->item = (gpointer)key;

    // we might need to not clear the freq dict when evict one element
//    gpointer gp = g_hash_table_lookup(LFU_params->hashtable, cp->item_p);
//    node->pri = GPOINTER_TO_UINT(gp)+1;
    
    node->pri = 1;
    
    
    pqueue_insert(LFU_params->pq, (void *)node);
    g_hash_table_insert (LFU_params->hashtable, (gpointer)key, (gpointer)node);
}


inline gboolean LFU_check_element(struct_cache* cache, cache_line* cp){
    return g_hash_table_contains(
                                 ((struct LFU_params*)(cache->cache_params))->hashtable,
                                 (gconstpointer)(cp->item_p)
                                 );
}


inline void __LFU_update_element(struct_cache* cache, cache_line* cp){
    struct LFU_params* LFU_params = (struct LFU_params*)(cache->cache_params);
    pq_node_t* node = (pq_node_t*) g_hash_table_lookup(LFU_params->hashtable, (gconstpointer)(cp->item_p));
    pqueue_change_priority(LFU_params->pq, node->pri+1, (void*)node);
}



inline void __LFU_evict_element(struct_cache* cache, cache_line* cp){
    struct LFU_params* LFU_params = (struct LFU_params*)(cache->cache_params);
    
    pq_node_t* node = (pq_node_t*) pqueue_pop(LFU_params->pq);
    
    
    g_hash_table_remove(LFU_params->hashtable, (gconstpointer)(node->item));
}




inline gboolean LFU_add_element(struct_cache* cache, cache_line* cp){
    struct LFU_params* LFU_params = (struct LFU_params*)(cache->cache_params);
    
    if (LFU_check_element(cache, cp)){
        __LFU_update_element(cache, cp);
        return TRUE;
    }
    else{
        __LFU_insert_element(cache, cp);
        if ( (long)g_hash_table_size( LFU_params->hashtable) > cache->core->size)
            __LFU_evict_element(cache, cp);
        return FALSE;
    }
}


inline void LFU_destroy(struct_cache* cache){
    struct LFU_params* LFU_params = (struct LFU_params*)(cache->cache_params);
    
    g_hash_table_destroy(LFU_params->hashtable);
    pqueue_free(LFU_params->pq);
    cache_destroy(cache);
}


inline void LFU_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in LFU, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    LFU_destroy(cache);

    
//    struct LFU_params* LFU_params = (struct LFU_params*)(cache->cache_params);
//    g_hash_table_destroy(LFU_params->hashtable);
//    pqueue_free(LFU_params->pq);
//    g_free(cache->cache_params);
//    g_free(cache->core);
//    g_free(cache);
}



struct_cache* LFU_init(guint64 size, char data_type, void* params){
    struct_cache* cache = cache_init(size, data_type);
    struct LFU_params* LFU_params = g_new0(struct LFU_params, 1);
    cache->cache_params = (void*) LFU_params;
    
    cache->core->type = e_LFU;
    cache->core->cache_init = LFU_init;
    cache->core->destroy = LFU_destroy;
    cache->core->destroy_unique = LFU_destroy_unique;
    cache->core->add_element = LFU_add_element;
    cache->core->check_element = LFU_check_element;
    
    
    if (data_type == 'l'){
        LFU_params->hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_g_key_value_destroyer, simple_g_key_value_destroyer);
    }
    
    else if (data_type == 'c'){
        LFU_params->hashtable = g_hash_table_new_full(g_str_hash, g_str_equal, simple_g_key_value_destroyer, simple_g_key_value_destroyer);
    }
    else{
        g_error("does not support given data type: %c\n", data_type);
    }
    
    LFU_params->pq = pqueue_init(size, cmp_pri, get_pri, set_pri, get_pos, set_pos);
    
    return cache;
}

