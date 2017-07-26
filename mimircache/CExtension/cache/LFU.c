//
//  LFU.c
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

/* this module uses priority queue to order cache lines, 
 * which is O(logN) at each request, 
 * but this approach saves some memory compared to the other approach, 
 * which uses a hashmap and linkedlist and gives O(1) at each request 
 *
 * this LFU clear the frequenct of an item after evicting from cache 
 * when there are more than one items with the smallest freq, the behavior is 
 * LRU, but can be tuned to MRU, Random or unstable-pq-decided  
 */ 


#include "LFU.h"


#ifdef __cplusplus
extern "C"
{
#endif


/********************* priority queue structs and def ***********************/
static inline int cmp_pri(pqueue_pri_t next, pqueue_pri_t curr){
    /* the one with smallest priority is poped out first */
    if (next.pri1 == curr.pri1)
        return (next.pri2 > curr.pri2);         // LRU
    else
        return (next.pri1 > curr.pri1);
}


static inline pqueue_pri_t get_pri(void *a){
    return ((pq_node_t *) a)->pri;
}


static inline void set_pri(void *a, pqueue_pri_t pri){
    ((pq_node_t *) a)->pri = pri;
}


static inline size_t get_pos(void *a){
    return ((pq_node_t *) a)->pos;
}


static inline void set_pos(void *a, size_t pos){
    ((pq_node_t *) a)->pos = pos;
}

/********************************** LFU ************************************/
void __LFU_insert_element(struct_cache* LFU, cache_line* cp){
    LFU_params_t* LFU_params = (LFU_params_t*)(LFU->cache_params);
    
    pq_node_t *node = g_new(pq_node_t, 1);
    gpointer key;
    if (cp->type == 'l'){
        key = (gpointer)g_new(guint64, 1);
        *(guint64*)key = *(guint64*)(cp->item_p);
    }
    else{
        key = (gpointer)g_strdup((gchar*)(cp->item_p));
    }
    node->data_type = cp->type;
    node->item = (gpointer)key;

    // frequency not cleared after eviction
//    gpointer gp = g_hash_table_lookup(LFU_params->hashtable, cp->item_p);
//    node->pri = GPOINTER_TO_UINT(gp)+1;
    
    /* node priority is set to one for first time,
     * frequency cleared after eviction
     */
    node->pri.pri1 = 1;
    node->pri.pri2 = cp->ts;
    
    
    pqueue_insert(LFU_params->pq, (void *)node);
    g_hash_table_insert (LFU_params->hashtable, (gpointer)key, (gpointer)node);
}


gboolean LFU_check_element(struct_cache* cache, cache_line* cp){
    return g_hash_table_contains(
                ((LFU_params_t*)(cache->cache_params))->hashtable,
                (gconstpointer)(cp->item_p));
}


void __LFU_update_element(struct_cache* cache, cache_line* cp){
    LFU_params_t* LFU_params = (LFU_params_t*)(cache->cache_params);
    pq_node_t* node = (pq_node_t*)
                        g_hash_table_lookup(LFU_params->hashtable,
                                            (gconstpointer)(cp->item_p));
    pqueue_pri_t pri;
    pri.pri1 = node->pri.pri1 + 1;
    pri.pri2 = cp->ts;
    pqueue_change_priority(LFU_params->pq, pri, (void*)node);
}



void __LFU_evict_element(struct_cache* cache, cache_line* cp){
    LFU_params_t* LFU_params = (LFU_params_t*)(cache->cache_params);
    pq_node_t* node = (pq_node_t*) pqueue_pop(LFU_params->pq);
    g_hash_table_remove(LFU_params->hashtable, (gconstpointer)(node->item));
}


gpointer __LFU__evict_with_return(struct_cache* cache, cache_line* cp){
    LFU_params_t* LFU_params = (LFU_params_t*)(cache->cache_params);
    
    pq_node_t* node = (pq_node_t*) pqueue_pop(LFU_params->pq);
    gpointer evicted_key;
    if (cp->type == 'l'){
        evicted_key = (gpointer)g_new(guint64, 1);
        *(guint64*)evicted_key = *(guint64*)(node->item);
    }
    else{
        evicted_key = (gpointer)g_strdup((gchar*)node->item);
    }
    g_hash_table_remove(LFU_params->hashtable, (gconstpointer)(node->item));
    return evicted_key;
}



gboolean LFU_add_element(struct_cache* cache, cache_line* cp){
    LFU_params_t* LFU_params = (LFU_params_t*)(cache->cache_params);
    
    if (LFU_check_element(cache, cp)){
        __LFU_update_element(cache, cp);
        return TRUE;
    }
    else{
        __LFU_insert_element(cache, cp);
        if ( (long)g_hash_table_size(LFU_params->hashtable) > cache->core->size)
            __LFU_evict_element(cache, cp);
        return FALSE;
    }
}


gboolean LFU_add_element_only(struct_cache* cache, cache_line* cp){
    return LFU_add_element(cache, cp);
}


gboolean LFU_add_element_with_size(struct_cache* cache, cache_line* cp){
    int i, n = 0;
    gint64 original_lbn = *(gint64*)(cp->item_p);
    gboolean ret_val;
    
    if (cache->core->block_unit_size != 0 && cp->disk_sector_size != 0){
        *(gint64*)(cp->item_p) = (gint64) (*(gint64*)(cp->item_p) *
                                           cp->disk_sector_size /
                                           cache->core->block_unit_size);
        n = (int)ceil((double) cp->size/cache->core->block_unit_size);
    }
    
    ret_val = LFU_add_element(cache, cp);
    
    
    if (cache->core->block_unit_size != 0 && cp->disk_sector_size != 0){
        if (cache->core->block_unit_size != 0){
            for (i=0; i<n-1; i++){
                (*(guint64*)(cp->item_p)) ++;
                LFU_add_element_only(cache, cp);
            }
        }
    }
    
    *(gint64*)(cp->item_p) = original_lbn;
    return ret_val;
    
}



 void LFU_destroy(struct_cache* cache){
    LFU_params_t* LFU_params = (LFU_params_t*)(cache->cache_params);
    
    g_hash_table_destroy(LFU_params->hashtable);
    pqueue_free(LFU_params->pq);
    cache_destroy(cache);
}


 void LFU_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in LFU, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    LFU_destroy(cache);
}



struct_cache* LFU_init(guint64 size, char data_type, int block_size, void* params){
    struct_cache* cache = cache_init(size, data_type, block_size);
    LFU_params_t* LFU_params = g_new0(LFU_params_t, 1);
    cache->cache_params = (void*) LFU_params;
    
    cache->core->type               =   e_LFU;
    cache->core->cache_init         =   LFU_init;
    cache->core->destroy            =   LFU_destroy;
    cache->core->destroy_unique     =   LFU_destroy_unique;
    cache->core->add_element        =   LFU_add_element;
    cache->core->check_element      =   LFU_check_element;
    cache->core->add_element_only   =   LFU_add_element;

    cache->core->__insert_element   =   __LFU_insert_element;
    cache->core->__update_element   =   __LFU_update_element;
    cache->core->__evict_element    =   __LFU_evict_element;
    cache->core->__evict_with_return=   __LFU__evict_with_return;
    cache->core->get_size           =   LFU_get_size;
    cache->core->cache_init_params  =   NULL;
    cache->core->add_element_only   =   LFU_add_element_only;
    cache->core->add_element_withsize = LFU_add_element_with_size; 
    
    
    if (data_type == 'l'){
        LFU_params->hashtable =
            g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                  simple_g_key_value_destroyer,
                                  simple_g_key_value_destroyer);
    }
    
    else if (data_type == 'c'){
        LFU_params->hashtable =
            g_hash_table_new_full(g_str_hash, g_str_equal,
                                  simple_g_key_value_destroyer,
                                  simple_g_key_value_destroyer);
    }
    else{
        ERROR("does not support given data type: %c\n", data_type);
    }
    
    LFU_params->pq = pqueue_init(size, cmp_pri, get_pri,
                                 set_pri, get_pos, set_pos);
    
    return cache;
}


gint64 LFU_get_size(struct_cache* cache){
    LFU_params_t* LFU_params = (LFU_params_t*)(cache->cache_params);
    return (guint64) g_hash_table_size(LFU_params->hashtable);
}

void LFU_remove_element(struct_cache* cache, void* data_to_remove){
    LFU_params_t* LFU_params = (LFU_params_t*)(cache->cache_params);
    pq_node_t* node = (pq_node_t*)
        g_hash_table_lookup(LFU_params->hashtable,
                            (gconstpointer)data_to_remove);
    pqueue_remove(LFU_params->pq, (void*)node);
    g_hash_table_remove(LFU_params->hashtable,
                        (gconstpointer)data_to_remove);
}



#ifdef __cplusplus
}
#endif