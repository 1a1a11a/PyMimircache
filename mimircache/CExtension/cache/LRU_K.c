//
//  LRU_K.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h"
#include "LRU_K.h"

/* need to add support for p and c type of data
   need to control the size of ghost_hashtable
 
 */



/** priority queue structs and def
 */




/* changed to make node with small priority on the top */
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


//static void print_GQ(gpointer data, gpointer user_data){
//    printf("%lu\t", *(guint64*)data);
//}





/* Jason: try to reuse the key from ghost_hashtable for better memory efficiency */

inline void __LRU_K_insert_element(struct_cache* LRU_K, cache_line* cp){
    /** update request is done at checking element, 
     * now insert request into cache_hashtable and pq 
     *
     **/
    
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(LRU_K->cache_params);
    gpointer key = NULL;
    
    pq_node_t* node = g_new(pq_node_t, 1);
    node->data_type = cp->type;
    
    GQueue* queue = NULL;
    g_hash_table_lookup_extended(LRU_K_params->ghost_hashtable,
                                 (gconstpointer)(cp->item_p),
                                 &key, (gpointer)&queue);
    
    pqueue_pri_t pri;
    if (queue->length < LRU_K_params->K){
        pri = INITIAL_TS;
    }
    else
        pri = *(guint64*) g_queue_peek_nth(queue, LRU_K_params->K-1);
    
    node->item = key;
    node->pri = pri;
    pqueue_insert(LRU_K_params->pq, (void*)node);
    g_hash_table_insert(LRU_K_params->cache_hashtable, (gpointer)key, (gpointer)node);
}


inline gboolean LRU_K_check_element(struct_cache* cache, cache_line* cp){
    /** check whether request is in the cache_hashtable, 
     * then update ghost_hashtable and pq accordingly, 
     * if in ghost_hashtable, then update it,
     * else create a new entry in ghost_hashtable 
     * the size of ghost_hashtable is maintained separately
     * if in cache_hashtable, update pq by updating priority value
     * if not in cache_hashtable, do nothing
     **/
    
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(cache->cache_params);
    GQueue* queue = g_hash_table_lookup( LRU_K_params->ghost_hashtable, (gconstpointer)(cp->item_p) );
    if (queue == NULL){
        
        /* need to insert the new element into ghost */
        gpointer key;
        if (cp->type == 'l'){
            key = (gpointer)g_new(gint64, 1);
            *(guint64*)key = *(guint64*)(cp->item_p);
        }
        else{
            key = (gpointer)g_strdup((gchar*)(cp->item_p));
        }
        
        queue = g_queue_new();        
        g_hash_table_insert(LRU_K_params->ghost_hashtable, (gpointer)key, (gpointer)queue);
    }
    
    /* now update the K-element queue */
    guint64* ts = g_new(guint64, 1);
    *ts = LRU_K_params->ts;
    g_queue_push_head(queue, (gpointer)ts);
    if (queue->length > LRU_K_params->maxK)
        g_free(g_queue_pop_tail(queue));
    
    if (g_hash_table_contains(LRU_K_params->cache_hashtable, (gpointer)(cp->item_p)))
        return TRUE;
    else
        return FALSE;
}


inline void __LRU_K_update_element(struct_cache* cache, cache_line* cp){
    /* needs to update pq */
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(cache->cache_params);

    GQueue* queue = g_hash_table_lookup( LRU_K_params->ghost_hashtable,
                                        (gconstpointer)(cp->item_p) );

    pq_node_t* node = (pq_node_t*) g_hash_table_lookup(LRU_K_params->cache_hashtable,
                                                               (gconstpointer)(cp->item_p));
    
    pqueue_pri_t pri;
    if (queue->length < LRU_K_params->K)
        pri = INITIAL_TS;
    else
        pri = *(guint64*) g_queue_peek_nth(queue, LRU_K_params->K-1);
    
    pqueue_change_priority(LRU_K_params->pq, pri, (void*)node);
    return;
}


inline void __LRU_K_evict_element(struct_cache* LRU_K){
    /** pop one node from pq, remove it from cache_hashtable 
     **/
    
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(LRU_K->cache_params);
    
    pq_node_t* node = (pq_node_t*) pqueue_pop(LRU_K_params->pq);
    g_hash_table_remove(LRU_K_params->cache_hashtable, (gconstpointer)(node->item));
}




inline gboolean LRU_K_add_element(struct_cache* cache, cache_line* cp){
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(cache->cache_params);
    LRU_K_params->ts++;
    if (LRU_K_check_element(cache, cp)){
        __LRU_K_update_element(cache, cp);
        return TRUE;
    }
    else{
        __LRU_K_insert_element(cache, cp);
        if ( (long)g_hash_table_size( LRU_K_params->cache_hashtable) > cache->core->size )
            __LRU_K_evict_element(cache);
        return FALSE;
    }
}




void LRU_K_destroy(struct_cache* cache){
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(cache->cache_params);
    
    g_hash_table_destroy(LRU_K_params->cache_hashtable);
    g_hash_table_destroy(LRU_K_params->ghost_hashtable);
    pqueue_free(LRU_K_params->pq);

    cache_destroy(cache);
}

void LRU_K_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(cache->cache_params);
    
    g_hash_table_destroy(LRU_K_params->cache_hashtable);
    g_hash_table_destroy(LRU_K_params->ghost_hashtable);
    pqueue_free(LRU_K_params->pq);
    cache_destroy_unique(cache);
}


struct_cache* LRU_K_init(guint64 size, char data_type, void* params){
    struct_cache *cache = cache_init(size, data_type);
//    cache->cache_params = calloc(1, sizeof(struct LRU_K_params));
    cache->cache_params = g_new0(struct LRU_K_params, 1);
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(cache->cache_params);
    
    cache->core->type = e_LRU_K;
    cache->core->cache_init = LRU_K_init;
    cache->core->destroy = LRU_K_destroy;
    cache->core->destroy_unique = LRU_K_destroy_unique;
    cache->core->add_element = LRU_K_add_element;
    cache->core->check_element = LRU_K_check_element;
    cache->core->cache_init_params = params;

    LRU_K_params->ts = 0;
    LRU_K_params->pq = pqueue_init((size_t)size, cmp_pri, get_pri, set_pri, get_pos, set_pos);
    
    int K, maxK;
    K = ((struct LRU_K_init_params*) params)->K;            // because in gqueue, sequence begins with 0
    maxK = ((struct LRU_K_init_params*) params)->maxK;
    
    LRU_K_params->K = K;
    LRU_K_params->maxK = maxK;
    
    if (data_type == 'l'){
        // don't use pqueue_node_destroyer here, because the item inside node is going to be freed by ghost_hashtable key
        LRU_K_params->cache_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                              NULL,
                                                              simple_g_key_value_destroyer);
        LRU_K_params->ghost_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                              simple_g_key_value_destroyer,
                                                              gqueue_destroyer);
    }
    else if (data_type == 'c'){
        // don't use pqueue_node_destroyer here, because the item inside node is going to be freed by ghost_hashtable key
        LRU_K_params->cache_hashtable = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                              NULL,
                                                              simple_g_key_value_destroyer);
        LRU_K_params->ghost_hashtable = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                              simple_g_key_value_destroyer,
                                                              gqueue_destroyer);
    }
    else{
        g_error("does not support given data type: %c\n", data_type);
    }
    
//    printf("new initialized cache, cache->K = %d, maxK = %d\n", ((struct LRU_K_params*)(cache->cache_params))->K, ((struct LRU_K_params*)(cache->cache_params))->maxK);
    return cache;
}



