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

typedef struct node_t
{
    pqueue_pri_t pri;
    char data_type;
    union{
        guint64 long_item;
        void* item_pointer;
    };
    size_t pos;
} node_pq_LRU_K;


/* changed to make node with small priority on the top */
static inline int
cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
    return (next > curr);
}


static inline pqueue_pri_t
get_pri(void *a)
{
    return ((node_pq_LRU_K *) a)->pri;
}


static inline void
set_pri(void *a, pqueue_pri_t pri)
{
    ((node_pq_LRU_K *) a)->pri = pri;
}


static inline size_t
get_pos(void *a)
{
    return ((node_pq_LRU_K *) a)->pos;
}


static inline void
set_pos(void *a, size_t pos)
{
    ((node_pq_LRU_K *) a)->pos = pos;
}


static void print_GQ(gpointer data, gpointer user_data){
    printf("%lu\t", *(guint64*)data);
}





/* Jason: try to reuse the key from ghost_hashtable for better memory efficiency */

inline void __LRU_K_insert_element_long(struct_cache* LRU_K, cache_line* cp){
    /** update request is done at checking element, 
     * now insert request into cache_hashtable and pq 
     *
     **/
    
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(LRU_K->cache_params);
    
//    gint64* key = g_new(gint64, 1);
//    if (key == NULL){
//        printf("not enough memory\n");
//        exit(1);
//    }
//    *key = cp->long_content;
    guint64* key = NULL;
    
    node_pq_LRU_K* node = (node_pq_LRU_K*) malloc(sizeof(node_pq_LRU_K));
    node->data_type = cp->type;
    node->long_item = cp->long_content;
    
    
    GQueue* queue = NULL; // = g_hash_table_lookup( LRU_K_params->ghost_hashtable, (gconstpointer)(&(cp->long_content)) );
    g_hash_table_lookup_extended(LRU_K_params->ghost_hashtable,
                                 (gconstpointer)(&(cp->long_content)),
                                 (gpointer)&key, (gpointer)&queue);
    
    pqueue_pri_t pri;
    if (queue->length < LRU_K_params->K){
        pri = INITIAL_TS;
    }
    else
        pri = *(guint64*) g_queue_peek_nth(queue, LRU_K_params->K-1);
    
    node->pri = pri;
    pqueue_insert(LRU_K_params->pq, (void*)node);
    g_hash_table_insert(LRU_K_params->cache_hashtable, (gpointer)key, (gpointer)node);
}


inline gboolean LRU_K_check_element_long(struct_cache* cache, cache_line* cp){
    /** check whether request is in the cache_hashtable, 
     * then update ghost_hashtable and pq accordingly, 
     * if in ghost_hashtable, then update it,
     * else create a new entry in ghost_hashtable 
     * the size of ghost_hashtable is maintained separately
     * if in cache_hashtable, update pq by updating priority value
     * if not in cache_hashtable, do nothing
     **/
    
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(cache->cache_params);
    GQueue* queue = g_hash_table_lookup( LRU_K_params->ghost_hashtable, (gconstpointer)(&(cp->long_content)) );
    if (queue == NULL){
        
        /* need to insert the new element into ghost */
        
        guint64* key = g_new(guint64, 1);
        if (key == NULL){
            printf("not enough memory\n");
            exit(1);
        }
        *key = cp->long_content;
        
        queue = g_queue_new();        
        g_hash_table_insert(LRU_K_params->ghost_hashtable, (gpointer)key, (gpointer)queue);
    }
    
    /* now update the K-element queue */
    guint64* ts = g_new(guint64, 1);
    if (ts == NULL){
        printf("not enough memory\n");
        exit(1);
    }
    *ts = LRU_K_params->ts;
    g_queue_push_head(queue, (gpointer)ts);
    if (queue->length > LRU_K_params->maxK)
        free(g_queue_pop_tail(queue));
    
    if (g_hash_table_contains(LRU_K_params->cache_hashtable, (gpointer)&(cp->long_content)))
        return TRUE;
    else
        return FALSE;
}


inline void __LRU_K_update_element_long(struct_cache* cache, cache_line* cp){
    /* needs to update pq */
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(cache->cache_params);

    GQueue* queue = g_hash_table_lookup( LRU_K_params->ghost_hashtable,
                                        (gconstpointer)(&(cp->long_content)) );

    node_pq_LRU_K* node = (node_pq_LRU_K*) g_hash_table_lookup(LRU_K_params->cache_hashtable,
                                                               (gconstpointer)(&(cp->long_content)));
    
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
    
    node_pq_LRU_K* node = (node_pq_LRU_K*) pqueue_pop(LRU_K_params->pq);
    g_hash_table_remove(LRU_K_params->cache_hashtable, (gconstpointer)&(node->long_item));
}




inline gboolean LRU_K_add_element_long(struct_cache* cache, cache_line* cp){
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(cache->cache_params);
    LRU_K_params->ts++;
//    printf("set ts %ld\n", LRU_K_params->ts);
    if (LRU_K_check_element_long(cache, cp)){
        __LRU_K_update_element_long(cache, cp);
        return TRUE;
    }
    else{
        __LRU_K_insert_element_long(cache, cp);
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


struct_cache* LRU_K_init(long long size, char data_type, void* params){
    struct_cache *cache = cache_init(size, data_type);
    cache->cache_params = calloc(1, sizeof(struct LRU_K_params));
    struct LRU_K_params* LRU_K_params = (struct LRU_K_params*)(cache->cache_params);
    
    cache->core->type = e_LRU_K;
    cache->core->cache_init = LRU_K_init;
    cache->core->destroy = LRU_K_destroy;
    cache->core->destroy_unique = LRU_K_destroy_unique;
    cache->core->cache_init_params = params;

    LRU_K_params->ts = 0;
    LRU_K_params->pq = pqueue_init((size_t)size, cmp_pri, get_pri, set_pri, get_pos, set_pos);
    
    int K, maxK;
    K = ((struct LRU_K_init_params*) params)->K;            // because in gqueue, sequence begins with 0
    maxK = ((struct LRU_K_init_params*) params)->maxK;
    
    LRU_K_params->K = K;
    LRU_K_params->maxK = maxK;
    
    if (data_type == 'v'){
        cache->core->add_element = LRU_K_add_element_long;
        cache->core->check_element = LRU_K_check_element_long;
        LRU_K_params->cache_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                              NULL,
                                                              simple_key_value_destroyed);
        LRU_K_params->ghost_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                              simple_key_value_destroyed,
                                                              gqueue_destroyer);
    }
    
    else if (data_type == 'p'){
        printf("not supported yet\n");
    }
    else if (data_type == 'c'){
        printf("not supported yet\n");
    }
    else{
        g_error("does not support given data type: %c\n", data_type);
    }
    
//    printf("new initialized cache, cache->K = %d, maxK = %d\n", ((struct LRU_K_params*)(cache->cache_params))->K, ((struct LRU_K_params*)(cache->cache_params))->maxK);
    return cache;
}



