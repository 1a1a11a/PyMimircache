////
////  ML.c
////  mimircache
////
////  Created by Juncheng on 7/15/16.
////  Copyright © 2016 Juncheng. All rights reserved.
////
//
//#include "ML.h"
//
//
//
// void __ML_insert_element(struct_cache* ML, cache_line* cp){
//    /** update request is done at checking element,
//     * now insert request into cache_hashtable and pq
//     *
//     **/
//
//    struct ML_params* ML_params = (struct ML_params*)(ML->cache_params);
//    gpointer key = NULL;
//
//    pq_node_t* node = g_new(pq_node_t, 1);
//    node->data_type = cp->type;
//
//    GQueue* queue = NULL;
//    g_hash_table_lookup_extended(ML_params->ghost_hashtable,
//                                 (gconstpointer)(cp->item_p),
//                                 &key, (gpointer)&queue);
//
//    pqueue_pri_t pri;
//    if (queue->length < ML_params->K){
//        pri = INITIAL_TS;
//    }
//    else
//        pri = *(guint64*) g_queue_peek_nth(queue, ML_params->K-1);
//
//    node->item = key;
//    node->pri = pri;
//    pqueue_insert(ML_params->pq, (void*)node);
//    g_hash_table_insert(ML_params->cache_hashtable, (gpointer)key, (gpointer)node);
//}
//
//
// gboolean ML_check_element(struct_cache* cache, cache_line* cp){
//    /** check whether request is in the cache_hashtable,
//     * then update ghost_hashtable and pq accordingly,
//     * if in ghost_hashtable, then update it,
//     * else create a new entry in ghost_hashtable
//     * the size of ghost_hashtable is maintained separately
//     * if in cache_hashtable, update pq by updating priority value
//     * if not in cache_hashtable, do nothing
//     **/
//
//    struct ML_params* ML_params = (struct ML_params*)(cache->cache_params);
//    ML_hashtable_value_struct* value = g_hash_table_lookup( ML_params->ghost_hashtable, (gconstpointer)(cp->item_p) );
//    if (value == NULL){
//        /* need to insert the new element into ghost */
//        gpointer key;
//        if (cp->type == 'l'){
//            key = (gpointer)g_new(gint64, 1);
//            *(guint64*)key = *(guint64*)(cp->item_p);
//        }
//        else{
//            key = (gpointer)g_strdup((gchar*)(cp->item_p));
//        }
//        value = g_new0(ML_hashtable_value_struct, 1);
//        value->cold_miss_rate =
//        value->request_rate =
//        value->freq = 1;
//        value->freq_clear = 1;
//        value->rd = -1;
//        value->rd_2 = -1;
//
//        g_hash_table_insert(ML_params->ghost_hashtable, (gpointer)key, (gpointer)value);
//    }
//
//    /* now update the K-element queue */
//    guint64* ts = g_new(guint64, 1);
//    *ts = ML_params->ts;
//    g_queue_push_head(queue, (gpointer)ts);
//    if (queue->length > ML_params->maxK)
//        g_free(g_queue_pop_tail(queue));
//
//    if (g_hash_table_contains(ML_params->cache_hashtable, (gpointer)(cp->item_p)))
//        return TRUE;
//    else
//        return FALSE;
//}
//
//
// void __ML_update_element(struct_cache* cache, cache_line* cp){
//    /* move from segments to segments */
//
//    ;
//}
//
//
// void __ML_evict_element(struct_cache* ML){
//    /** pop one node from pq, remove it from cache_hashtable
//     **/
//
//    struct ML_params* ML_params = (struct ML_params*)(ML->cache_params);
//
//    pq_node_t* node = (pq_node_t*) pqueue_pop(ML_params->pq);
//    g_hash_table_remove(ML_params->cache_hashtable, (gconstpointer)(node->item));
//}
//
//
//
//
// gboolean ML_add_element(struct_cache* cache, cache_line* cp){
//    struct ML_params* ML_params = (struct ML_params*)(cache->cache_params);
//    ML_params->ts++;
//    if (ML_check_element(cache, cp)){
//        __ML_update_element(cache, cp);
//        return TRUE;
//    }
//    else{
//        __ML_insert_element(cache, cp);
//        if ( (long)g_hash_table_size( ML_params->cache_hashtable) > cache->core->size )
//            __ML_evict_element(cache);
//        return FALSE;
//    }
//}
//
//
//
//
//void ML_destroy(struct_cache* cache){
//    struct ML_params* ML_params = (struct ML_params*)(cache->cache_params);
//
//    g_hash_table_destroy(ML_params->cache_hashtable);
//    g_hash_table_destroy(ML_params->ghost_hashtable);
//    pqueue_free(ML_params->pq);
//
//    cache_destroy(cache);
//}
//
//void ML_destroy_unique(struct_cache* cache){
//    /* the difference between destroy_unique and destroy
//     is that the former one only free the resources that are
//     unique to the cache, freeing these resources won't affect
//     other caches copied from original cache
//     in Optimal, next_access should not be freed in destroy_unique,
//     because it is shared between different caches copied from the original one.
//     */
//    struct ML_params* ML_params = (struct ML_params*)(cache->cache_params);
//
//    g_hash_table_destroy(ML_params->cache_hashtable);
//    g_hash_table_destroy(ML_params->ghost_hashtable);
//    pqueue_free(ML_params->pq);
//    cache_destroy_unique(cache);
//}
//
//
//
//extern  ML_hashtable_value_struct* report_feature_add_element(struct_cache* cache, cache_line* cp){
//
//
//    ;
//}
//
//
//struct_cache* ML_init(guint64 size, char data_type, void* params){
//    struct_cache *cache = cache_init(size, data_type);
//    cache->cache_params = g_new0(struct ML_params, 1);
//    struct ML_params* ML_params = (struct ML_params*)(cache->cache_params);
//
//    cache->core->type = e_ML;
//    cache->core->cache_init = ML_init;
//    cache->core->destroy = ML_destroy;
//    cache->core->destroy_unique = ML_destroy_unique;
//    cache->core->add_element = ML_add_element;
//    cache->core->check_element = ML_check_element;
//    cache->core->cache_init_params = params;
//
//    ML_params->ts = 0;
//
//
//    if (data_type == 'l'){
//        // don't use pqueue_node_destroyer here, because the item inside node is going to be freed by ghost_hashtable key
//        ML_params->cache_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal,
//                                                              NULL,
//                                                              simple_g_key_value_destroyer);
//        ML_params->ghost_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal,
//                                                              simple_g_key_value_destroyer,
//                                                              simple_g_key_value_destroyer);
//    }
//    else if (data_type == 'c'){
//        // don't use pqueue_node_destroyer here, because the item inside node is going to be freed by ghost_hashtable key
//        ML_params->cache_hashtable = g_hash_table_new_full(g_str_hash, g_str_equal,
//                                                              NULL,
//                                                              simple_g_key_value_destroyer);
//        ML_params->ghost_hashtable = g_hash_table_new_full(g_str_hash, g_str_equal,
//                                                              simple_g_key_value_destroyer,
//                                                              simple_g_key_value_destroyer);
//    }
//    else{
//        g_error("does not support given data type: %c\n", data_type);
//    }
//
//    return cache;
//
//
//
//    ;
//}