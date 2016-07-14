//
//  optimal.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h"
#include "pqueue.h" 
#include "heatmap.h"
#include "Optimal.h" 


/* need add support for p and c type of data
 
 */


/** priority queue structs and def 
 */


static inline int
cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
    return (next < curr);
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


//static void print_cacheline(struct_cache* optimal){
//    struct optimal_params* optimal_params = (struct optimal_params*)(optimal->cache_params);
//    if (!pqueue_peek(optimal_params->pq))
//        printf("pq size: %zu, hashtable size: %u, peek: EMPTY\n", pqueue_size(optimal_params->pq), g_hash_table_size(optimal_params->hashtable));
//    else
//        printf("pq size: %zu, hashtable size: %u, peek: %lld\n", pqueue_size(optimal_params->pq), g_hash_table_size(optimal_params->hashtable), ((pq_node_t*)pqueue_peek(optimal_params->pq))->long_item);
//}



inline void __optimal_insert_element(struct_cache* optimal, cache_line* cp){
    struct optimal_params* optimal_params = (struct optimal_params*)(optimal->cache_params);
    
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
    if ((gint)g_array_index(optimal_params->next_access, gint, optimal_params->ts) == -1)
        node->pri = G_MAXUINT64;
    else
        node->pri = optimal_params->ts + (gint)g_array_index(optimal_params->next_access, gint, optimal_params->ts);
    pqueue_insert(optimal_params->pq, (void *)node);
    g_hash_table_insert (optimal_params->hashtable, (gpointer)key, (gpointer)node);
}


inline gboolean optimal_check_element(struct_cache* cache, cache_line* cp){
    return g_hash_table_contains(
                                ((struct optimal_params*)(cache->cache_params))->hashtable,
                                (gconstpointer)(cp->item_p)
                                );
}


inline void __optimal_update_element(struct_cache* optimal, cache_line* cp){
    struct optimal_params* optimal_params = (struct optimal_params*)(optimal->cache_params);
    void* node;
    node = (void*) g_hash_table_lookup(optimal_params->hashtable, (gconstpointer)(cp->item_p));
    
    if ((gint) g_array_index(optimal_params->next_access, gint, optimal_params->ts) == -1)
        pqueue_change_priority(optimal_params->pq, G_MAXUINT64, node);
    else
        pqueue_change_priority(optimal_params->pq,
                               optimal_params->ts +
                               (gint)g_array_index(optimal_params->next_access, gint, optimal_params->ts),
                               node);
}



inline void __optimal_evict_element(struct_cache* optimal, cache_line* cp){
    struct optimal_params* optimal_params = (struct optimal_params*)(optimal->cache_params);

    pq_node_t* node = (pq_node_t*) pqueue_pop(optimal_params->pq);
    
    
//    pq_node_t* node_i = (pq_node_t* ) pqueue_peek(optimal_params->pq);
//    guint64 a=0;
//    guint64 b=0;
//    if (node_i){
//        a = *(guint64*)(node_i->item);
//        b = node_i->pri;
//    }
//    printf("evicting %lu, inside: %lu, pri: %lu\n", *(guint64*)(node->item), a, b);
    
    if (optimal->core->cache_debug_level == 1){
        // save eviction 
        if (cp->type == 'l'){
            ((guint64*)(optimal->core->eviction_array))[cp->ts] = *(guint64*)(node->item);
        }
        else{
            gchar* key = g_strdup((gchar*)(node->item));
            ((gchar**)(optimal->core->eviction_array))[cp->ts] = key;
        }
    }
    
    
    g_hash_table_remove(optimal_params->hashtable, (gconstpointer)(node->item));
}




inline gboolean optimal_add_element(struct_cache* cache, cache_line* cp){
    struct optimal_params* optimal_params = (struct optimal_params*)(cache->cache_params);
    
    if (optimal_check_element(cache, cp)){
        __optimal_update_element(cache, cp);

        
        if (cache->core->cache_debug_level == 1)
            if ((gint)g_array_index(optimal_params->next_access, gint, optimal_params->ts) == -1)
                __optimal_evict_element(cache, cp);

        
        (optimal_params->ts) ++ ;
        return TRUE;
    }
    else{
        __optimal_insert_element(cache, cp);
        
        if (cache->core->cache_debug_level == 1)
            if ((gint)g_array_index(optimal_params->next_access, gint, optimal_params->ts) == -1)
                __optimal_evict_element(cache, cp);
        
        (optimal_params->ts) ++ ;
        if ( (long)g_hash_table_size( optimal_params->hashtable) > cache->core->size)
            __optimal_evict_element(cache, cp);
        return FALSE;
    }
}


inline void optimal_destroy(struct_cache* cache){
    struct optimal_params* optimal_params = (struct optimal_params*)(cache->cache_params);

    g_hash_table_destroy(optimal_params->hashtable);
    pqueue_free(optimal_params->pq);
    g_array_free (optimal_params->next_access, TRUE);
    ((struct optimal_init_params*)(cache->core->cache_init_params))->next_access = NULL;
   
    cache_destroy(cache);
}


inline void optimal_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy 
     is that the former one only free the resources that are 
     unique to the cache, freeing these resources won't affect 
     other caches copied from original cache 
     in Optimal, next_access should not be freed in destroy_unique, 
     because it is shared between different caches copied from the original one.
     */
    
    struct optimal_params* optimal_params = (struct optimal_params*)(cache->cache_params);
    g_hash_table_destroy(optimal_params->hashtable);
    pqueue_free(optimal_params->pq);
    g_free(cache->cache_params);
    g_free(cache->core);
    g_free(cache);
}



struct_cache* optimal_init(guint64 size, char data_type, void* params){
#define pq_size_multiplier 10       // ??? WHY 
    struct_cache* cache = cache_init(size, data_type);
    
//    struct optimal_params* optimal_params = (struct optimal_params*) calloc(1, sizeof(struct optimal_params));
    struct optimal_params* optimal_params = g_new0(struct optimal_params, 1);
    cache->cache_params = (void*) optimal_params;
    
    cache->core->type = e_Optimal;
    cache->core->cache_init = optimal_init;
    cache->core->destroy = optimal_destroy;
    cache->core->destroy_unique = optimal_destroy_unique;
    cache->core->add_element = optimal_add_element;
    cache->core->check_element = optimal_check_element;
    optimal_params->ts = ((struct optimal_init_params*)params)->ts;

    READER* reader = ((struct optimal_init_params*)params)->reader;
    optimal_params->reader = reader;
    
    
    if (data_type == 'l'){
        optimal_params->hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_g_key_value_destroyer, simple_g_key_value_destroyer);
    }
    
    else if (data_type == 'c'){
        optimal_params->hashtable = g_hash_table_new_full(g_str_hash, g_str_equal, simple_g_key_value_destroyer, simple_g_key_value_destroyer);
    }
    else{
        g_error("does not support given data type: %c\n", data_type);
    }
    
    
    optimal_params->pq = pqueue_init(size*pq_size_multiplier, cmp_pri, get_pri, set_pri, get_pos, set_pos);
    
    if (((struct optimal_init_params*)params)->next_access == NULL){
        if (reader->total_num == -1)
            get_num_of_cache_lines(reader);
        optimal_params->next_access = g_array_sized_new (FALSE, FALSE, sizeof (gint), (guint)reader->total_num);
        GArray* array = optimal_params->next_access;
        GSList* list = get_last_access_dist_seq(reader, read_one_element_above);
        if (list == NULL){
            printf("error getting last access distance in optimal_init\n");
            exit(1);
        }
        GSList* list_move = list;
    
    
        gint dist = (GPOINTER_TO_INT(list_move->data));
        g_array_append_val(array, dist);
        while ( (list_move=g_slist_next(list_move)) != NULL){
            dist = (GPOINTER_TO_INT(list_move->data));
            g_array_append_val(array, dist);
        }
        g_slist_free(list);
        
        ((struct optimal_init_params*)params)->next_access = optimal_params->next_access;
    }
    else
        optimal_params->next_access = ((struct optimal_init_params*)params)->next_access;

    cache->core->cache_init_params = params;

    
    
    return cache;
    
}

