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

typedef struct node_t
{
    pqueue_pri_t pri;
    char data_type;
    union{
        long long long_item;
        void* item_mem;
    };
    size_t pos;
} node_t;


static inline int
cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
    return (next < curr);
}


static inline pqueue_pri_t
get_pri(void *a)
{
    return ((node_t *) a)->pri;
}


static inline void
set_pri(void *a, pqueue_pri_t pri)
{
    ((node_t *) a)->pri = pri;
}


static inline size_t
get_pos(void *a)
{
    return ((node_t *) a)->pos;
}


static inline void
set_pos(void *a, size_t pos)
{
    ((node_t *) a)->pos = pos;
}


static void print_cacheline(struct_cache* optimal){
    struct optimal_params* optimal_params = (struct optimal_params*)(optimal->cache_params);
    if (!pqueue_peek(optimal_params->pq))
        printf("pq size: %zu, hashtable size: %u, peek: EMPTY\n", pqueue_size(optimal_params->pq), g_hash_table_size(optimal_params->hashtable));
    else
        printf("pq size: %zu, hashtable size: %u, peek: %lld\n", pqueue_size(optimal_params->pq), g_hash_table_size(optimal_params->hashtable), ((node_t*)pqueue_peek(optimal_params->pq))->long_item);
}



inline void __optimal_insert_element_long(struct_cache* optimal, cache_line* cp){
    struct optimal_params* optimal_params = (struct optimal_params*)(optimal->cache_params);
    
    if ((long long)g_array_index(optimal_params->next_access, gint64, optimal_params->ts) == -1)
        return;
    gint64* key = g_new(gint64, 1);
    if (key == NULL){
        printf("not enough memory\n");
        exit(1);
    }
    *key = cp->long_content;
    node_t *node = (node_t*)malloc(sizeof(node_t));
    node->long_item = cp->long_content;
    node->pri = optimal_params->ts + (long long)g_array_index(optimal_params->next_access, gint64, optimal_params->ts);
    pqueue_insert(optimal_params->pq, (void *)node);
    g_hash_table_insert (optimal_params->hashtable, (gpointer)key, (gpointer)node);
}

inline gboolean optimal_check_element_long(struct_cache* cache, cache_line* cp){
    return g_hash_table_contains(
                                 ((struct optimal_params*)(cache->cache_params))->hashtable,
                                 (gconstpointer)(&(cp->long_content))
                                 );
}


inline void __optimal_update_element_long(struct_cache* optimal, cache_line* cp){
    struct optimal_params* optimal_params = (struct optimal_params*)(optimal->cache_params);

    void* node = (void*) g_hash_table_lookup(optimal_params->hashtable, (gconstpointer)(&(cp->long_content)));
    if ((long long) g_array_index(optimal_params->next_access, gint64, optimal_params->ts) == -1)
        pqueue_change_priority(optimal_params->pq, G_MAXINT64, node);
    else
        pqueue_change_priority(optimal_params->pq,
                               optimal_params->ts +
                               (long long)g_array_index(optimal_params->next_access, guint64, optimal_params->ts),
                               node);
    
    return;
}


inline void __optimal_evict_element(struct_cache* optimal){
    struct optimal_params* optimal_params = (struct optimal_params*)(optimal->cache_params);

    node_t* node = (node_t*) pqueue_pop(optimal_params->pq);
    g_hash_table_remove(optimal_params->hashtable, (gconstpointer)&(node->long_item));
}




inline gboolean optimal_add_element_long(struct_cache* cache, cache_line* cp){
    struct optimal_params* optimal_params = (struct optimal_params*)(cache->cache_params);
    
    if (optimal_check_element_long(cache, cp)){
        __optimal_update_element_long(cache, cp);
        (optimal_params->ts) ++ ;
//        print_cacheline((Optimal*)cache);
        return TRUE;
    }
    else{
        __optimal_insert_element_long(cache, cp);
        (optimal_params->ts) ++ ;
        if ( (long)g_hash_table_size( optimal_params->hashtable) > cache->core->size)
            __optimal_evict_element(cache);
//        print_cacheline((Optimal*)cache);
        return FALSE;
    }
}


void optimal_destroy(struct_cache* cache){
    struct optimal_params* optimal_params = (struct optimal_params*)(cache->cache_params);

    g_hash_table_destroy(optimal_params->hashtable);
    pqueue_free(optimal_params->pq);
    g_array_free (optimal_params->next_access, TRUE);
    ((struct optimal_init_params*)(cache->core->cache_init_params))->next_access = NULL;
   
    cache_destroy(cache);
}


void optimal_destroy_unique(struct_cache* cache){
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
    free(cache->cache_params);
    free(cache->core);
    free(cache);
}



struct_cache* optimal_init(long long size, char data_type, void* params){
#define pq_size_multiplier 10
    struct_cache* cache = cache_init(size, data_type);
    
    struct optimal_params* optimal_params = (struct optimal_params*) calloc(1, sizeof(struct optimal_params));
    cache->cache_params = (void*) optimal_params;
    
    cache->core->type = e_Optimal;
    cache->core->cache_init = optimal_init;
    cache->core->destroy = optimal_destroy;
    cache->core->destroy_unique = optimal_destroy_unique;
    optimal_params->ts = ((struct optimal_init_params*)params)->ts;
    
    READER* reader = ((struct optimal_init_params*)params)->reader;
    optimal_params->reader = reader;
    
    
    if (data_type == 'v'){
        cache->core->add_element = optimal_add_element_long;
        cache->core->check_element = optimal_check_element_long;
        optimal_params->hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_key_value_destroyed, simple_key_value_destroyed);
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
    optimal_params->pq = pqueue_init(size*pq_size_multiplier, cmp_pri, get_pri, set_pri, get_pos, set_pos);
    
    if (((struct optimal_init_params*)params)->next_access == NULL){
        if (reader->total_num == -1)
            get_num_of_cache_lines(reader);
        optimal_params->next_access = g_array_sized_new (FALSE, FALSE, sizeof (gint64), (guint)reader->total_num);
        GArray* array = optimal_params->next_access;
    
    
        GSList* list = get_last_access_dist_seq(reader, read_one_element_above);
        GSList* list_move = list;
    
    
        gint dist = (GPOINTER_TO_INT(list_move->data));
        g_array_append_val(array, dist );
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

