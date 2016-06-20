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


int
cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
    return (next < curr);
}


pqueue_pri_t
get_pri(void *a)
{
    return ((node_t *) a)->pri;
}


void
set_pri(void *a, pqueue_pri_t pri)
{
    ((node_t *) a)->pri = pri;
}


size_t
get_pos(void *a)
{
    return ((node_t *) a)->pos;
}


void
set_pos(void *a, size_t pos)
{
    ((node_t *) a)->pos = pos;
}


static void print_cacheline(Optimal* optimal){
    
    
    if (!pqueue_peek(optimal->pq))
        printf("pq size: %zu, hashtable size: %u, peek: EMPTY\n", pqueue_size(optimal->pq), g_hash_table_size(optimal->hashtable));
    else
        printf("pq size: %zu, hashtable size: %u, peek: %lld\n", pqueue_size(optimal->pq), g_hash_table_size(optimal->hashtable), ((node_t*)pqueue_peek(optimal->pq))->long_item);
}



inline void __optimal_insert_element_long(Optimal* optimal, cache_line* cp){
    if ((long long)g_array_index(optimal->next_access, gint64, optimal->ts) == -1)
        return;
    gint64* key = g_new(gint64, 1);
    if (key == NULL){
        printf("not enough memory\n");
        exit(1);
    }
    *key = cp->long_content;
    node_t *node = (node_t*)malloc(sizeof(node_t));
    node->long_item = cp->long_content;
    node->pri = optimal->ts + (long long)g_array_index(optimal->next_access, gint64, optimal->ts);
    pqueue_insert(optimal->pq, (void *)node);
    g_hash_table_insert (optimal->hashtable, (gpointer)key, (gpointer)node);
}

inline gboolean optimal_check_element_long(struct_cache* cache, cache_line* cp){
    return g_hash_table_contains( ((Optimal*)cache)->hashtable, (gconstpointer)(&(cp->long_content)) );
}


inline void __optimal_update_element_long(Optimal* optimal, cache_line* cp){
    void* node = (void*) g_hash_table_lookup(optimal->hashtable, (gconstpointer)(&(cp->long_content)));
    if ((long long)g_array_index(optimal->next_access, gint64, optimal->ts) == -1)
        pqueue_change_priority(optimal->pq, G_MAXINT64, node);
    else
        pqueue_change_priority(optimal->pq, optimal->ts + (long long)g_array_index(optimal->next_access, guint64, optimal->ts),
                               node);
    
    return;
}


inline void __optimal_evict_element(Optimal* optimal){
    node_t* node = (node_t*) pqueue_pop(optimal->pq);
    g_hash_table_remove(optimal->hashtable, (gconstpointer)&(node->long_item));
//    free(node);
}




inline gboolean optimal_add_element_long(struct_cache* cache, cache_line* cp){
//    if (pqueue_size(((Optimal*)cache)->pq) != (size_t)g_hash_table_size( ((Optimal*)cache)->hashtable)){
//        printf("error \n");
//        exit(1);
//    }
    
    if (optimal_check_element_long(cache, cp)){
        __optimal_update_element_long((Optimal*)cache, cp);
        ((Optimal*)cache)->ts ++ ;
//        print_cacheline((Optimal*)cache);
        return TRUE;
    }
    else{
        __optimal_insert_element_long((Optimal*)cache, cp);
        ((Optimal*)cache)->ts ++ ;
        if ( (long)g_hash_table_size( ((Optimal*)cache)->hashtable ) > cache->size)
            __optimal_evict_element((Optimal*)cache);
//        print_cacheline((Optimal*)cache);
        return FALSE;
    }
}




inline void optimal_destroy(struct_cache* cache){
    Optimal* optimal = (Optimal* )cache;
    g_hash_table_destroy(optimal->hashtable);
    pqueue_free(optimal->pq);
    g_array_free (((Optimal*)cache)->next_access, TRUE);
    free(cache);
}


inline void optimal_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy 
     is that the former one only free the resources that are 
     unique to the cache, freeing these resources won't affect 
     other caches copied from original cache 
     in Optimal, next_access should not be freed in destroy_unique, 
     because it is shared between different caches copied from the original one.
     */
    
    Optimal* optimal = (Optimal* )cache;
    g_hash_table_destroy(optimal->hashtable);
    pqueue_free(optimal->pq);
    free(optimal);
}



struct_cache* optimal_init(long long size, char data_type, void* params){
#define pq_size_multiplier 10
    
    Optimal* optimal = (Optimal*) calloc(1, sizeof(Optimal));
    
    optimal->type = e_Optimal;
    optimal->size = size;
    optimal->data_type = data_type;
    optimal->cache_init = optimal_init;
    optimal->destroy = optimal_destroy;
    optimal->destroy_unique = optimal_destroy_unique;
    optimal->ts = ((struct optimal_init_params*)params)->ts;
    
    READER* reader = ((struct optimal_init_params*)params)->reader;
    optimal->reader = reader;
    
    
    if (data_type == 'v'){
        optimal->add_element = optimal_add_element_long;
        optimal->check_element = optimal_check_element_long;
        optimal->hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_key_value_destroyed, simple_key_value_destroyed);
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
    optimal->pq = pqueue_init(size*pq_size_multiplier, cmp_pri, get_pri, set_pri, get_pos, set_pos);
    
    if (((struct optimal_init_params*)params)->next_access == NULL){
        if (reader->total_num == -1)
            get_num_of_cache_lines(reader);
        optimal->next_access = g_array_sized_new (FALSE, FALSE, sizeof (gint64), (guint)reader->total_num);
        GArray* array = optimal->next_access;
    
    
        GSList* list = get_last_access_dist_seq(reader, read_one_element_above);
        GSList* list_move = list;
    
    
        gint64 dist = (gint64)(GPOINTER_TO_INT(list_move->data));
        g_array_append_val(array, dist );
        while ( (list_move=g_slist_next(list_move)) != NULL){
            dist = (gint64)(GPOINTER_TO_INT(list_move->data));
            g_array_append_val(array, dist);
        }
        g_slist_free(list);
        
        ((struct optimal_init_params*)params)->next_access = optimal->next_access;
        optimal->cache_init_params = params;
    }
    else
        optimal->next_access = ((struct optimal_init_params*)params)->next_access;

    
    
    return (struct_cache*)optimal;
    
}

