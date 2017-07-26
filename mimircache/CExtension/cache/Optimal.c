//
//  optimal.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "Optimal.h"

#ifdef __cplusplus
extern "C"
{
#endif


/******************* priority queue structs and def **********************/

static int cmp_pri(pqueue_pri_t next, pqueue_pri_t curr){
    return (next.pri1 < curr.pri1);
}


static pqueue_pri_t get_pri(void *a){
    return ((pq_node_t *) a)->pri;
}


static void set_pri(void *a, pqueue_pri_t pri){
    ((pq_node_t *) a)->pri = pri;
}


static size_t get_pos(void *a){
    return ((pq_node_t *) a)->pos;
}


static void set_pos(void *a, size_t pos){
    ((pq_node_t *) a)->pos = pos;
}




/*************************** OPT related ****************************/

void __optimal_insert_element(struct_cache* optimal, cache_line* cp){
    optimal_params_t* optimal_params = (optimal_params_t*)(optimal->cache_params);

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
    if ((gint)g_array_index(optimal_params->next_access, gint, optimal_params->ts) == -1)
        node->pri.pri1 = G_MAXUINT64;
    else
        node->pri.pri1 = optimal_params->ts +
        (gint)g_array_index(optimal_params->next_access, gint, optimal_params->ts);
    pqueue_insert(optimal_params->pq, (void *)node);
    g_hash_table_insert (optimal_params->hashtable, (gpointer)key, (gpointer)node);
}


gboolean optimal_check_element(struct_cache* cache, cache_line* cp){
    return g_hash_table_contains(
                                ((optimal_params_t*)(cache->cache_params))->hashtable,
                                (gconstpointer)(cp->item_p)
                                );
}


void __optimal_update_element(struct_cache* optimal, cache_line* cp){
    optimal_params_t* optimal_params = (optimal_params_t*)(optimal->cache_params);
    void* node;
    node = (void*) g_hash_table_lookup(optimal_params->hashtable, (gconstpointer)(cp->item_p));
    pqueue_pri_t pri;
    
    if ((gint) g_array_index(optimal_params->next_access, gint, optimal_params->ts) == -1)
        pri.pri1 = G_MAXUINT64;
    else
        pri.pri1 = optimal_params->ts +
            (gint)g_array_index(optimal_params->next_access, gint, optimal_params->ts);

    pqueue_change_priority(optimal_params->pq, pri, node);    
}



void __optimal_evict_element(struct_cache* optimal, cache_line* cp){
    optimal_params_t* optimal_params = (optimal_params_t*)(optimal->cache_params);
    
    pq_node_t* node = (pq_node_t*) pqueue_pop(optimal_params->pq);
    if (optimal->core->cache_debug_level == 1){
        // save eviction
        if (cp->type == 'l'){
            ((guint64*)(optimal->core->eviction_array))[optimal_params->ts] = *(guint64*)(node->item);
        }
        else{
            gchar* key = g_strdup((gchar*)(node->item));
            ((gchar**)(optimal->core->eviction_array))[optimal_params->ts] = key;
        }
    }
    
    g_hash_table_remove(optimal_params->hashtable, (gconstpointer)(node->item));
}


void* __optimal_evict_with_return(struct_cache* optimal, cache_line* cp){
    optimal_params_t* optimal_params = (optimal_params_t*)(optimal->cache_params);
    
    void* evicted_key;
    pq_node_t* node = (pq_node_t*) pqueue_pop(optimal_params->pq);
    
    if (cp->type == 'l'){
        evicted_key = (gpointer)g_new(guint64, 1);
        *(guint64*)evicted_key = *(guint64*)(node->item);
    }
    else{
        evicted_key = (gpointer)g_strdup((gchar*)(node->item));
    }

    g_hash_table_remove(optimal_params->hashtable, (gconstpointer)(node->item));
    return evicted_key;
}


gint64 optimal_get_size(struct_cache* cache){
    optimal_params_t* optimal_params = (optimal_params_t*)(cache->cache_params);
    return (guint64) g_hash_table_size(optimal_params->hashtable);
}



gboolean optimal_add_element(struct_cache* cache, cache_line* cp){
    optimal_params_t* optimal_params = (optimal_params_t*)(cache->cache_params);

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
        
        if ( (long)g_hash_table_size( optimal_params->hashtable) > cache->core->size)
            __optimal_evict_element(cache, cp);
        (optimal_params->ts) ++ ;
        return FALSE;
    }
}


gboolean optimal_add_element_only(struct_cache* cache, cache_line* cp){
    optimal_params_t* optimal_params = (optimal_params_t*)(cache->cache_params);
    
    if (optimal_check_element(cache, cp)){
        __optimal_update_element(cache, cp);
        (optimal_params->ts) ++ ;   // do not move
        return TRUE;
    }
    else{
        __optimal_insert_element(cache, cp);
        if ( (long)g_hash_table_size( optimal_params->hashtable) > cache->core->size)
            __optimal_evict_element(cache, cp);
        (optimal_params->ts) ++ ;
        return FALSE;
    }
}


gboolean optimal_add_element_withsize(struct_cache* cache, cache_line* cp){
    ERROR("optimal does not support size now\n");
    abort(); 
    
    int i, n = 0;
    gint64 original_lbn = *(gint64*)(cp->item_p);
    gboolean ret_val;
    
    if (cache->core->block_unit_size != 0){
        *(gint64*)(cp->item_p) = (gint64) (*(gint64*)(cp->item_p) *
                                           cp->disk_sector_size /
                                           cache->core->block_unit_size);
        n = (int)ceil((double) cp->size/cache->core->block_unit_size);
    }
    ret_val = optimal_add_element(cache, cp);
    
    
    if (cache->core->block_unit_size != 0){
        for (i=0; i<n-1; i++){
            (*(guint64*)(cp->item_p)) ++;
            optimal_add_element_only(cache, cp);
        }
    }
    
    *(gint64*)(cp->item_p) = original_lbn;
    return ret_val;
}



 void optimal_destroy(struct_cache* cache){
    optimal_params_t* optimal_params = (optimal_params_t*)(cache->cache_params);

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
    
    optimal_params_t* optimal_params = (optimal_params_t*)(cache->cache_params);
    g_hash_table_destroy(optimal_params->hashtable);
    pqueue_free(optimal_params->pq);
    g_free(cache->cache_params);
    g_free(cache->core);
    g_free(cache);
}



struct_cache* optimal_init(guint64 size, char data_type, int block_size, void* params){
#define pq_size_multiplier 10       // ??? WHY
    struct_cache* cache = cache_init(size, data_type, block_size);
    
    optimal_params_t* optimal_params = g_new0(optimal_params_t, 1);
    cache->cache_params = (void*) optimal_params;
    
    cache->core->type                   =   e_Optimal;
    cache->core->cache_init             =   optimal_init;
    cache->core->destroy                =   optimal_destroy;
    cache->core->destroy_unique         =   optimal_destroy_unique;
    cache->core->add_element            =   optimal_add_element;
    cache->core->check_element          =   optimal_check_element;
    
    cache->core->__insert_element       =   __optimal_insert_element;
    cache->core->__update_element       =   __optimal_update_element;
    cache->core->__evict_element        =   __optimal_evict_element;
    cache->core->__evict_with_return    =   __optimal_evict_with_return;
    cache->core->get_size               =   optimal_get_size;
    cache->core->add_element_only       =   optimal_add_element_only; 
    cache->core->add_element_withsize   =   optimal_add_element_withsize; 

    
    
    optimal_params->ts = ((struct optimal_init_params*)params)->ts;

    reader_t* reader = ((struct optimal_init_params*)params)->reader;
    optimal_params->reader = reader;
    
    
    if (data_type == 'l'){
        optimal_params->hashtable =
            g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                  simple_g_key_value_destroyer,
                                  simple_g_key_value_destroyer);
    }
    
    else if (data_type == 'c'){
        optimal_params->hashtable =
            g_hash_table_new_full(g_str_hash, g_str_equal,
                                  simple_g_key_value_destroyer,
                                  simple_g_key_value_destroyer);
    }
    else{
        ERROR("does not support given data type: %c\n", data_type);
    }
    
    
    optimal_params->pq = pqueue_init(size*pq_size_multiplier, cmp_pri,
                                     get_pri, set_pri, get_pos, set_pos);
    
    if (((struct optimal_init_params*)params)->next_access == NULL){
        if (reader->base->total_num == -1)
            get_num_of_cache_lines(reader);
        optimal_params->next_access = g_array_sized_new (FALSE, FALSE,
                                                         sizeof (gint),
                                                         (guint)reader->base->total_num);
        GArray* array = optimal_params->next_access;
        GSList* list = get_last_access_dist_seq(reader, read_one_element_above);
        if (list == NULL){
            ERROR("error getting last access distance in optimal_init\n");
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

#ifdef __cplusplus
}
#endif