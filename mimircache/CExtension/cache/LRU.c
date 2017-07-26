//
//  LRU.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "LRU.h"

#ifdef __cplusplus
extern "C"
{
#endif


void __LRU_insert_element(struct_cache* LRU, cache_line* cp){
    struct LRU_params* LRU_params = (struct LRU_params*)(LRU->cache_params);
    
    gpointer key;
    if (cp->type == 'l'){
        key = (gpointer)g_new(guint64, 1);
        *(guint64*)key = *(guint64*)(cp->item_p);
    }
    else{
        key = (gpointer)g_strdup((gchar*)(cp->item_p));
    }
    
    GList* node = g_list_alloc();
    node->data = key;
    
    
    g_queue_push_tail_link(LRU_params->list, node);
    g_hash_table_insert(LRU_params->hashtable, (gpointer)key, (gpointer)node);
    
}

gboolean LRU_check_element(struct_cache* cache, cache_line* cp){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    return g_hash_table_contains( LRU_params->hashtable, cp->item_p );
}


void __LRU_update_element(struct_cache* cache, cache_line* cp){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    GList* node = (GList* ) g_hash_table_lookup(LRU_params->hashtable, cp->item_p);
    g_queue_unlink(LRU_params->list, node);
    g_queue_push_tail_link(LRU_params->list, node);
}


void __LRU_evict_element(struct_cache* LRU, cache_line* cp){
    struct LRU_params* LRU_params = (struct LRU_params*)(LRU->cache_params);

    if (LRU->core->cache_debug_level == 2){     // compare to Oracle
        while (LRU_params->ts > (gint64) g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos)){
            if ( (long) g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos) -
                (long) g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos-1) != 0 ){
                
                LRU->core->evict_err_array[LRU->core->bp_pos-1] = LRU->core->evict_err /
                    (g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos) -
                        g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos-1));
                LRU->core->evict_err = 0;
            }
            else
                LRU->core->evict_err_array[LRU->core->bp_pos-1] = 0;
            
            LRU->core->evict_err = 0;
            LRU->core->bp_pos++;
        }
        
        if (LRU_params->ts == (long) g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos)){
            LRU->core->evict_err_array[LRU->core->bp_pos-1] = (double)LRU->core->evict_err /
            (g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos) -
             g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos-1));
            LRU->core->evict_err = 0;
            LRU->core->bp_pos ++;
        }
            
        gpointer data = g_queue_peek_head(LRU_params->list);
        if (cp->type == 'l'){
            if (*(guint64*)(data) != ((guint64*)LRU->core->oracle)[LRU_params->ts]){
                printf("error at %lu, LRU: %lu, Optimal: %lu\n", LRU_params->ts,
                       *(guint64*)(data), ((guint64*)LRU->core->oracle)[LRU_params->ts]);
                LRU->core->evict_err ++;
            }
            else
                printf("no error at %lu: %lu, %lu\n", LRU_params->ts, *(guint64*)(data),
                       *(guint64*)(g_queue_peek_tail(LRU_params->list)));
            gpointer data_oracle = g_hash_table_lookup(LRU_params->hashtable,
                                        (gpointer)&((guint64* )LRU->core->oracle)[LRU_params->ts]);
            g_queue_delete_link(LRU_params->list, (GList*)data_oracle);
            g_hash_table_remove(LRU_params->hashtable, (gpointer)&((guint64*)LRU->core->oracle)[LRU_params->ts]);
        }
        else{
            if (strcmp((gchar*)data, ((gchar**)(LRU->core->oracle))[LRU_params->ts]) != 0)
                LRU->core->evict_err ++;
            gpointer data_oracle = g_hash_table_lookup(LRU_params->hashtable,
                                                       (gpointer)((gchar**)LRU->core->oracle)[LRU_params->ts]);
            g_hash_table_remove(LRU_params->hashtable,
                                (gpointer)((gchar**)LRU->core->oracle)[LRU_params->ts]);
            g_queue_remove(LRU_params->list, ((GList*) data_oracle)->data);
        }
        
    }
    
    else if (LRU->core->cache_debug_level == 1){
        // record eviction list
        
        gpointer data = g_queue_pop_head(LRU_params->list);
        if (cp->type == 'l'){
            ((guint64*)(LRU->core->eviction_array))[LRU_params->ts] = *(guint64*)(data);
        }
        else{
            gchar* key = g_strdup((gchar*)(data));
            ((gchar**)(LRU->core->eviction_array))[LRU_params->ts] = key;
        }

        g_hash_table_remove(LRU_params->hashtable, (gconstpointer)data);
    }

    
    else{
        gpointer data = g_queue_pop_head(LRU_params->list);
        g_hash_table_remove(LRU_params->hashtable, (gconstpointer)data);
    }
}


gpointer __LRU__evict_with_return(struct_cache* LRU, cache_line* cp){
    /** evict one element and return the evicted element, 
     * needs to free the memory of returned data 
     */
    
    struct LRU_params* LRU_params = (struct LRU_params*)(LRU->cache_params);
    
    gpointer data = g_queue_pop_head(LRU_params->list);
    
    gpointer evicted_key;
    if (cp->type == 'l'){
        evicted_key = (gpointer)g_new(guint64, 1);
        *(guint64*)evicted_key = *(guint64*)(data);
    }
    else{
        evicted_key = (gpointer)g_strdup((gchar*)data);
    }

    g_hash_table_remove(LRU_params->hashtable, (gconstpointer)data);
    return evicted_key;
}




gboolean LRU_add_element(struct_cache* cache, cache_line* cp){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    if (LRU_check_element(cache, cp)){
        __LRU_update_element(cache, cp);
        LRU_params->ts ++;
        return TRUE;
    }
    else{
        __LRU_insert_element(cache, cp);
        if ((long)g_hash_table_size(LRU_params->hashtable) > cache->core->size)
            __LRU_evict_element(cache, cp);
        LRU_params->ts ++;
        return FALSE;
    }
}


gboolean LRU_add_element_only(struct_cache* cache, cache_line* cp){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    if (LRU_check_element(cache, cp)){
        __LRU_update_element(cache, cp);
        return TRUE;
    }
    else{
        __LRU_insert_element(cache, cp);
        while ((long)g_hash_table_size(LRU_params->hashtable) > cache->core->size)
            __LRU_evict_element(cache, cp);
        return FALSE;
    }
}


gboolean LRU_add_element_withsize(struct_cache* cache, cache_line* cp){
    int i;
    gboolean ret_val;
    
    *(gint64*)(cp->item_p) = (gint64) (*(gint64*)(cp->item_p) *
                                       cp->disk_sector_size /
                                       cache->core->block_unit_size);
    ret_val = LRU_add_element(cache, cp);
    
    int n = (int)ceil((double) cp->size/cache->core->block_unit_size);
    
    for (i=0; i<n-1; i++){
        (*(guint64*)(cp->item_p)) ++;
        LRU_add_element_only(cache, cp);
    }
    *(gint64*)(cp->item_p) -= (n-1);
    return ret_val;
}





void LRU_destroy(struct_cache* cache){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);

//    g_queue_free(LRU_params->list);                 // Jason: should call g_queue_free_full to free the memory of node content
    // 0921
    g_queue_free(LRU_params->list);
    g_hash_table_destroy(LRU_params->hashtable);
    cache_destroy(cache);
}

void LRU_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    LRU_destroy(cache);
}


/*-----------------------------------------------------------------------------
 *
 * LRU_init --
 *      initialize a LRU cache
 *
 * Input: 
 *      size:       cache size
 *      data_type:  the type of data, currently support l for long or c for string
 *      block_size: the basic unit size of block, used for profiling with size
 *                  if not profiling with size, this is 0
 *      params:     params used for initialization, NULL for LRU 
 *
 * Return:
 *      a LRU cache struct
 *
 *-----------------------------------------------------------------------------
 */
struct_cache* LRU_init(guint64 size, char data_type, int block_size, void* params){
    struct_cache *cache = cache_init(size, data_type, block_size);
    cache->cache_params = g_new0(struct LRU_params, 1);
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    
    cache->core->type                   =   e_LRU;
    cache->core->cache_init             =   LRU_init;
    cache->core->destroy                =   LRU_destroy;
    cache->core->destroy_unique         =   LRU_destroy_unique;
    cache->core->add_element            =   LRU_add_element;
    cache->core->check_element          =   LRU_check_element;
    cache->core->__insert_element       =   __LRU_insert_element;
    cache->core->__update_element       =   __LRU_update_element;
    cache->core->__evict_element        =   __LRU_evict_element;
    cache->core->__evict_with_return    =   __LRU__evict_with_return;
    cache->core->get_size               =   LRU_get_size;
    cache->core->remove_element         =   LRU_remove_element; 
    cache->core->cache_init_params      =   NULL;
    cache->core->add_element_only       =   LRU_add_element_only; 
    cache->core->add_element_withsize   =   LRU_add_element_withsize;
    

    if (data_type == 'l'){
        LRU_params->hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                      simple_g_key_value_destroyer, NULL);
    }
    else if (data_type == 'c'){
        LRU_params->hashtable = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                      simple_g_key_value_destroyer, NULL);
    }
    else{
        ERROR("does not support given data type: %c\n", data_type);
    }
    LRU_params->list = g_queue_new();
    
    
    return cache;
}




void LRU_remove_element(struct_cache* cache, void* data_to_remove){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    
    gpointer data = g_hash_table_lookup(LRU_params->hashtable, data_to_remove);
    if (!data){
        fprintf(stderr, "LRU_remove_element: data to remove is not in the cache\n");
        exit(1);
    }
    g_queue_delete_link(LRU_params->list, (GList*) data);
    g_hash_table_remove(LRU_params->hashtable, data_to_remove);
}

gint64 LRU_get_size(struct_cache* cache){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    return (guint64) g_hash_table_size(LRU_params->hashtable);
}






#ifdef __cplusplus
}
#endif
