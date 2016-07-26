//
//  LRU.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h" 
#include "LRU.h"



inline void __LRU_insert_element(struct_cache* LRU, cache_line* cp){
    struct LRU_params* LRU_params = (struct LRU_params*)(LRU->cache_params);
    
    gpointer key;
    if (cp->type == 'l'){
        key = (gpointer)g_new(gint64, 1);
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

inline gboolean LRU_check_element(struct_cache* cache, cache_line* cp){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    return g_hash_table_contains( LRU_params->hashtable, cp->item_p );
}


inline void __LRU_update_element(struct_cache* cache, cache_line* cp){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    GList* node = (GList* ) g_hash_table_lookup(LRU_params->hashtable, cp->item_p);
    g_queue_unlink(LRU_params->list, node);
    g_queue_push_tail_link(LRU_params->list, node);
}


inline void __LRU_evict_element(struct_cache* LRU, cache_line* cp){
    struct LRU_params* LRU_params = (struct LRU_params*)(LRU->cache_params);

    if (LRU->core->cache_debug_level == 2){     // compare to Oracle
        while (cp->ts > g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos)){
            if ( g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos) -
                g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos-1) != 0 ){
                
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
        
        if (cp->ts == g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos)){
            LRU->core->evict_err_array[LRU->core->bp_pos-1] = (double)LRU->core->evict_err /
            (g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos) -
             g_array_index(LRU->core->bp->array, guint64, LRU->core->bp_pos-1));
            LRU->core->evict_err = 0;
            LRU->core->bp_pos ++;
        }
            
        gpointer data = g_queue_peek_head(LRU_params->list);
        if (cp->type == 'l'){
            if (*(guint64*)(data) != ((guint64*)LRU->core->oracle)[cp->ts]){
                printf("error at %lu, LRU: %lu, Optimal: %lu\n", cp->ts, *(guint64*)(data), ((guint64*)LRU->core->oracle)[cp->ts]);
                LRU->core->evict_err ++;
            }
            else
                printf("no error at %lu: %lu, %lu\n", cp->ts, *(guint64*)(data), *(guint64*)(g_queue_peek_tail(LRU_params->list)));
            gpointer data_oracle = g_hash_table_lookup(LRU_params->hashtable, (gpointer)&((guint64* )LRU->core->oracle)[cp->ts]);
            g_queue_delete_link(LRU_params->list, (GList*)data_oracle);
//            g_queue_remove(LRU_params->list, ((GList*) data_oracle)->data);
            g_hash_table_remove(LRU_params->hashtable, (gpointer)&((guint64*)LRU->core->oracle)[cp->ts]);
        }
        else{
            if (strcmp((gchar*)data, ((gchar**)(LRU->core->oracle))[cp->ts]) != 0)
                LRU->core->evict_err ++;
            gpointer data_oracle = g_hash_table_lookup(LRU_params->hashtable, (gpointer)((gchar**)LRU->core->oracle)[cp->ts]);
            g_hash_table_remove(LRU_params->hashtable, (gpointer)((gchar**)LRU->core->oracle)[cp->ts]);
            g_queue_remove(LRU_params->list, ((GList*) data_oracle)->data);
        }
        
    }
    
    else if (LRU->core->cache_debug_level == 1){
        // record eviction list
        
        gpointer data = g_queue_pop_head(LRU_params->list);
        if (cp->type == 'l'){
            ((guint64*)(LRU->core->eviction_array))[cp->ts] = *(guint64*)(data);
        }
        else{
            gchar* key = g_strdup((gchar*)(data));
            ((gchar**)(LRU->core->eviction_array))[cp->ts] = key;
        }

        g_hash_table_remove(LRU_params->hashtable, (gconstpointer)data);
    }

    
    else{
        gpointer data = g_queue_pop_head(LRU_params->list);
        g_hash_table_remove(LRU_params->hashtable, (gconstpointer)data);
    }
}




inline gboolean LRU_add_element(struct_cache* cache, cache_line* cp){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    if (LRU_check_element(cache, cp)){
        __LRU_update_element(cache, cp);
        return TRUE;
    }
    else{
        __LRU_insert_element(cache, cp);
        if ( (long)g_hash_table_size( LRU_params->hashtable) > cache->core->size)
            __LRU_evict_element(cache, cp);
        return FALSE;
    }
}




void LRU_destroy(struct_cache* cache){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);

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


struct_cache* LRU_init(guint64 size, char data_type, void* params){
    struct_cache *cache = cache_init(size, data_type);
    cache->cache_params = g_new0(struct LRU_params, 1);
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    
    cache->core->type = e_LRU;
    cache->core->cache_init = LRU_init;
    cache->core->destroy = LRU_destroy;
    cache->core->destroy_unique = LRU_destroy_unique;
    cache->core->add_element = LRU_add_element;
    cache->core->check_element = LRU_check_element;
    cache->core->cache_init_params = NULL;

    if (data_type == 'l'){
        LRU_params->hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_g_key_value_destroyer, NULL);
    }
    else if (data_type == 'c'){
        LRU_params->hashtable = g_hash_table_new_full(g_str_hash, g_str_equal, simple_g_key_value_destroyer, NULL);
    }
    else{
        g_error("does not support given data type: %c\n", data_type);
    }
    LRU_params->list = g_queue_new();
    
    
    return cache;
}




inline void __LRU_remove_element(struct_cache* cache, void* data_to_remove){
    struct LRU_params* LRU_params = (struct LRU_params*)(cache->cache_params);
    
//    if (cp->type == 'l'){
        gpointer data = g_hash_table_lookup(LRU_params->hashtable, data_to_remove);
        g_queue_delete_link(LRU_params->list, (GList*) data);
        g_hash_table_remove(LRU_params->hashtable, data_to_remove);
//    }
//    else{
//        if (strcmp((gchar*)data, ((gchar**)(LRU->core->oracle))[cp->ts]) != 0)
//            LRU->core->evict_err ++;
//        gpointer data_oracle = g_hash_table_lookup(LRU_params->hashtable, (gpointer)((gchar**)LRU->core->oracle)[cp->ts]);
//        g_hash_table_remove(LRU_params->hashtable, (gpointer)((gchar**)LRU->core->oracle)[cp->ts]);
//        g_queue_remove(LRU_params->list, ((GList*) data_oracle)->data);
//    }

}

