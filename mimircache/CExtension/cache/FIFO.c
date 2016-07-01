//
//  FIFO.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h" 
#include "FIFO.h"

/* need add support for p and c type of data 
 
 */

inline void __fifo_insert_element(struct_cache* fifo, cache_line* cp){
    struct FIFO_params* fifo_params = (struct FIFO_params*)(fifo->cache_params);
    
    gpointer key;
    if (cp->type == 'l'){
        key = (gpointer)g_new(gint64, 1);
        *(guint64*)key = *(guint64*)(cp->item_p);
    }
    else{
        key = (gpointer)g_strdup((gchar*)(cp->item_p));
    }
    g_hash_table_add(fifo_params->hashtable, (gpointer)key);
    // store in a reversed order 
    g_queue_push_tail(fifo_params->list, (gpointer)key);
}

inline gboolean fifo_check_element(struct_cache* cache, cache_line* cp){
    struct FIFO_params* fifo_params = (struct FIFO_params*)(cache->cache_params);
    return g_hash_table_contains( fifo_params->hashtable, cp->item_p );
}


inline void __fifo_update_element(struct_cache* fifo, cache_line* cp){
    return;
}


inline void __fifo_evict_element(struct_cache* fifo){
    struct FIFO_params* fifo_params = (struct FIFO_params*)(fifo->cache_params);
    gpointer data = g_queue_pop_head(fifo_params->list);
    g_hash_table_remove(fifo_params->hashtable, (gconstpointer)data);
}




inline gboolean fifo_add_element(struct_cache* cache, cache_line* cp){
    struct FIFO_params* fifo_params = (struct FIFO_params*)(cache->cache_params);
    if (fifo_check_element(cache, cp)){
        return TRUE;
    }
    else{
        __fifo_insert_element(cache, cp);
        if ( (long)g_hash_table_size( fifo_params->hashtable) > cache->core->size)
            __fifo_evict_element(cache);
        return FALSE;
    }
}




void fifo_destroy(struct_cache* cache){
    struct FIFO_params* fifo_params = (struct FIFO_params*)(cache->cache_params);

    g_queue_free(fifo_params->list);
    g_hash_table_destroy(fifo_params->hashtable);
    cache_destroy(cache);
}

void fifo_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    fifo_destroy(cache);
}


struct_cache* fifo_init(guint64 size, char data_type, void* params){
    struct_cache *cache = cache_init(size, data_type);
    cache->cache_params = g_new0(struct FIFO_params, 1);
    struct FIFO_params* fifo_params = (struct FIFO_params*)(cache->cache_params);
    
    cache->core->type = e_FIFO;
    cache->core->cache_init = fifo_init;
    cache->core->destroy = fifo_destroy;
    cache->core->destroy_unique = fifo_destroy_unique;
    cache->core->add_element = fifo_add_element;
    cache->core->check_element = fifo_check_element;
    cache->core->cache_init_params = NULL;

    if (data_type == 'l'){
        fifo_params->hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_g_key_value_destroyer, NULL);
    }
    else if (data_type == 'c'){
        fifo_params->hashtable = g_hash_table_new_full(g_str_hash, g_str_equal, simple_g_key_value_destroyer, NULL);
    }
    else{
        g_error("does not support given data type: %c\n", data_type);
    }
    fifo_params->list = g_queue_new();
    
    
    return cache;
}



