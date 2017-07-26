//
//  FIFO.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "FIFO.h"

#ifdef __cplusplus
extern "C"
{
#endif


 void __fifo_insert_element(struct_cache* fifo, cache_line* cp){
    struct FIFO_params* fifo_params = (struct FIFO_params*)(fifo->cache_params);
    
    gpointer key;
    if (cp->type == 'l'){
        key = (gpointer)g_new(guint64, 1);
        *(guint64*)key = *(guint64*)(cp->item_p);
    }
    else{
        key = (gpointer)g_strdup((gchar*)(cp->item_p));
    }
    g_hash_table_add(fifo_params->hashtable, (gpointer)key);
    // store in a reversed order 
    g_queue_push_tail(fifo_params->list, (gpointer)key);
}

 gboolean fifo_check_element(struct_cache* cache, cache_line* cp){
    struct FIFO_params* fifo_params = (struct FIFO_params*)(cache->cache_params);
    return g_hash_table_contains( fifo_params->hashtable, cp->item_p );
}


 void __fifo_update_element(struct_cache* fifo, cache_line* cp){
    return;
}


void __fifo_evict_element(struct_cache* fifo, cache_line* cp){
    struct FIFO_params* fifo_params = (struct FIFO_params*)(fifo->cache_params);
    gpointer data = g_queue_pop_head(fifo_params->list);
    g_hash_table_remove(fifo_params->hashtable, (gconstpointer)data);
}


gpointer __fifo__evict_with_return(struct_cache* fifo, cache_line* cp){
    struct FIFO_params* fifo_params = (struct FIFO_params*)(fifo->cache_params);
    gpointer data = g_queue_pop_head(fifo_params->list);
    gpointer gp;
    if (cp->type == 'l'){
        gp = g_new(guint64, 1);
        *(guint64*) gp = *(guint64*) data;
    }
    else{
        gp = g_strdup((gchar*)data);
    }
    g_hash_table_remove(fifo_params->hashtable, (gconstpointer)data);
    
    return gp;
}


gboolean fifo_add_element(struct_cache* cache, cache_line* cp){
    struct FIFO_params* fifo_params = (struct FIFO_params*)(cache->cache_params);
    if (fifo_check_element(cache, cp)){
        return TRUE;
    }
    else{
        __fifo_insert_element(cache, cp);
        if ( (long)g_hash_table_size( fifo_params->hashtable) > cache->core->size)
            __fifo_evict_element(cache, cp);
        return FALSE;
    }
}



gboolean fifo_add_element_only(struct_cache* cache, cache_line* cp){
    return fifo_add_element(cache, cp);
}


gboolean fifo_add_element_withsize(struct_cache* cache, cache_line* cp){
    int i, n = 0;
    gint64 original_lbn = *(gint64*)(cp->item_p);
    gboolean ret_val;
    
    if (cache->core->block_unit_size != 0){
        *(gint64*)(cp->item_p) = (gint64) (*(gint64*)(cp->item_p) *
                                           cp->disk_sector_size /
                                           cache->core->block_unit_size);
        n = (int)ceil((double) cp->size/cache->core->block_unit_size);
    }
    ret_val = fifo_add_element(cache, cp);
    
    
    if (cache->core->block_unit_size != 0){
        for (i=0; i<n-1; i++){
            (*(guint64*)(cp->item_p)) ++;
            fifo_add_element_only(cache, cp);
        }
    }

    *(gint64*)(cp->item_p) = original_lbn;
    return ret_val;
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


struct_cache* fifo_init(guint64 size, char data_type, int block_size, void* params){
    struct_cache *cache = cache_init(size, data_type, block_size); 
    cache->cache_params = g_new0(struct FIFO_params, 1);
    struct FIFO_params* fifo_params = (struct FIFO_params*)(cache->cache_params);
    
    cache->core->type                       =       e_FIFO;
    cache->core->cache_init                 =       fifo_init;
    cache->core->destroy                    =       fifo_destroy;
    cache->core->destroy_unique             =       fifo_destroy_unique;
    cache->core->add_element                =       fifo_add_element;
    cache->core->check_element              =       fifo_check_element;
    cache->core->__evict_element            =       __fifo_evict_element;
    cache->core->__insert_element           =       __fifo_insert_element;
    cache->core->__update_element           =       __fifo_update_element;
    cache->core->__evict_with_return        =       __fifo__evict_with_return;
    cache->core->get_size                   =       fifo_get_size; 
    cache->core->add_element_only           =       fifo_add_element; 
    cache->core->add_element_withsize       =       fifo_add_element_withsize; 
    
    cache->core->cache_init_params = NULL;

    if (data_type == 'l'){
        fifo_params->hashtable =
            g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                  simple_g_key_value_destroyer, NULL);
    }
    else if (data_type == 'c'){
        fifo_params->hashtable =
            g_hash_table_new_full(g_str_hash, g_str_equal,
                                  simple_g_key_value_destroyer, NULL);
    }
    else{
        ERROR("does not support given data type: %c\n", data_type);
    }
    fifo_params->list = g_queue_new();
    
    
    return cache;
}


gint64 fifo_get_size(struct_cache *cache){
    struct FIFO_params* fifo_params = (struct FIFO_params*)(cache->cache_params);
    return g_hash_table_size(fifo_params->hashtable);
}



#ifdef __cplusplus
}
#endif