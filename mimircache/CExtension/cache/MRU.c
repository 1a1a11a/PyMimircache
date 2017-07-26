//
//  MRU.c
//  mimircache
//
//  MRU cache replacement policy 
//
//  Created by Juncheng on 8/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "MRU.h"

#ifdef __cplusplus
extern "C"
{
#endif


void __MRU_insert_element(struct_cache* MRU, cache_line* cp){
    struct MRU_params* MRU_params = (struct MRU_params*)(MRU->cache_params);
    
    gpointer key;
    if (cp->type == 'l'){
        key = (gpointer)g_new(guint64, 1);
        *(guint64*)key = *(guint64*)(cp->item_p);
    }
    else{
        key = (gpointer)g_strdup((gchar*)(cp->item_p));
    }
    g_hash_table_add (MRU_params->hashtable, (gpointer)key);
}


gboolean MRU_check_element(struct_cache* cache, cache_line* cp){
    return g_hash_table_contains(
                                 ((struct MRU_params*)(cache->cache_params))->hashtable,
                                 (gconstpointer)(cp->item_p)
                                 );
}


void __MRU_update_element(struct_cache* cache, cache_line* cp){
    ;
}



void __MRU_evict_element(struct_cache* cache, cache_line* cp){
    ;
}




gboolean MRU_add_element(struct_cache* cache, cache_line* cp){
    struct MRU_params* MRU_params = (struct MRU_params*)(cache->cache_params);
    
    if (MRU_check_element(cache, cp)){
        return TRUE;
    }
    else{
        if ((long)g_hash_table_size(MRU_params->hashtable) < cache->core->size)
            __MRU_insert_element(cache, cp);
        return FALSE;
    }
}


void MRU_destroy(struct_cache* cache){
    struct MRU_params* MRU_params = (struct MRU_params*)(cache->cache_params);
    
    g_hash_table_destroy(MRU_params->hashtable);
    cache_destroy(cache);
}

 
void MRU_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in MRU, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    MRU_destroy(cache);
}



struct_cache* MRU_init(guint64 size, char data_type, int block_size, void* params){
    struct_cache* cache = cache_init(size, data_type, block_size); 
    struct MRU_params* MRU_params = g_new0(struct MRU_params, 1);
    cache->cache_params = (void*) MRU_params;
    
    cache->core->type = e_MRU;
    cache->core->cache_init = MRU_init;
    cache->core->destroy = MRU_destroy;
    cache->core->destroy_unique = MRU_destroy_unique;
    cache->core->add_element = MRU_add_element;
    cache->core->check_element = MRU_check_element;
    cache->core->add_element_only = MRU_add_element;

    
    if (data_type == 'l'){
        MRU_params->hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_g_key_value_destroyer, NULL);
    }
    
    else if (data_type == 'c'){
        MRU_params->hashtable = g_hash_table_new_full(g_str_hash, g_str_equal, simple_g_key_value_destroyer, NULL);
    }
    else{
        ERROR("does not support given data type: %c\n", data_type);
    }
    
    return cache;
}



#ifdef __cplusplus
}
#endif