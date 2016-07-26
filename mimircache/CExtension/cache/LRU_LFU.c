//
//  LRU_LFU.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h" 
#include "LRU_LFU.h"



inline void __LRU_LFU_insert_element(struct_cache* LRU_LFU, cache_line* cp){
    // insert into LRU segment
    
    struct LRU_LFU_params* LRU_LFU_params = (struct LRU_LFU_params*)(LRU_LFU->cache_params);
    __LRU_insert_element(LRU_LFU_params->LRU, cp);
}


inline gboolean LRU_LFU_check_element(struct_cache* cache, cache_line* cp){
    struct LRU_LFU_params* LRU_LFU_params = (struct LRU_LFU_params*)(cache->cache_params);
    return (LRU_check_element(LRU_LFU_params->LRU, cp) || LFU_check_element(LRU_LFU_params->LFU, cp));
}


inline void __LRU_LFU_update_element(struct_cache* cache, cache_line* cp){
    struct LRU_LFU_params* LRU_LFU_params = (struct LRU_LFU_params*)(cache->cache_params);
    struct LRU_params* LRU_params = (struct LRU_params*)(LRU_LFU_params->LRU->cache_params);
//    struct LFU_params* LFU_params = (struct LFU_params*)(LRU_LFU_params->LFU->cache_params);
    
    if (g_hash_table_contains( LRU_params->hashtable, cp->item_p )){
        // remove from LRU part, insert into LRU segment
        __LRU_remove_element(LRU_LFU_params->LRU, cp->item_p);
        __LFU_insert_element(LRU_LFU_params->LFU, cp);

        // the line below is not necessary, remove it then all elements in LFU begin with freq of 1
//        __LFU_update_element(LRU_LFU_params->LFU, cp);
    }
    else{
        // already in LFU segment, just increase priority
        __LFU_update_element(LRU_LFU_params->LFU, cp);
    }
}


inline void __LRU_LFU_evict_element(struct_cache* LRU_LFU, cache_line* cp){
    ;
}




inline gboolean LRU_LFU_add_element(struct_cache* cache, cache_line* cp){
    static long printOrNot = 1;
    struct LRU_LFU_params* LRU_LFU_params = (struct LRU_LFU_params*)(cache->cache_params);
    struct LRU_params* LRU_params = (struct LRU_params*)(LRU_LFU_params->LRU->cache_params);
    struct LFU_params* LFU_params = (struct LFU_params*)(LRU_LFU_params->LFU->cache_params);
    
    if (LRU_LFU_check_element(cache, cp)){
        __LRU_LFU_update_element(cache, cp);
        return TRUE;
    }
    else{
        __LRU_LFU_insert_element(cache, cp);
        if ( (long)g_hash_table_size(LRU_params->hashtable) > cache->core->size * LRU_LFU_params->LRU_percentage)
            __LRU_evict_element(LRU_LFU_params->LRU, cp);
        if ( (long)g_hash_table_size(LFU_params->hashtable) > cache->core->size * (1-LRU_LFU_params->LRU_percentage)){
            if (cache->core->size > printOrNot){
                printf("LFU full, cache size %ld\n", cache->core->size);
                printOrNot = cache->core->size;
            }
            __LFU_evict_element(LRU_LFU_params->LFU, cp);
        }
        
        return FALSE;
    }
}




void LRU_LFU_destroy(struct_cache* cache){
    struct LRU_LFU_params* LRU_LFU_params = (struct LRU_LFU_params*)(cache->cache_params);
    LRU_destroy(LRU_LFU_params->LRU);
    LFU_destroy(LRU_LFU_params->LFU);

    cache_destroy(cache);
}

void LRU_LFU_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    struct LRU_LFU_params* LRU_LFU_params = (struct LRU_LFU_params*)(cache->cache_params);
    LRU_destroy(LRU_LFU_params->LRU);
    LFU_destroy(LRU_LFU_params->LFU);
    
    g_free(cache->cache_params);
    cache->cache_params = NULL;
        
    g_free(cache->core);
    cache->core = NULL;
    g_free(cache);

}


struct_cache* LRU_LFU_init(guint64 size, char data_type, void* params){
    struct_cache *cache = cache_init(size, data_type);
    cache->cache_params = (void*) g_new0(struct LRU_LFU_params, 1);
    struct LRU_LFU_params* LRU_LFU_params = (struct LRU_LFU_params*)(cache->cache_params);
    LRU_LFU_params->LRU_percentage = ((struct LRU_LFU_init_params*) params)->LRU_percentage;
    LRU_LFU_params->LFU = LFU_init(size*(1-LRU_LFU_params->LRU_percentage), data_type, NULL);
    LRU_LFU_params->LRU = LRU_init(size*LRU_LFU_params->LRU_percentage, data_type, NULL);
    
    
    cache->core->type = e_LRU_LFU;
    cache->core->cache_init = LRU_LFU_init;
    cache->core->destroy = LRU_LFU_destroy;
    cache->core->destroy_unique = LRU_LFU_destroy_unique;
    cache->core->add_element = LRU_LFU_add_element;
    cache->core->check_element = LRU_LFU_check_element;
    cache->core->cache_init_params = params;

    
    return cache;
}



