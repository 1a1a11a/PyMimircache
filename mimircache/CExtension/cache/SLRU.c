//
//  SLRU.h
//  mimircache
//
//  Created by Juncheng on 2/12/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//


#include "SLRU.h"


#ifdef __cplusplus
extern "C"
{
#endif


void __SLRU_insert_element(struct_cache* SLRU, cache_line* cp){
    SLRU_params_t* SLRU_params = (SLRU_params_t*)(SLRU->cache_params);
    __LRU_insert_element(SLRU_params->LRUs[0], cp);
    SLRU_params->current_sizes[0] ++;
}

gboolean SLRU_check_element(struct_cache* cache, cache_line* cp){
    SLRU_params_t* SLRU_params = (SLRU_params_t*)(cache->cache_params);
    gboolean retVal = FALSE;
    int i;
    for (i=0; i<SLRU_params->N_segments; i++)
        retVal = retVal || LRU_check_element(SLRU_params->LRUs[i], cp);
    return retVal;
}


void __SLRU_update_element(struct_cache* cache, cache_line* cp){
    SLRU_params_t* SLRU_params = (SLRU_params_t*)(cache->cache_params);
    int i;
    for (i=0; i<SLRU_params->N_segments; i++){
        if (LRU_check_element(SLRU_params->LRUs[i], cp)){
            /* move to upper LRU 
             * first remove from current LRU 
             * then add to upper LRU, 
             * if upper LRU is full, evict one, insert into current LRU
             */
            if (i != SLRU_params->N_segments - 1){
                LRU_remove_element(SLRU_params->LRUs[i], cp->item_p);
                SLRU_params->current_sizes[i] --;
                __LRU_insert_element(SLRU_params->LRUs[i+1], cp);
                SLRU_params->current_sizes[i+1] ++;
                if (LRU_get_size(SLRU_params->LRUs[i+1]) >
                    SLRU_params->LRUs[i+1]->core->size){
                    gpointer old_itemp = cp->item_p;
                    gpointer evicted = __LRU__evict_with_return(SLRU_params->LRUs[i+1], cp);
                    SLRU_params->current_sizes[i+1] --;
                    cp->item_p = evicted;
                    __LRU_insert_element(SLRU_params->LRUs[i], cp);
                    SLRU_params->current_sizes[i] ++;
                    cp->item_p = old_itemp;
                    g_free(evicted); 
                }
            }
            else{
                // this is the last segment, just update
                __LRU_update_element(SLRU_params->LRUs[i], cp);
            }
            return;
        }
    }
}


void __SLRU_evict_element(struct_cache* SLRU, cache_line* cp){
    /* because insert only happens at LRU0, 
     * then eviction also can only happens at LRU0 
     */
    SLRU_params_t* SLRU_params = (SLRU_params_t*)(SLRU->cache_params);

    __LRU_evict_element(SLRU_params->LRUs[0], cp);
#ifdef SANITY_CHECK
    if (LRU_get_size(SLRU_params->LRUs[0]) != SLRU_params->LRUs[0]->core->size){
        fprintf(stderr, "ERROR: SLRU_evict_element, after eviction, LRU0 size %lu, "
                "full size %ld\n", (unsigned long)LRU_get_size(SLRU_params->LRUs[0]),
                SLRU_params->LRUs[0]->core->size);
        exit(1);
    }
#endif
}


gpointer __SLRU__evict_with_return(struct_cache* SLRU, cache_line* cp){
    /** evict one element and return the evicted element, 
     user needs to free the memory of returned data **/
    
    SLRU_params_t* SLRU_params = (SLRU_params_t*)(SLRU->cache_params);
    return __LRU__evict_with_return(SLRU_params->LRUs[0], cp);
}



gboolean SLRU_add_element(struct_cache* cache, cache_line* cp){
    SLRU_params_t* SLRU_params = (SLRU_params_t*)(cache->cache_params);
    if (SLRU_check_element(cache, cp)){
        __SLRU_update_element(cache, cp);
        return TRUE;
    }
    else{
        __SLRU_insert_element(cache, cp);
        if ( LRU_get_size(SLRU_params->LRUs[0]) > SLRU_params->LRUs[0]->core->size)
            __SLRU_evict_element(cache, cp);
        return FALSE;
    }
}




void SLRU_destroy(struct_cache* cache){
    SLRU_params_t* SLRU_params = (SLRU_params_t*)(cache->cache_params);
    int i;
    for (i=0; i<SLRU_params->N_segments; i++)
        LRU_destroy(SLRU_params->LRUs[i]);
    g_free(SLRU_params->LRUs);
    g_free(SLRU_params->current_sizes);    
    cache_destroy(cache);
}

void SLRU_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    SLRU_params_t* SLRU_params = (SLRU_params_t*)(cache->cache_params);
    int i;
    for (i=0; i<SLRU_params->N_segments; i++)
        LRU_destroy(SLRU_params->LRUs[i]);
    g_free(SLRU_params->LRUs); 
    g_free(SLRU_params->current_sizes);
    cache_destroy_unique(cache);
}


struct_cache* SLRU_init(guint64 size, char data_type, int block_size, void* params){
    struct_cache *cache = cache_init(size, data_type, block_size);
    cache->cache_params = g_new0(struct SLRU_params, 1);
    SLRU_params_t* SLRU_params = (SLRU_params_t*)(cache->cache_params);
    SLRU_init_params_t* init_params = (SLRU_init_params_t*) params;
    
    cache->core->type                   =   e_SLRU;
    cache->core->cache_init             =   SLRU_init;
    cache->core->destroy                =   SLRU_destroy;
    cache->core->destroy_unique         =   SLRU_destroy_unique;
    cache->core->add_element            =   SLRU_add_element;
    cache->core->check_element          =   SLRU_check_element;
    cache->core->__insert_element       =   __SLRU_insert_element;
    cache->core->__update_element       =   __SLRU_update_element;
    cache->core->__evict_element        =   __SLRU_evict_element;
    cache->core->__evict_with_return    =   __SLRU__evict_with_return;
    cache->core->get_size               =   SLRU_get_size;
    cache->core->cache_init_params      =   params;
    cache->core->add_element_only       =   SLRU_add_element; 

    SLRU_params->N_segments = init_params->N_segments;
    SLRU_params->current_sizes = g_new0(uint64_t, SLRU_params->N_segments);
    SLRU_params->LRUs = g_new(cache_t*, SLRU_params->N_segments);
    int i;
    for (i=0; i<SLRU_params->N_segments; i++){
        SLRU_params->LRUs[i] = LRU_init(size/SLRU_params->N_segments, data_type, block_size, NULL);
    }
    
    return cache;
}


gint64 SLRU_get_size(struct_cache* cache){
    SLRU_params_t* SLRU_params = (SLRU_params_t*)(cache->cache_params);
    int i;
    uint64_t size = 0;
    for (i=0; i<SLRU_params->N_segments; i++)
        size += SLRU_params->current_sizes[i];
    return size;
}


#ifdef __cplusplus
}
#endif