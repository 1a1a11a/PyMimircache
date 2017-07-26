//
//  ARC.h
//  mimircache
//
//  Created by Juncheng on 2/12/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//


#include "ARC.h"

#ifdef __cplusplus
extern "C"
{
#endif


void __ARC_insert_element(struct_cache* cache, cache_line* cp){
    /* first time add, then it should be add to LRU1 */
    ARC_params_t* ARC_params = (ARC_params_t*)(cache->cache_params);
    
#ifdef SANITY_CHECK
    if (ARC_params->LRU1->core->check_element(ARC_params->LRU1, cp)){
        fprintf(stderr, "ERROR: in ARC insert, inserted item is in cache\n");
        exit(1);
    }
#endif
    
    ARC_params->LRU1->core->__insert_element(ARC_params->LRU1, cp);
    ARC_params->size1 ++;
}


gboolean ARC_check_element(struct_cache* cache, cache_line* cp){
    /* check both segments */ 
    ARC_params_t* ARC_params = (ARC_params_t*)(cache->cache_params);
    return (ARC_params->LRU1->core->check_element(ARC_params->LRU1, cp) ||
            ARC_params->LRU2->core->check_element(ARC_params->LRU2, cp) );
}


void __ARC_update_element(struct_cache* cache, cache_line* cp){
    /* if in seg1, then move to seg2; 
     * if in seg2, just update 
     */
    ARC_params_t* ARC_params = (ARC_params_t*)(cache->cache_params);
    if (ARC_params->LRU1->core->check_element(ARC_params->LRU1, cp)){
        ARC_params->LRU1->core->remove_element(ARC_params->LRU1, cp->item_p);
        ARC_params->LRU2->core->__insert_element(ARC_params->LRU2, cp);
        ARC_params->size1 --;
        ARC_params->size2 ++;
    }
    else{
#ifdef SANITY_CHECK
        if (!ARC_params->LRU2->core->check_element(ARC_params->LRU2, cp)){
            fprintf(stderr, "ERROR: in ARC insert, update element in LRU2, but"
                    " item is not in cache\n");
            exit(1);
        }
#endif
        ARC_params->LRU2->core->__update_element(ARC_params->LRU2, cp);
    }
}


void __ARC_evict_element(struct_cache* cache, cache_line* cp){
    ARC_params_t* ARC_params = (ARC_params_t*)(cache->cache_params);
    
    /* check whether it is a hit on LRU1g or LRU2g,
     * if yes, evcit from the other LRU segments 
     * if no,  evict from LRU1. 
     * after evicting, needs to add to corresponding ghost list, 
     * if ghost list is too large, then evict from ghost list
     */
    gpointer evicted_item, old_cp_itemp;
    struct_cache* evicted_from, *remove_ghost, *add_ghost;
    gint64* size;
    if (ARC_params->LRU1g->core->check_element(ARC_params->LRU1g, cp)){
        // hit on LRU1g, evict from LRU2 and remove from LRU1g,
        // then add evicted to LRU2g
        if (ARC_params->size2 > 0){
            evicted_from = ARC_params->LRU2;
            remove_ghost = ARC_params->LRU1g;
            add_ghost    = ARC_params->LRU2g;
            size         = &(ARC_params->size2);
        }
        else{
            evicted_from = ARC_params->LRU1;
            remove_ghost = ARC_params->LRU1g;
            add_ghost    = ARC_params->LRU1g;
            size         = &(ARC_params->size1);
        }
    }
    else if (ARC_params->LRU2g->core->check_element(ARC_params->LRU2g, cp)){
        // hit on LRU2g, evict from LRU1 and remove from LRU2g,
        // then add evicted to LRU1g
        evicted_from = ARC_params->LRU1;
        remove_ghost = ARC_params->LRU2g;
        add_ghost    = ARC_params->LRU1g;
        size         = &(ARC_params->size1);
    }
    else{
        // not hit anywhere, evict from LRU1, then add evicted to LRU1g
        evicted_from = ARC_params->LRU1;
        remove_ghost = NULL;
        add_ghost    = ARC_params->LRU1g;
        size         = &(ARC_params->size1);
    }
    
    // now do the eviction, remove ghost, add ghost entry
    evicted_item = evicted_from->core->__evict_with_return(evicted_from, cp);
    if (remove_ghost)
        remove_ghost->core->remove_element(remove_ghost, cp->item_p);
    (*size) --;
    old_cp_itemp = cp->item_p;
    cp->item_p = evicted_item;
#ifdef SANITY_CHECK
    if (add_ghost->core->check_element(add_ghost, cp)){
        fprintf(stderr, "ERROR: in ARC evict, evicted element in ghost list\n");
        exit(1);
    }
#endif
    add_ghost->core->add_element(add_ghost, cp);
    cp->item_p = old_cp_itemp;

    g_free(evicted_item);
}


gpointer __ARC__evict_with_return(struct_cache* cache, cache_line* cp){
    /* evict one element and return the evicted element, 
     * needs to free the memory of returned data later by user
     */
    
    ARC_params_t* ARC_params = (ARC_params_t*)(cache->cache_params);

    gpointer evicted_item, old_cp_itemp;
    if (ARC_params->LRU1g->core->check_element(ARC_params->LRU1g, cp)){
        // hit on LRU1g, evict from LRU2 and remove from LRU1g,
        // then add evicted to LRU2g
        evicted_item = ARC_params->LRU2->core->__evict_with_return(ARC_params->LRU2, cp);
        ARC_params->LRU1g->core->remove_element(ARC_params->LRU1g, cp->item_p);
        ARC_params->size2 --;
        old_cp_itemp = cp->item_p;
        cp->item_p = evicted_item;
        ARC_params->LRU2g->core->add_element(ARC_params->LRU2g, cp);
        cp->item_p = old_cp_itemp;
    }
    else if (ARC_params->LRU2g->core->check_element(ARC_params->LRU2g, cp)){
        // hit on LRU2g, evict from LRU1 and remove from LRU2g,
        // then add evicted to LRU1g
        evicted_item = ARC_params->LRU1->core->__evict_with_return(ARC_params->LRU1, cp);
        ARC_params->LRU2g->core->remove_element(ARC_params->LRU2g, cp->item_p);
        ARC_params->size1 --;
        old_cp_itemp = cp->item_p;
        cp->item_p = evicted_item;
        ARC_params->LRU1g->core->add_element(ARC_params->LRU1g, cp);
        cp->item_p = old_cp_itemp;
    }
    else{
        // not hit anywhere, evict from LRU1, then add evicted to LRU1g
        evicted_item = ARC_params->LRU1->core->__evict_with_return(ARC_params->LRU1, cp);
        ARC_params->size1 --;
        old_cp_itemp = cp->item_p;
        cp->item_p = evicted_item;
        ARC_params->LRU1g->core->add_element(ARC_params->LRU1g, cp);
        cp->item_p = old_cp_itemp;
    }
    return evicted_item;
}


gboolean ARC_add_element(struct_cache* cache, cache_line* cp){
    ARC_params_t* ARC_params = (ARC_params_t*)(cache->cache_params);
    if (ARC_check_element(cache, cp)){
        __ARC_update_element(cache, cp);
        return TRUE;
    }
    else{
        __ARC_insert_element(cache, cp);
        if ( ARC_params->size1 + ARC_params->size2 > cache->core->size)
            __ARC_evict_element(cache, cp);
#ifdef SANITY_CHECK
        if ((ARC_params->size1 + ARC_params->size2 > cache->core->size) ||
             (ARC_params->LRU1->core->get_size(ARC_params->LRU1) +
              ARC_params->LRU2->core->get_size(ARC_params->LRU2)) > cache->core->size){
            fprintf(stderr, "ERROR: in ARC add_element, after inserting, "
                    "sum of two LRUs sizes: %lu, ARC size1+size2=%lu, "
                    "but cache size=%ld\n", (unsigned long)
                    (ARC_params->LRU1->core->get_size(ARC_params->LRU1) +
                    ARC_params->LRU2->core->get_size(ARC_params->LRU2)),
                    (unsigned long)(ARC_params->size1 + ARC_params->size2),
                    cache->core->size);
            exit(1);
        }
#endif
        return FALSE;
    }
}




void ARC_destroy(struct_cache* cache){
    ARC_params_t* ARC_params = (ARC_params_t*)(cache->cache_params);
    ARC_params->LRU1->core->destroy(ARC_params->LRU1);
    ARC_params->LRU1g->core->destroy(ARC_params->LRU1g);
    ARC_params->LRU2->core->destroy(ARC_params->LRU2);
    ARC_params->LRU2g->core->destroy(ARC_params->LRU2g);
    cache_destroy(cache);
}

void ARC_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    ARC_params_t* ARC_params = (ARC_params_t*)(cache->cache_params);
    ARC_params->LRU1->core->destroy(ARC_params->LRU1);
    ARC_params->LRU1g->core->destroy(ARC_params->LRU1g);
    ARC_params->LRU2->core->destroy(ARC_params->LRU2);
    ARC_params->LRU2g->core->destroy(ARC_params->LRU2g);
    cache_destroy_unique(cache);
}


struct_cache* ARC_init(guint64 size, char data_type, int block_size, void* params){
    struct_cache *cache = cache_init(size, data_type, block_size);
    cache->cache_params = g_new0(struct ARC_params, 1);
    ARC_params_t* ARC_params = (ARC_params_t*)(cache->cache_params);
    ARC_init_params_t* init_params = (ARC_init_params_t*)params;
    
    cache->core->type                   =       e_ARC;
    cache->core->cache_init             =       ARC_init;
    cache->core->destroy                =       ARC_destroy;
    cache->core->destroy_unique         =       ARC_destroy_unique;
    cache->core->add_element            =       ARC_add_element;
    cache->core->check_element          =       ARC_check_element;
    cache->core->__insert_element       =       __ARC_insert_element;
    cache->core->__update_element       =       __ARC_update_element;
    cache->core->__evict_element        =       __ARC_evict_element;
    cache->core->__evict_with_return    =       __ARC__evict_with_return;
    cache->core->get_size               =       ARC_get_size;
    cache->core->cache_init_params      =       params;
    cache->core->add_element_only       =       ARC_add_element; 

    ARC_params->ghost_list_factor = init_params->ghost_list_factor;
    
    ARC_params->LRU1    =   LRU_init(size, data_type, block_size, NULL);
    ARC_params->LRU1g   =   LRU_init(size * ARC_params->ghost_list_factor,
                                     data_type, block_size, NULL);
    ARC_params->LRU2    =   LRU_init(size, data_type, block_size, NULL);
    ARC_params->LRU2g   =   LRU_init(size * ARC_params->ghost_list_factor,
                                     data_type, block_size, NULL);

    return cache;
}




gint64 ARC_get_size(struct_cache* cache){
    ARC_params_t* ARC_params = (ARC_params_t*)(cache->cache_params);
    return (uint64_t)(ARC_params->size1 + ARC_params->size2);
}



#ifdef __cplusplus
}
#endif