//
//  YJC.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#define LFU_AGING_INTERVAL 100000
#define FIXED_INIT_PREFETCH_INTERVAL 2200000



#include "cache.h" 
#include "LFU.h"
#include "LRU.h"
#include "YJC.h"


void LFU_aging (gpointer key, gpointer value, gpointer user_data){
    ((pq_node_t* )value)->pri = (int)(((pq_node_t* )value)->pri/2);
}



void __YJC_insert_element(struct_cache* YJC, cache_line* cp){
    // insert into LRU segment
    
    struct YJC_params* YJC_params = (struct YJC_params*)(YJC->cache_params);
    __LRU_insert_element(YJC_params->LRU, cp);
    
    gpointer evicted_data = NULL;
    if (LRU_get_size(YJC_params->LRU) > YJC_params->LRU_size)
        evicted_data = __LRU_evict_element_with_return(YJC_params->LRU, cp);

    
    
    
    if (evicted_data){
        // NEEDS TO GET PREDICTED NEXT USAGE TIME
        gint64 next_pos = YJC_params->ts + FIXED_INIT_PREFETCH_INTERVAL;
        
        
        
        // ADD TO PREFETCH DICT
        gpointer gq = g_hash_table_lookup(YJC_params->prefetch_hashtable, &next_pos);
        if (gq == NULL){
            gq = (gpointer) g_queue_new();
            g_queue_push_tail((GQueue*)gq, evicted_data);
            
            gpointer key = (gpointer)g_new(gint64, 1);
            *(gint64*)key = next_pos;
            
            g_hash_table_insert(YJC_params->prefetch_hashtable, key, gq);
        }
        else
            g_queue_push_tail((GQueue*)gq, evicted_data);
        
    }

}




void XCheck(struct_cache* cache, cache_line* cp){
    struct YJC_params* YJC_params = (struct YJC_params*)(cache->cache_params);

    
    int checker = 0;
    if (LRU_check_element(YJC_params->LRU, cp))
        checker ++;
    if (LFU_check_element(YJC_params->LFU, cp))
        checker ++;
    if (LRU_check_element(YJC_params->prefetch, cp))
        checker ++;
    
    if (checker !=0 && checker != 1)
        printf("ERROR: checker %d, LRU:%d, LFU:%d, prefetch:%d\n", checker, LRU_check_element(YJC_params->LRU, cp), \
               LFU_check_element(YJC_params->LFU, cp), LRU_check_element(YJC_params->prefetch, cp));

}

void prefetch(struct_cache* cache, cache_line* cp){
    struct YJC_params* YJC_params = (struct YJC_params*)(cache->cache_params);

    
    // check and prefetch
    gpointer value = g_hash_table_lookup(YJC_params->prefetch_hashtable, &(YJC_params->ts));
    if (value){
        gpointer element = g_queue_pop_head((GQueue*)value);        // element is a pointer to the key, needs free
        cache_line* prefetch_cp = new_cacheline();
        prefetch_cp->type = cp->type;
        
        while (element != NULL){
            // ONLY SUPPORT VSCSI FOR NOW !!!!!!!!!!!!!!!!!!!!!!!!!
            *(guint64*)(prefetch_cp->item_p) = *(guint64*)(element);
            //            if (! (LFU_check_element(YJC_params->LFU, prefetch_cp) || LRU_check_element(YJC_params->LRU, prefetch_cp)) )
            if (LFU_check_element(YJC_params->LFU, prefetch_cp)){
                LFU_remove_element(YJC_params->LFU, prefetch_cp->item_p);
            }
            if (LRU_check_element(YJC_params->LRU, prefetch_cp))
                LRU_remove_element(YJC_params->LRU, prefetch_cp->item_p);
            
            LRU_add_element((struct_cache*)(YJC_params->prefetch), prefetch_cp);
            g_free(element);
            element = g_queue_pop_head((GQueue*)value);
        }
        destroy_cacheline(prefetch_cp);
//        g_queue_free((GQueue*)value);
        g_hash_table_remove(YJC_params->prefetch_hashtable, &(YJC_params->ts));
    }

}




gboolean YJC_check_element(struct_cache* cache, cache_line* cp){
    /** check elements, meanwhile, change ts, aging LFU and prefetch */
    
    struct YJC_params* YJC_params = (struct YJC_params*)(cache->cache_params);
    gboolean result = LRU_check_element(YJC_params->LRU, cp) || LFU_check_element(YJC_params->LFU, cp) \
                        || LRU_check_element(YJC_params->prefetch, cp);
    
    
    XCheck(cache, cp);

    
    int checker = 0;
    if (LRU_check_element(YJC_params->LRU, cp))
        checker ++;
    if (LFU_check_element(YJC_params->LFU, cp))
        checker ++;
    if (LRU_check_element(YJC_params->prefetch, cp))
        checker ++;
    
    if (checker !=0 && checker != 1)
        printf("ERROR: checker %d, LRU:%d, LFU:%d, prefetch:%d\n", checker, LRU_check_element(YJC_params->LRU, cp), \
               LFU_check_element(YJC_params->LFU, cp), LRU_check_element(YJC_params->prefetch, cp));

    
    (YJC_params->ts) ++;
    
    // check and perform aging
    if (YJC_params->ts % LFU_AGING_INTERVAL == 0){
        // perform aging
        struct LFU_params* LFU_params = (struct LFU_params*) (YJC_params->LFU->cache_params);
        g_hash_table_foreach (LFU_params->hashtable, LFU_aging, NULL);
    }
    
    return result;
}


void __YJC_update_element(struct_cache* cache, cache_line* cp){
    struct YJC_params* YJC_params = (struct YJC_params*)(cache->cache_params);
    gpointer evicted_data = NULL;
    
    if (LRU_check_element(YJC_params->LRU, cp)){
        // remove from LRU part, insert into LFU segment
        LRU_remove_element(YJC_params->LRU, cp->item_p);
//        printf("after LRU remove, check element existence: %d\n", LRU_check_element(YJC_params->LRU, cp)); 
        __LFU_insert_element(YJC_params->LFU, cp);
        if (LFU_get_size(YJC_params->LFU) > YJC_params->LFU_size)
            evicted_data = __LFU_evict_element_with_return(YJC_params->LFU, cp);
    }
    else if (LRU_check_element(YJC_params->prefetch, cp)){
        LRU_remove_element(YJC_params->prefetch, cp->item_p);
//        __LFU_insert_element(YJC_params->LFU, cp);
        __LRU_insert_element(YJC_params->LRU, cp);
        
        // CHECK WHETHER NEEDS EVICTION
        if (LRU_get_size(YJC_params->LRU) > YJC_params->LRU_size)
            evicted_data = __LRU_evict_element_with_return(YJC_params->LRU, cp);
    }
    else{
        // already in LFU segment, just increase priority
        __LFU_update_element(YJC_params->LFU, cp);
    }
    
    if (evicted_data){
        // NEEDS TO GET PREDICTED NEXT USAGE TIME
        gint64 next_pos = YJC_params->ts + FIXED_INIT_PREFETCH_INTERVAL;
        
        
        
        // ADD TO PREFETCH DICT
        gpointer gq = g_hash_table_lookup(YJC_params->prefetch_hashtable, &next_pos);
        if (gq == NULL){
            gq = (gpointer) g_queue_new();
            g_queue_push_tail((GQueue*)gq, evicted_data);

            gpointer key = (gpointer)g_new(gint64, 1);
            *(gint64*)key = next_pos;

            g_hash_table_insert(YJC_params->prefetch_hashtable, key, gq);
        }
        else
            g_queue_push_tail((GQueue*)gq, evicted_data);

    }
}


void __YJC_evict_element(struct_cache* YJC, cache_line* cp){
    ;
}




gboolean YJC_add_element(struct_cache* cache, cache_line* cp){    
//    XCheck(cache, cp);
    if (YJC_check_element(cache, cp)){
        __YJC_update_element(cache, cp);
        prefetch(cache, cp);
        return TRUE;
    }
    else{
        __YJC_insert_element(cache, cp);
        prefetch(cache, cp);
        return FALSE;
    }
}




void YJC_destroy(struct_cache* cache){
    struct YJC_params* YJC_params = (struct YJC_params*)(cache->cache_params);
    LRU_destroy(YJC_params->LRU);
    LFU_destroy(YJC_params->LFU);
    LRU_destroy(YJC_params->prefetch);
    g_hash_table_destroy(YJC_params->prefetch_hashtable);

    cache_destroy(cache);
}


void YJC_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    struct YJC_params* YJC_params = (struct YJC_params*)(cache->cache_params);
    LRU_destroy(YJC_params->LRU);
    LFU_destroy(YJC_params->LFU);
    LRU_destroy(YJC_params->prefetch);
    g_hash_table_destroy(YJC_params->prefetch_hashtable);
    
    g_free(cache->cache_params);
    cache->cache_params = NULL;
        
    g_free(cache->core);
    cache->core = NULL;
    g_free(cache);
}


struct_cache* YJC_init(guint64 size, char data_type, void* params){
    struct_cache *cache = cache_init(size, data_type);
    cache->cache_params = (void*) g_new0(struct YJC_params, 1);
    struct YJC_params* YJC_params = (struct YJC_params*)(cache->cache_params);
    YJC_params->LRU_size = (guint64)(((struct YJC_init_params*) params)->LRU_percentage * size);
    YJC_params->LFU_size = (guint64)(((struct YJC_init_params*) params)->LFU_percentage * size);
    YJC_params->prefetch_size = (guint64)(size - YJC_params->LRU_size - YJC_params->LFU_size);

    YJC_params->LFU = LFU_init(YJC_params->LFU_size, data_type, NULL);
    YJC_params->LRU = LRU_init(YJC_params->LRU_size, data_type, NULL);


    YJC_params->prefetch = LRU_init(YJC_params->prefetch_size, data_type, NULL);
    
    YJC_params->ts = 0;
    YJC_params->prefetch_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_g_key_value_destroyer, gqueue_destroyer);

    
    
    cache->core->type = e_YJC;
    cache->core->cache_init = YJC_init;
    cache->core->destroy = YJC_destroy;
    cache->core->destroy_unique = YJC_destroy_unique;
    cache->core->add_element = YJC_add_element;
    cache->core->check_element = YJC_check_element;
    cache->core->cache_init_params = params;

    
    return cache;
}



