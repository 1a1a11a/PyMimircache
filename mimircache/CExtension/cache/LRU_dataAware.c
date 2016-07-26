//
//  LRU_dataAware.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h" 
#include "LRU_dataAware.h"



inline void __LRU_dataAware_insert_element_LFU(struct_cache* LRU_dataAware, cache_line* cp){
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(LRU_dataAware->cache_params);
    __LFU_insert_element(LRU_dataAware_params->LFU, cp);
}

inline void __LRU_dataAware_insert_element_LRU(struct_cache* LRU_dataAware, cache_line* cp){
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(LRU_dataAware->cache_params);
    __LRU_insert_element(LRU_dataAware_params->LRU, cp);
}


inline gboolean LRU_dataAware_check_element(struct_cache* cache, cache_line* cp){
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(cache->cache_params);
    if (LRU_dataAware_params->in_scan)
        return LFU_check_element(LRU_dataAware_params->LFU, cp);
    else
        return LRU_check_element(LRU_dataAware_params->LRU, cp);
}

inline gboolean LRU_dataAware_check_element_LFU(struct_cache* cache, cache_line* cp){
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(cache->cache_params);
    return LFU_check_element(LRU_dataAware_params->LFU, cp);
}

inline gboolean LRU_dataAware_check_element_LRU(struct_cache* cache, cache_line* cp){
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(cache->cache_params);
    return LRU_check_element(LRU_dataAware_params->LRU, cp);
}


inline void __LRU_dataAware_update_element_LFU(struct_cache* cache, cache_line* cp){
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(cache->cache_params);
    __LFU_update_element(LRU_dataAware_params->LFU, cp);
}

inline void __LRU_dataAware_update_element_LRU(struct_cache* cache, cache_line* cp){
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(cache->cache_params);
    __LRU_update_element(LRU_dataAware_params->LRU, cp);
}


inline void __LRU_dataAware_evict_element_LFU(struct_cache* LRU_dataAware, cache_line* cp){
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(LRU_dataAware->cache_params);
    __LFU_evict_element(LRU_dataAware_params->LFU, cp);
}

inline void __LRU_dataAware_evict_element_LRU(struct_cache* LRU_dataAware, cache_line* cp){
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(LRU_dataAware->cache_params);
    __LRU_evict_element(LRU_dataAware_params->LRU, cp);
}



inline gboolean LRU_dataAware_add_element(struct_cache* cache, cache_line* cp){
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(cache->cache_params);
    
    LRU_dataAware_params->total_request_num ++;
    guint64 *ts = g_new0(guint64, 1);
    *ts = cp->real_time;
    if (LRU_dataAware_params->begin_real_time == 0)
        LRU_dataAware_params->begin_real_time = cp->real_time;
    
    g_queue_push_tail(LRU_dataAware_params->time_gqueue, ts);
    
    if ((long)g_hash_table_size(((struct LRU_params*) (LRU_dataAware_params->LRU->cache_params))-> hashtable) >= cache->core->size)
        
    {
    
        guint64 old_ts = *((guint64*)g_queue_peek_head(LRU_dataAware_params->time_gqueue));
        while ( *ts - old_ts > TIME_THRESHOLD){
            g_free(g_queue_pop_head(LRU_dataAware_params->time_gqueue));
            old_ts = *((guint64*)g_queue_peek_head(LRU_dataAware_params->time_gqueue));
        }
        LRU_dataAware_params->avg_request_rate = (guint)((gdouble)(LRU_dataAware_params->total_request_num) /
                                                     (cp->real_time - LRU_dataAware_params->begin_real_time)
                                                     * TIME_THRESHOLD);
    
//        printf("avg request rate %u, current rate: %u\n", LRU_dataAware_params->avg_request_rate, LRU_dataAware_params->time_gqueue->length);
        
        if (LRU_dataAware_params->time_gqueue->length > LRU_dataAware_params->avg_request_rate * AVG_REQUEST_RATE_MAGNIFIER)
            LRU_dataAware_params->scan_checker ++;
        else
            LRU_dataAware_params->scan_checker = 0;
    
//        printf("scan_checker: %u\n", LRU_dataAware_params->scan_checker);
        if (LRU_dataAware_params->scan_checker > SCAN_CHECKER_THRESHOLD){
            if (! LRU_dataAware_params->in_scan){
                LRU_dataAware_params->in_scan = TRUE;
//                printf("in scan %ld, \n", cache->core->size);
            // transform for scan, LRU->LFU, copy everything over
//                GList *list = ((struct LRU_params* )LRU_dataAware_params->LRU->cache_params)->list->head;
//                cache_line* cp2 = g_new0(cache_line, 1);
//                cp2->valid = TRUE;
//                cp2->type = cp->type;
//                cp2->item_p = (gpointer)(cp2->item);
//            
//                while (list){
////                  if (cp2->type == 'l'){
////                    *((guint64*)(cp2->item_p)) = *((guint64*)list->data;
////                }
//                    cp2->item_p = list->data;
//                    LFU_add_element(LRU_dataAware_params->LFU, cp);
//                    list = list->next;
//                }
            }
        }
        else{
            if (LRU_dataAware_params->in_scan){
                LRU_dataAware_params->in_scan = FALSE;
//                printf("back to normal\n");
            // transform back
            
            
            }
        }
    }
    
    
    
    
    if (LRU_dataAware_params->in_scan){
        if (LRU_dataAware_check_element_LFU(cache, cp)){
            __LRU_dataAware_update_element_LFU(cache, cp);
            return TRUE;
        }
        else{
            __LRU_dataAware_insert_element_LFU(cache, cp);
            if ( (long)g_hash_table_size( ((struct LFU_params*)
                                           (LRU_dataAware_params->LFU->cache_params))->hashtable)
                                            > cache->core->size)
                __LRU_dataAware_evict_element_LFU(cache, cp);
            return FALSE;
        }
    }
    else{
        if (LRU_dataAware_check_element_LRU(cache, cp)){
            __LRU_dataAware_update_element_LRU(cache, cp);
            return TRUE;
        }
        else{
            __LRU_dataAware_insert_element_LRU(cache, cp);
            if ( (long)g_hash_table_size( ((struct LRU_params*)
                                           (LRU_dataAware_params->LRU->cache_params))->hashtable)
                > cache->core->size)
                __LRU_dataAware_evict_element_LRU(cache, cp);
            return FALSE;
        }

    }
}




void LRU_dataAware_destroy(struct_cache* cache){
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(cache->cache_params);

    LFU_destroy(LRU_dataAware_params->LFU);
    LRU_destroy(LRU_dataAware_params->LRU);
    g_queue_free_full(LRU_dataAware_params->time_gqueue, simple_g_key_value_destroyer);
    cache_destroy(cache);
}

void LRU_dataAware_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    LRU_dataAware_destroy(cache);
}


struct_cache* LRU_dataAware_init(guint64 size, char data_type, void* params){
    struct_cache *cache = cache_init(size, data_type);
    cache->cache_params = g_new0(struct LRU_dataAware_params, 1);
    struct LRU_dataAware_params* LRU_dataAware_params = (struct LRU_dataAware_params*)(cache->cache_params);
    
    cache->core->type = e_LRU_dataAware;
    cache->core->cache_init = LRU_dataAware_init;
    cache->core->destroy = LRU_dataAware_destroy;
    cache->core->destroy_unique = LRU_dataAware_destroy_unique;
    cache->core->add_element = LRU_dataAware_add_element;
    cache->core->check_element = LRU_dataAware_check_element;
    cache->core->cache_init_params = NULL;


    LRU_dataAware_params->LRU = LRU_init(size, data_type, NULL);
    LRU_dataAware_params->LFU = LFU_init(size, data_type, NULL);

    LRU_dataAware_params->avg_request_rate = 0;
    LRU_dataAware_params->begin_real_time = 0;
    LRU_dataAware_params->scan_checker = 0;
    LRU_dataAware_params->in_scan = FALSE;
    LRU_dataAware_params->time_gqueue = g_queue_new();
    LRU_dataAware_params->total_request_num = 0;
    
    return cache;
}


