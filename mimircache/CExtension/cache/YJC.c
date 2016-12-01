//
//  YJC.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#define LFU_AGING_INTERVAL 100000
#define FIXED_INIT_PREFETCH_INTERVAL 2200000

#define THRESHHOLD 2000000
#define global_param1 2000
#define global_param2 10


#include "cache.h" 
#include "LFU.h"
#include "LRU.h"
#include "YJC.h"


/***************************** helping functions ********************************/
 
static void
printHashTable (gpointer key, gpointer value, gpointer user_data){
    printf("key %lu, value %p\n", *(guint64*)key, value);
}
 



void clustering_group_destroyer(gpointer data){
    struct clustering_group* cgroup = (struct clustering_group*)data;
    g_array_free(cgroup->predictions, TRUE);
    g_slist_free(cgroup->list);
    g_free(cgroup);
}


/************************** end helping functions *******************************/






void LFU_aging (gpointer key, gpointer value, gpointer user_data){
    ((pq_node_t* )value)->pri = (int)(((pq_node_t* )value)->pri/2);
}



void __YJC_insert_element(struct_cache* YJC, cache_line* cp){
    // insert into LRU segment
    
    struct YJC_params* YJC_params = (struct YJC_params*)(YJC->cache_params);
    __LRU_insert_element(YJC_params->LRU, cp);
    
    gpointer evicted_data = NULL;
    if (LRU_get_size(YJC_params->LRU) > YJC_params->LRU_size)
        evicted_data = __LRU__evict_with_return(YJC_params->LRU, cp);

    if (evicted_data){
        if (g_hash_table_contains(YJC_params->clustering_hashtable, evicted_data)){
            //            printf("element hashtable size %u, ts %ld, evicting: %lu\n", g_hash_table_size(YJC_params->element_hashtable), YJC_params->ts, *(guint64*)evicted_data);
            //            printf("LFU size %u\n", g_hash_table_size(((struct LFU_params*)(YJC_params->LFU->cache_params))->hashtable));
            //            g_hash_table_foreach(YJC_params->element_hashtable, printHashTable, NULL);
            struct element_info* info = (struct element_info*)g_hash_table_lookup(YJC_params->element_hashtable, evicted_data);
            info->evicted = TRUE;
            info->already_prefetched = FALSE;
        }
        
        g_free(evicted_data);
    }
    
    
//    if (evicted_data){
//        // NEEDS TO GET PREDICTED NEXT USAGE TIME
//        gint64 next_pos = YJC_params->ts + FIXED_INIT_PREFETCH_INTERVAL;
//        
//        
//        
//        // ADD TO PREFETCH DICT
//        gpointer gq = g_hash_table_lookup(YJC_params->prefetch_hashtable, &next_pos);
//        if (gq == NULL){
//            gq = (gpointer) g_queue_new();
//            g_queue_push_tail((GQueue*)gq, evicted_data);
//            
//            gpointer key = (gpointer)g_new(gint64, 1);
//            *(gint64*)key = next_pos;
//            
//            g_hash_table_insert(YJC_params->prefetch_hashtable, key, gq);
//        }
//        else
//            g_queue_push_tail((GQueue*)gq, evicted_data);
//        
//    }

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
//        YJC_params->counter += 1;
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
    /** new: add element to element dict or alter its info struct **/
    
    
    struct YJC_params* YJC_params = (struct YJC_params*)(cache->cache_params);
    gboolean result = LRU_check_element(YJC_params->LRU, cp) || LFU_check_element(YJC_params->LFU, cp) \
                        || LRU_check_element(YJC_params->prefetch, cp);
    
    if (LRU_check_element(YJC_params->prefetch, cp))
        YJC_params->counter ++;
    
//    XCheck(cache, cp);
    if (g_hash_table_contains(YJC_params->clustering_hashtable, cp->item_p)){
        // if element is not in clustering hashtable, we don't need to store its info
        if (g_hash_table_contains(YJC_params->element_hashtable, cp->item_p)){
            /** update element info struct,
             *  
             **/
            struct element_info* info = (struct element_info*)g_hash_table_lookup(YJC_params->element_hashtable, cp->item_p);
            guint64 interval = YJC_params->ts - info->last_access_time;
            
            if ( (!info->already_prefetched) && info->evicted && interval > THRESHHOLD ){
                /** update all prefetch time in the same group,
                 *  this should happen only when interval is not too small, 
                 *  otherwise, we don't need prefetch, 
                 *  when the element is not evicted yet, we also don't prefetch, 
                 *  if the element is already prefetched by other elements in the group, 
                 *  no prefetch either
                 **/

                struct clustering_group* cgroup = (struct clustering_group*) g_hash_table_lookup(YJC_params->clustering_hashtable, cp->item_p);
                
                
                
                if (!(cgroup->num_of_last_predictions >= PREDICTION_LIMIT ||
                      cgroup->num_of_last_predictions >= ((int) (g_slist_length(cgroup->list)))/100+1 )){
                    // we need to have either 1% element in the group agree on the prediction or enough (PREDICTION_LIMIT) prediction agree with each other
                    cgroup->last_predictions[(cgroup->num_of_last_predictions)++] = interval;
//                    printf("I am here3, num of last predictions %d\n", cgroup->num_of_last_predictions);
                    int i;
                    gint64* last_predictions = cgroup->last_predictions;
                    gint64 temp[PREDICTION_LIMIT+2];
                    gint pos = 0;
                    
                    // remove outliers
                    if (cgroup->num_of_last_predictions > 1){
//                        printf("1: %ld, 2: %ld\n", last_predictions[0], last_predictions[1]);
                        for (i=1; i<cgroup->num_of_last_predictions; i++){
                            if (last_predictions[i] - last_predictions[i-1] <= global_param1*global_param2)
                                temp[pos++] = last_predictions[i-1];
                        }
                        if (last_predictions[cgroup->num_of_last_predictions-1] - last_predictions[cgroup->num_of_last_predictions-2] <= global_param1*global_param2)
                            temp[pos++] = last_predictions[cgroup->num_of_last_predictions-1];
                        for (i=0; i<pos; i++){
                            last_predictions[i] = temp[i];
                        }
//                        printf("before remove %d, after remove %d\n", cgroup->num_of_last_predictions, pos);
                        cgroup->num_of_last_predictions = pos;
                    }
                }
                
                // verify, maybe use variance ???????????????????
                if ( (cgroup->num_of_last_predictions >= PREDICTION_LIMIT ||
                      cgroup->num_of_last_predictions >= ((int) (g_slist_length(cgroup->list)))/100+1 ) ){
//                    printf("last predictions size %d\n", cgroup->num_of_last_predictions);
                    
                    // set new interval to be minimal, (or maybe second to the minimal)
                    guint64 min=cgroup->last_predictions[0];
                    int i;
                    for (i=1; i<cgroup->num_of_last_predictions; i++)
                        if (cgroup->last_predictions[i] < min)
                            min = cgroup->last_predictions[i];
                    interval = min;
                
//                if (1){
//                    interval -= global_param1;
                
                
                
                
                
                

                
                
                    
                    GSList* list = cgroup->list;
                    GSList* list_node = list;
                    guint64 predicted_time;
    //                printf("checking node %lu, last access %lu, accessed_time %d, evicted %d, prefetched %d\n", *((guint64*)(cp->item_p)), info->last_access_time, info->accessed_times, info->evicted, info->already_prefetched);
                    struct element_info* info2;
                    while (list_node){
    //                    printf("in while\n");
    //                    printf("looking for %lu\n", *((guint64*)(list_node->data)));
                        info2 = (struct element_info*)g_hash_table_lookup(YJC_params->element_hashtable, list_node->data);
                        if (!info2){
                            /* fix this hack!!!!!!!!!!!!!!!! This is to avoid the situation, one appears the second time, while others in the group haven't shown up yet */
                            list_node = list_node->next;
                            continue;
                        }
                        
                        predicted_time = info2->last_access_time + interval;  // make sure, it is prefetched
                        if (*(guint64*)(list_node->data) == 39691319){
                            printf("39691319 interval %lu, predicted time %lu, current ts %ld, calling for prefetch %ld \n", interval, predicted_time, YJC_params->ts, *(guint64*)(cp->item_p));
                        }
                        info2->already_prefetched = TRUE;
                        if (predicted_time <= YJC_params->ts){
                            predicted_time = YJC_params->ts+1;
                            
                            
//                            list_node = list_node->next;
//                            continue;
                        }
                        // ADD TO PREFETCH DICT
                        gpointer gq = g_hash_table_lookup(YJC_params->prefetch_hashtable, &predicted_time);
                        gint64* evicted_data = g_new(gint64, 1);
                        memcpy(evicted_data, list_node->data, sizeof(gint64));
                        if (gq == NULL){
                            gq = (gpointer) g_queue_new();
                            g_queue_push_tail((GQueue*)gq, evicted_data);
                            
                            gpointer key = (gpointer)g_new(gint64, 1);
                            *(gint64*)key = predicted_time;
                            
                            g_hash_table_insert(YJC_params->prefetch_hashtable, key, gq);
                        }
                        else
                            g_queue_push_tail((GQueue*)gq, evicted_data);
                        list_node = list_node->next;

                    }
                
                    cgroup->num_of_last_predictions = 0;
                
                }
                
            }
            info->accessed_times ++;
            info->evicted = FALSE;
            info->last_access_time = YJC_params->ts;
            info->already_prefetched = TRUE;
        }
        else{
            /** insert into element_hashtable 
             **/
            struct element_info *info = g_new(struct element_info, 1);
            info->accessed_times = 1;
            info->effective_accessed_times = 1;
            info->last_access_time = YJC_params->ts;
            info->evicted = FALSE;
            info->already_prefetched = FALSE;
            gint64* key = g_new(gint64, 1);
            *key = (gint64) *(guint64*)(cp->item_p);
            g_hash_table_insert(YJC_params->element_hashtable, (gpointer)key, (gpointer) info);
        }
    
    }

    
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
            evicted_data = __LFU__evict_with_return(YJC_params->LFU, cp);
    }
    else if (LRU_check_element(YJC_params->prefetch, cp)){
        LRU_remove_element(YJC_params->prefetch, cp->item_p);
//        __LFU_insert_element(YJC_params->LFU, cp);
        __LRU_insert_element(YJC_params->LRU, cp);
        
        // CHECK WHETHER NEEDS EVICTION
        if (LRU_get_size(YJC_params->LRU) > YJC_params->LRU_size)
            evicted_data = __LRU__evict_with_return(YJC_params->LRU, cp);
    }
    else{
        // already in LFU segment, just increase priority
        __LFU_update_element(YJC_params->LFU, cp);
    }
    
    if (evicted_data){
        if (g_hash_table_contains(YJC_params->clustering_hashtable, evicted_data)){
//            printf("element hashtable size %u, ts %ld, evicting: %lu\n", g_hash_table_size(YJC_params->element_hashtable), YJC_params->ts, *(guint64*)evicted_data);
//            printf("LFU size %u\n", g_hash_table_size(((struct LFU_params*)(YJC_params->LFU->cache_params))->hashtable));
//            g_hash_table_foreach(YJC_params->element_hashtable, printHashTable, NULL);
            struct element_info* info = (struct element_info*)g_hash_table_lookup(YJC_params->element_hashtable, evicted_data);
            info->evicted = TRUE;
            info->already_prefetched = FALSE;
        }
        
        g_free(evicted_data);
        
        
//        // NEEDS TO GET PREDICTED NEXT USAGE TIME
//        gint64 next_pos = YJC_params->ts + FIXED_INIT_PREFETCH_INTERVAL;
//        
//        
//        
//        // ADD TO PREFETCH DICT
//        gpointer gq = g_hash_table_lookup(YJC_params->prefetch_hashtable, &next_pos);
//        if (gq == NULL){
//            gq = (gpointer) g_queue_new();
//            g_queue_push_tail((GQueue*)gq, evicted_data);
//
//            gpointer key = (gpointer)g_new(gint64, 1);
//            *(gint64*)key = next_pos;
//
//            g_hash_table_insert(YJC_params->prefetch_hashtable, key, gq);
//        }
//        else
//            g_queue_push_tail((GQueue*)gq, evicted_data);

    }
}


void __YJC_evict_element(struct_cache* YJC, cache_line* cp){
    ;
}




gboolean YJC_add_element(struct_cache* cache, cache_line* cp){    
//    XCheck(cache, cp);
    struct YJC_params* YJC_params = (struct YJC_params*)(cache->cache_params);
    if (YJC_params->ts % 1000000 == 0)
    printf("ts: %ld, prefetch table size: %u, element table size: %u\n", YJC_params->ts, g_hash_table_size(YJC_params->prefetch_hashtable), g_hash_table_size(YJC_params->element_hashtable));
    if (YJC_check_element(cache, cp)){
        __YJC_update_element(cache, cp);
        prefetch(cache, cp);
        return TRUE;
    }
    else{
//        if (YJC_params->ts > 3000000 )
//            printf("missed %lu, ts %ld\n", *(guint64*)(cp->item_p), YJC_params->ts);
        if (*(guint64*)(cp->item_p) == 39691319)
            printf("39691319 missed, ts %ld\n\n\n", YJC_params->ts);
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
    g_hash_table_destroy(YJC_params->clustering_hashtable);

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
    printf("counter: %lu\n\n", YJC_params->counter);
//    g_hash_table_foreach(YJC_params->prefetch_hashtable, printHashTable, NULL);
    
    LRU_destroy(YJC_params->LRU);
    LFU_destroy(YJC_params->LFU);
    LRU_destroy(YJC_params->prefetch);
    g_hash_table_destroy(YJC_params->prefetch_hashtable);
    g_hash_table_destroy(YJC_params->element_hashtable);
    
    
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
    YJC_params->counter = 0;
    
    
    
    cache->core->type = e_YJC;
    cache->core->cache_init = YJC_init;
    cache->core->destroy = YJC_destroy;
    cache->core->destroy_unique = YJC_destroy_unique;
    cache->core->add_element = YJC_add_element;
    cache->core->check_element = YJC_check_element;
    cache->core->cache_init_params = params;

    
    
    
    YJC_params->element_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_g_key_value_destroyer, simple_g_key_value_destroyer);
    
    
    
    if (((struct YJC_init_params*) params)->clustering_hashtable != NULL){
        YJC_params->clustering_hashtable = ((struct YJC_init_params*) params)->clustering_hashtable;
    }
    else{
        printf("build clustering table\n");
        // build clustering hashtable
        GHashTable *clustering_hashtable;
        if (data_type == 'l'){
            // memory leakage problem here
            clustering_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_g_key_value_destroyer, NULL);
            
            // clustering_group_destroyer);
            FILE* file = fopen("clusteringFile", "r");
            char *buf=NULL;
            size_t read_in_size;
            char *token;
            struct clustering_group* cgroup;
            gint64 element;
        
            while ((getline(&buf, &read_in_size, file)) != -1){
                cgroup = g_new0(struct clustering_group, 1);
                cgroup->predictions = g_array_new(FALSE, FALSE, sizeof(guint64));
                token = strtok(buf, "\t");
                while (token!=NULL){
                    element = (gint64)(atol(token));
                    gint64* list_node_content = g_new(gint64, 1);
                    *list_node_content = element;
                    cgroup->list = g_slist_prepend(cgroup->list, (gpointer)list_node_content);
                    token = strtok(NULL, "\t");
                }
                if (cgroup->list){
                    GSList* list_node = cgroup->list;
                    if (g_hash_table_contains(clustering_hashtable, list_node->data))
                        printf("error %ld\n", *( (gint64*) (list_node->data)));
                    g_hash_table_insert(clustering_hashtable, list_node->data, (gpointer)cgroup);
                    while ( (list_node=g_slist_next(list_node))!=NULL){
                        if (g_hash_table_contains(clustering_hashtable, list_node->data))
                            printf("error %ld\n", *( (gint64*) (list_node->data)));
                        g_hash_table_insert(clustering_hashtable, list_node->data, (gpointer)cgroup);
                    }
                }
//            printf("list length %d\n", g_slist_length(list));
                free(buf);
                buf = NULL;
            }
        
        }
        else{
            clustering_hashtable = g_hash_table_new_full(g_str_hash, g_str_equal, NULL, g_slist_destroyer);
            printf("current not supported\n");
            exit(1);
        }
        ((struct YJC_init_params*) params)->clustering_hashtable = clustering_hashtable;
        YJC_params->clustering_hashtable = clustering_hashtable;
    }

    
    return cache;
}



