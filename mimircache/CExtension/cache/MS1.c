//
//  MS1.h
//  MS1cache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h" 
#include "MS1.h"
#include <time.h>
#include <stdlib.h>


//#define PROFILING 1




static inline gboolean __MS1_check_sequential(struct_cache* MS1, cache_line *cp){
    struct MS1_params* MS1_params = (struct MS1_params*)(MS1->cache_params);
    int i;
    if (MS1->core->data_type != 'l')
        printf("ERROR sequential prefetching but data is not long int\n");
    gint64 last = *(gint64*)(cp->item_p);
    gboolean sequential = TRUE;
    gpointer old_item_p = cp->item_p;
    cp->item_p = &last;
    gint sequential_K = MS1_params->sequential_K;
    if (sequential_K == -1)
        /* when use AMP, this is -1 */
        sequential_K = 1;
    for (i=0; i<sequential_K; i++){
        last --;
        if ( !MS1_params->cache->core->check_element(MS1_params->cache, cp) ){
//        if (!g_hash_table_contains( ((struct LRU_params*)(MS1_params->cache->cache_params))->hashtable, &last)){
            sequential = FALSE;
            break;
        }
    }
    cp->item_p = old_item_p;
    return sequential;
}

void __MS1_insert_element(struct_cache* MS1, cache_line* cp){
    struct MS1_params* MS1_params = (struct MS1_params*)(MS1->cache_params);
    MS1_params->cache->core->__insert_element(MS1_params->cache, cp);
}

static inline void __MS1_record_entry(struct_cache* MS1, cache_line* cp){
    struct MS1_params* MS1_params = (struct MS1_params*)(MS1->cache_params);
    
    /** check whether this is a frequent item, if so, don't insert
     *  if the item is not in recording table, but in cache, means it is frequent, discard it   **/
    if (MS1_params->cache->core->check_element(MS1_params->cache, cp) &&
        !g_hash_table_contains(MS1_params->hashtable_for_training, cp->item_p))
        return;

    /* check it is sequtial or not 0925     */
    if (MS1_params->sequential_type && __MS1_check_sequential(MS1, cp))
        return;

    // check the item in hashtable for training
    GSList *list_node = (GSList*) g_hash_table_lookup(MS1_params->hashtable_for_training, cp->item_p);
    if (list_node == NULL){
        // the node is not in the training data, should be added
        struct MS1_training_data_node *data_node = g_new(struct MS1_training_data_node, 1);
        data_node->array = g_new(gint32, MS1_params->max_support);
        data_node->length = 1;

        gpointer item_p;
        if (cp->type == 'l'){
            item_p = (gpointer)g_new(guint64, 1);
            *(guint64*)item_p = *(guint64*)(cp->item_p);
        }
        else
            item_p = (gpointer)g_strdup((gchar*)(cp->item_p));
        data_node->item_p = item_p;

        
        (data_node->array)[0] = (MS1_params->ts) & ((1L<<32)-1);

        MS1_params->training_data = g_slist_prepend(MS1_params->training_data, (gpointer)data_node);
        
        gpointer key;
        if (cp->type == 'l'){
            key = (gpointer)g_new(guint64, 1);
            *(guint64*)key = *(guint64*)(cp->item_p);
        }
        else{
            key = (gpointer)g_strdup((gchar*)(cp->item_p));
        }
        g_hash_table_insert(MS1_params->hashtable_for_training, key, MS1_params->training_data);
    }
    else{
        /** first check how many entries are there in the list,
         *  if less than max support, add to list 
         *  otherwise recycle the list and add item to frequent hashset. 
         **/
        struct MS1_training_data_node* data_node = (struct MS1_training_data_node*)(list_node->data);

        if (data_node->length >= MS1_params->max_support){
            // we need to delete this list node, connect its precursor and successor, free data node inside it,
            MS1_params->training_data = g_slist_remove_link(MS1_params->training_data, list_node);
            
            
            g_hash_table_remove(MS1_params->hashtable_for_training, cp->item_p);
        }
        else{
            (data_node->array)[data_node->length] = (MS1_params->ts) & ((1L<<32)-1);
            (data_node->length)++;
        }
    }
}


static inline void __MS1_prefetch(struct_cache* MS1, cache_line* cp){
    struct MS1_params* MS1_params = (struct MS1_params*)(MS1->cache_params);
    GPtrArray* pArray = (GPtrArray*)g_hash_table_lookup(MS1_params->prefetch_hashtable, cp->item_p);
    if (pArray){
        gpointer old_cp_gp = cp->item_p;
        int i;
        for (i=0; i<pArray->len; i++){
            gpointer gp = g_ptr_array_index(pArray, i);
            cp->item_p = gp;
            
            if (MS1_params->output_statistics)
                MS1_params->num_of_check += 1;
            
            if (MS1_params->cache->core->check_element(MS1_params->cache, cp)){
//                MS1_params->cache->core->add_element(MS1_params->cache, cp);
                continue;
            }
            
//            MS1_params->cache->core->add_element(MS1_params->cache, cp);
            MS1_params->cache->core->__insert_element(MS1_params->cache, cp);
            while (MS1_params->cache->core->get_size(MS1_params->cache) > MS1_params->cache->core->size)
                __MS1_evict_element(MS1, cp);
            
            
            if (MS1_params->output_statistics){
                MS1_params->num_of_prefetch_MS1 += 1;
                
                gpointer item_p;
                if (cp->type == 'l'){
                    item_p = (gpointer)g_new(guint64, 1);
                    *(guint64*)item_p = *(guint64*)(cp->item_p);
                }
                else
                    item_p = (gpointer)g_strdup((gchar*)(cp->item_p));

                g_hash_table_add(MS1_params->prefetched_hashtable_MS1, item_p);
                
                // sanity check
                if (g_hash_table_contains(MS1_params->prefetched_hashtable_MS1, item_p) != g_hash_table_contains(((struct LRU_params*)(MS1_params->cache->cache_params))->hashtable, item_p)){
                    printf("ERROR in prefetched !same in cache\n");
                }
            }
        }
        cp->item_p = old_cp_gp;
    }
    
    
    
    
    
    
    // prefetch sequential      0925
    if (MS1_params->sequential_type == 1 && __MS1_check_sequential(MS1, cp)){
        gpointer old_gp = cp->item_p;
        gint64 next = *(gint64*) (cp->item_p) ;
        cp->item_p = &next;
        if (MS1_params->output_statistics)
            MS1_params->num_of_check += 4;
        
        next ++;
        if (MS1_params->cache->core->check_element(MS1_params->cache, cp)){
            MS1_params->cache->core->add_element(MS1_params->cache, cp);
            cp->item_p = old_gp;
            return;
        }

        MS1_params->cache->core->add_element(MS1_params->cache, cp);
        
        if (MS1_params->output_statistics){
            MS1_params->num_of_prefetch_sequential += 1;
                
            gpointer item_p;
            item_p = (gpointer)g_new(gint64, 1);
            *(gint64*)item_p = next;
            
            g_hash_table_add(MS1_params->prefetched_hashtable_sequential, item_p);
        }
        cp->item_p = old_gp;
    }
}



gboolean MS1_check_element(struct_cache* MS1, cache_line* cp){
     
    // maybe we can move prefetch to insert, meaning prefetch only when cache miss
    // check element, record entry and prefetch element
    struct MS1_params* MS1_params = (struct MS1_params*)(MS1->cache_params);
    if (MS1_params->output_statistics){
        if (g_hash_table_contains(MS1_params->prefetched_hashtable_MS1, cp->item_p)){
            MS1_params->hit_on_prefetch_MS1 += 1;
            g_hash_table_remove(MS1_params->prefetched_hashtable_MS1, cp->item_p);
        }
        if (g_hash_table_contains(MS1_params->prefetched_hashtable_sequential, cp->item_p)){
            MS1_params->hit_on_prefetch_sequential += 1;
            g_hash_table_remove(MS1_params->prefetched_hashtable_sequential, cp->item_p);
        }

    }
    
    gboolean result = MS1_params->cache->core->check_element(MS1_params->cache, cp);
    
    
    __MS1_record_entry(MS1, cp);

    // new 0918 change1, record last
    // check whether this is a frequent item, if so, don't insert
//    if (g_hash_table_contains(MS1_params->hashset_frequentItem, cp->item_p))
//        return result;

//    if (cp->type == 'l'){
//        if (MS1_params->last == NULL)
//            MS1_params->last = MS1_params->last_request_storage;
//        else
//            MS1_add_to_prefetch_table(MS1, MS1_params->last, cp->item_p);
//        
//        *(guint64*)(MS1_params->last) = *(guint64*)(cp->item_p);
//    }
//    else{
//        if (MS1_params->last == NULL)
//            MS1_params->last = MS1_params->last_request_storage;
//        else
//            MS1_add_to_prefetch_table(MS1, MS1_params->last, cp->item_p);
//        
//        strncpy(MS1_params->last_request_storage, cp->item_p, CACHE_LINE_LABEL_SIZE2);
//    }
    
    // sanity check
    if (MS1_params->cache->core->check_element(MS1_params->cache, cp) != result)
        printf("ERROR: not same2 %d, %d ************************\n", result, MS1_params->cache->core->check_element(MS1_params->cache, cp));
    

    return result;
}


void __MS1_update_element(struct_cache* MS1, cache_line* cp){
    struct MS1_params* MS1_params = (struct MS1_params*)(MS1->cache_params);
    MS1_params->cache->core->__update_element(MS1_params->cache, cp);
}


void __MS1_evict_element(struct_cache* MS1, cache_line* cp){
    struct MS1_params* MS1_params = (struct MS1_params*)(MS1->cache_params);
    if (MS1_params->output_statistics){
        gpointer gp;
        gp = MS1_params->cache->core->__evict_element_with_return(MS1_params->cache, cp);
        g_hash_table_remove(MS1_params->prefetched_hashtable_MS1, gp);
        g_hash_table_remove(MS1_params->prefetched_hashtable_sequential, gp);
        g_free(gp);
    }
    else
        MS1_params->cache->core->__evict_element(MS1_params->cache, cp);

}



gboolean MS1_add_element(struct_cache* MS1, cache_line* cp){
    struct MS1_params* MS1_params = (struct MS1_params*)(MS1->cache_params);
    // only support virtual time now
    if (MS1_params->training_period_type == 'v'){        
//        if (MS1_params->ts - MS1_params->last_train_time > MS1_params->training_period){
        if (g_hash_table_size(MS1_params->hashtable_for_training) == MS1_params->training_period){
            

            __MS1_mining(MS1);
            MS1_params->last_train_time = (double)(MS1_params->ts);
            
            
//            counter = 0; g_hash_table_foreach(MS1_params->prefetch_hashtable, prefetch_hashmap_count_length, &counter);
//            printf("training table size %u, prefetch table size %u, ave len: %lf\n\n\n",
//                   g_hash_table_size(MS1_params->hashtable_for_training),
//                   g_hash_table_size(MS1_params->prefetch_hashtable),
//                   (double) counter / g_hash_table_size(MS1_params->prefetch_hashtable));
        }
    }
    else if (MS1_params->training_period_type == 'r'){
        printf("currently don't support real time in MS1_add_element\n");
        exit(1);
    }
    else{
        fprintf(stderr, "cannot recognize given training period type: %c in MS1_add_element\n",
                MS1_params->training_period_type);
        exit(1);
    }

    
    MS1_params->ts ++;
    if (MS1_params->cache->core->type == e_AMP){
        gboolean result = MS1_check_element(MS1, cp);
        MS1_params->cache->core->add_element(MS1_params->cache, cp);
        __MS1_prefetch(MS1, cp);
        while ( MS1_params->cache->core->get_size(MS1_params->cache) > MS1->core->size)
            __MS1_evict_element(MS1, cp);

        return result;
    }
    else{
        if (MS1_check_element(MS1, cp)){
            __MS1_update_element(MS1, cp);
            __MS1_prefetch(MS1, cp);
            return TRUE;
        }
        else{
            __MS1_insert_element(MS1, cp);
            __MS1_prefetch(MS1, cp);
            while ( MS1_params->cache->core->get_size(MS1_params->cache) > MS1->core->size)
                __MS1_evict_element(MS1, cp);
            return FALSE;
        }
    }
}




void MS1_destroy(struct_cache* cache){
    struct MS1_params* MS1_params = (struct MS1_params*)(cache->cache_params);

    g_hash_table_destroy(MS1_params->prefetch_hashtable);
    g_hash_table_destroy(MS1_params->hashtable_for_training);
    g_hash_table_destroy(MS1_params->hashset_frequentItem);
//    g_slist_free(MS1_params->training_data);                       // data have already been freed by hashtable
    if (MS1_params->output_statistics){
        g_hash_table_destroy(MS1_params->prefetched_hashtable_MS1);
        g_hash_table_destroy(MS1_params->prefetched_hashtable_sequential);
    }
    MS1_params->cache->core->destroy(MS1_params->cache);            // 0921
//    g_free( ((struct MS1_init_params*)(cache->core->cache_init_params))->cache_type);
    cache_destroy(cache);
}

void MS1_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    struct MS1_params* MS1_params = (struct MS1_params*)(cache->cache_params);
    g_hash_table_destroy(MS1_params->prefetch_hashtable);
    g_hash_table_destroy(MS1_params->hashtable_for_training);
    g_hash_table_destroy(MS1_params->hashset_frequentItem);
    if (MS1_params->output_statistics){
        g_hash_table_destroy(MS1_params->prefetched_hashtable_MS1);
        g_hash_table_destroy(MS1_params->prefetched_hashtable_sequential);
    }
    MS1_params->cache->core->destroy_unique(MS1_params->cache);         // 0921
    cache_destroy_unique(cache);
}


struct_cache* MS1_init(guint64 size, char data_type, void* params){
    struct_cache *cache = cache_init(size, data_type);
    cache->cache_params = g_new0(struct MS1_params, 1);
    struct MS1_params* MS1_params = (struct MS1_params*)(cache->cache_params);
    
    cache->core->type = e_MS1;
    cache->core->cache_init = MS1_init;
    cache->core->destroy = MS1_destroy;
    cache->core->destroy_unique = MS1_destroy_unique;
    cache->core->add_element = MS1_add_element;
    cache->core->check_element = MS1_check_element;
    cache->core->cache_init_params = params;
    
    struct MS1_init_params *init_params = (struct MS1_init_params*) params;
    
    if (strcmp(init_params->cache_type, "LRU") == 0)
        MS1_params->cache = LRU_init(size, data_type, NULL);
    else if (strcmp(init_params->cache_type, "FIFO") == 0)
        MS1_params->cache = fifo_init(size, data_type, NULL);
    else if (strcmp(init_params->cache_type, "AMP") == 0){
        struct AMP_init_params *AMP_init_params = g_new0(struct AMP_init_params, 1);
        AMP_init_params->APT = 4;
        AMP_init_params->p_threshold = 256;
        AMP_init_params->read_size = 1;
        MS1_params->cache = AMP_init(size, data_type, AMP_init_params);
    }
    else if (strcmp(init_params->cache_type, "Optimal") == 0){
        struct optimal_init_params *optimal_init_params = g_new(struct optimal_init_params, 1);
        optimal_init_params->reader = NULL;
        MS1_params->cache = NULL;
        ;
    }
    else{
        fprintf(stderr, "can't recognize cache type: %s\n", init_params->cache_type); 
        MS1_params->cache = LRU_init(size, data_type, NULL);
    }

    
    MS1_params->item_set_size = init_params->item_set_size;
    MS1_params->max_support = init_params->max_support;
    MS1_params->min_support = init_params->min_support;
    MS1_params->confidence = init_params->confidence;
    MS1_params->prefetch_list_size = init_params->prefetch_list_size;
    MS1_params->sequential_type = init_params->sequential_type; 
    MS1_params->sequential_K = init_params->sequential_K;
    MS1_params->training_period = init_params->training_period;
    MS1_params->prefetch_table_size = init_params->prefetch_table_size;
    MS1_params->training_period_type = init_params->training_period_type;
    MS1_params->ts = 0;
    MS1_params->last_train_time = 0;

    MS1_params->output_statistics = init_params->output_statistics;
    MS1_params->output_statistics = 1;
    
    MS1_params->hit_on_prefetch_MS1 = 0;
    MS1_params->hit_on_prefetch_sequential = 0;
    MS1_params->num_of_prefetch_MS1 = 0;
    MS1_params->num_of_prefetch_sequential = 0;
    MS1_params->num_of_check = 0;
    
    

    if (data_type == 'l'){
        MS1_params->hashtable_for_training = g_hash_table_new_full(g_int64_hash, g_int_equal,
                                                                     simple_g_key_value_destroyer,
                                                                     MS1_training_node_destroyer);
        MS1_params->hashset_frequentItem = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                                   simple_g_key_value_destroyer,
                                                                   NULL);
        MS1_params->prefetch_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                                 simple_g_key_value_destroyer,
                                                                 prefetch_node_destroyer);
        
        if (MS1_params->output_statistics){
            MS1_params->prefetched_hashtable_MS1 = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                                             simple_g_key_value_destroyer,
                                                                             NULL);
            MS1_params->prefetched_hashtable_sequential = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                                             simple_g_key_value_destroyer,
                                                                             NULL);
        
        }
    }
    else if (data_type == 'c'){
        MS1_params->hashtable_for_training = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                     simple_g_key_value_destroyer,
                                                                     MS1_training_node_destroyer);
        MS1_params->hashset_frequentItem = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                   simple_g_key_value_destroyer,
                                                                   NULL);
        MS1_params->prefetch_hashtable = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                 simple_g_key_value_destroyer,
                                                                 prefetch_node_destroyer);
        if (MS1_params->output_statistics){
            MS1_params->prefetched_hashtable_MS1 = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                             simple_g_key_value_destroyer,
                                                                             NULL);
            MS1_params->prefetched_hashtable_sequential = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                                  simple_g_key_value_destroyer,
                                                                                  NULL);
        }
    }
    else{
        g_error("does not support given data type: %c\n", data_type);
    }
    
    return cache;
}


void __MS1_mining(struct_cache* MS1){
    struct MS1_params* MS1_params = (struct MS1_params*)(MS1->cache_params);
#ifdef PROFILING
    GTimer *timer = g_timer_new();
    gulong microsecond;
    g_timer_start(timer);
#endif 
    
    __MS1_aging(MS1);
#ifdef PROFILING
    printf("ts: %lu, aging takes %lf seconds\n", MS1_params->ts, g_timer_elapsed(timer, &microsecond));
#endif
    GSList* list_node1 = MS1_params->training_data;
    GSList* list_node2;
    struct MS1_training_data_node* data_node1, *data_node2;
    while (list_node1){
        
        data_node1 = (struct MS1_training_data_node*)(list_node1->data);
        if (data_node1->length < MS1_params->min_support){
            list_node1 = list_node1->next;
            continue;
        }
        list_node2 = list_node1->next;
        gboolean first_flag = TRUE;
        while (list_node2){
            data_node2 = (struct MS1_training_data_node*)(list_node2->data);
            if (data_node1->array[0] - data_node2->array[0] > MS1_params->item_set_size)
                break;
            /** We can have better approximation for association rule mining here, maybe using bitwise op, 
             *  and capture the situation where sequence 1 and 2 are of different length, but 
             *  1->2 is 100% confident, 2->1 is not.
             **/
            gint8 smaller_length = data_node1->length;
            if (data_node2->length < smaller_length)
                smaller_length = data_node2->length;
            if (data_node2->length < MS1_params->min_support ||
                        abs(data_node2->length - data_node1->length) > MS1_params->confidence){
                list_node2 = list_node2->next;
                continue;
            }
            else{
                int i;
                gboolean associated_flag = FALSE;            // new0918 change1
                // new 0920 change1
                if (first_flag){
                    associated_flag = TRUE;
                    first_flag = FALSE;
                }
                if (smaller_length == 1 && labs(data_node2->array[0] - data_node1->array[0]) == 1)
                    associated_flag = TRUE;
                
                
                for (i=1; i<smaller_length; i++){
                    if ( labs(data_node2->array[i] - data_node1->array[i]) > MS1_params->item_set_size){
                        associated_flag = FALSE;             // new0918 change1 
                        break;
                    }
                    // new0918 change1
                    if ( labs(data_node2->array[i] - data_node1->array[i]) == 1 ){
                        associated_flag = TRUE;
                    }
                    
                }
                if (associated_flag){
                    // finally, add to prefetch table
                    MS1_add_to_prefetch_table(MS1, data_node2->item_p, data_node1->item_p);
//                    MS1_add_to_prefetch_table(MS1, data_node1->item_p, data_node2->item_p);
                
                
                }
                
            }       // checking associated
            
            list_node2 = list_node2->next;
            
        }           // while list_node2

        list_node1 = list_node1->next;
        
    }               // while list_node1

#ifdef PROFILING
    printf("ts: %lu, training takes %lf seconds\n", MS1_params->ts, g_timer_elapsed(timer, &microsecond));
#endif
    
    /* clearing training hashtable and training list */
    GHashTable *old_hashtable = MS1_params->hashtable_for_training;
    
    // add the last record into new record table
    if (MS1->core->data_type == 'l'){
        MS1_params->hashtable_for_training = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                                     simple_g_key_value_destroyer,
                                                                     MS1_training_node_destroyer);
        struct MS1_training_data_node *last_data_node = (struct MS1_training_data_node*) (MS1_params->training_data->data);
        struct MS1_training_data_node *data_node = g_new(struct MS1_training_data_node, 1);
        data_node->array = g_new(gint32, MS1_params->max_support);
        data_node->length = 1;
        
        gpointer item_p;
        item_p = (gpointer)g_new(guint64, 1);
        *(guint64*)item_p = *(guint64*)(last_data_node->item_p);
        data_node->item_p = item_p;
        
        
        (data_node->array)[0] = last_data_node->array[last_data_node->length-1];
        
        MS1_params->training_data = NULL;
        MS1_params->training_data = g_slist_prepend(MS1_params->training_data, (gpointer)data_node);
        
        
        gpointer key;
        key = (gpointer)g_new(guint64, 1);
        *(guint64*)key = *(guint64*)(last_data_node->item_p);
        g_hash_table_insert(MS1_params->hashtable_for_training, key, MS1_params->training_data);

    
    }
    else{
        MS1_params->hashtable_for_training = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                     simple_g_key_value_destroyer,
                                                                     MS1_training_node_destroyer);
        struct MS1_training_data_node *last_data_node = (struct MS1_training_data_node*) (MS1_params->training_data->data);
        struct MS1_training_data_node *data_node = g_new(struct MS1_training_data_node, 1);
        data_node->array = g_new(gint32, MS1_params->max_support);
        data_node->length = 1;
        
        gpointer item_p;
        item_p = (gpointer)g_strdup((gchar*)(last_data_node->item_p));
        data_node->item_p = item_p;
        
        
        (data_node->array)[0] = last_data_node->array[last_data_node->length-1];
        
        MS1_params->training_data = NULL;
        MS1_params->training_data = g_slist_prepend(MS1_params->training_data, (gpointer)data_node);
        
        gpointer key;
        key = (gpointer)g_strdup((gchar*)(last_data_node->item_p));
        g_hash_table_insert(MS1_params->hashtable_for_training, key, MS1_params->training_data);

        
    }
    g_hash_table_destroy(old_hashtable);                                // do not need to free the list again
    
   
#ifdef PROFILING
    printf("ts: %lu, clearing training data takes %lf seconds\n", MS1_params->ts, g_timer_elapsed(timer, &microsecond));
    g_timer_stop(timer);
    g_timer_destroy(timer);
#endif
}



void MS1_add_to_prefetch_table(struct_cache* MS1, gpointer gp1, gpointer gp2){
    struct MS1_params* MS1_params = (struct MS1_params*)(MS1->cache_params);
    
    GPtrArray* pArray = g_hash_table_lookup(MS1_params->prefetch_hashtable,
                                            gp1
                                            );
//    printf("add %s %s\n", gp1, gp2); 
    /** if we want to save space here, we can reuse the pointer from hashtable
     *  This can cut the memory usage to 1/3, but involves a hashtable look up
     *  and complicated memory free problem *******************************
     **/
    
    gpointer key2;
    if (MS1->core->data_type == 'l'){
        key2 = (gpointer)g_new(guint64, 1);
        *(guint64*)key2 = *(guint64*)(gp2);
    }
    else{
        key2 = (gpointer)g_strdup((gchar*)gp2);
    }
    
    
    // insert into prefetch hashtable
    int i;
    if (pArray){
        gboolean insert = TRUE;
        if (MS1->core->data_type == 'l'){
            for (i=0; i<pArray->len; i++)
                // if this element is already in the array, then don't need prefetch again
                if (*(guint64*)(g_ptr_array_index(pArray, i)) == *(guint64*)(gp2)){
                    /* update score here, not implemented yet */
                    insert = FALSE;
                    g_free(key2);
                }
        }
        else{
            for (i=0; i<pArray->len; i++)
                // if this element is already in the cache, then don't need prefetch again
                if ( strcmp((gchar*)(g_ptr_array_index(pArray, i)), (gchar*)gp2) == 0 ){
                    /* update score here, not implemented yet */
                    insert = FALSE;
                    g_free(key2);
                }
        }
        
        if (insert){
            if (pArray->len >= MS1_params->prefetch_list_size){
                int p = rand()%MS1_params->prefetch_list_size;
                // free the content ??????????????
//                g_free(g_ptr_array_index(pArray, p));
                g_ptr_array_remove_index_fast(pArray, p);
                
            }
            g_ptr_array_add(pArray, key2);
        }
    }
    else{
        pArray = g_ptr_array_new_with_free_func(prefetch_array_node_destroyer);
        g_ptr_array_add(pArray, key2);
        gpointer gp1_dup;
        if (MS1->core->data_type == 'l'){
            gp1_dup = (gpointer)g_new(guint64, 1);
            *(guint64*)gp1_dup = *(guint64*)(gp1);
        }
        else{
            gp1_dup = (gpointer)g_strdup((gchar*)(gp1));
        }
        
        g_hash_table_insert(MS1_params->prefetch_hashtable, gp1_dup, pArray);
    }
    
    if (g_hash_table_size(MS1_params->prefetch_hashtable) > MS1_params->prefetch_table_size){
        ;
    }
    
}





void __MS1_aging(struct_cache* MS1){
    
    ;
}

void MS1_training_node_destroyer(gpointer data){
    GList* list_node = (GList*)data;
    struct MS1_training_data_node *data_node = (struct MS1_training_data_node*) (list_node->data);
    g_free(data_node->array);
    g_free(data_node->item_p);
    g_free(data_node);
    g_list_free_1(list_node);
}

