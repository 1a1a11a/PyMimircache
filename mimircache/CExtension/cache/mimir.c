//
//  mimir.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h" 
#include "mimir.h"
#include <time.h>
#include <stdlib.h>


//#define PROFILING 1




void
prefetch_hashmap_count_length (gpointer key, gpointer value, gpointer user_data){
    gint64* pArray = (gint64*) value;
//    gint prefetch_list_size = *(gint*)user_data; 
    guint64 *counter = (guint64*) user_data;
//    (*counter) += pArray->len;
}

void check_prefetched_hashtable (gpointer key, gpointer value, gpointer user_data){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*) user_data;
    GHashTable *h = ((struct LRU_params*)(MIMIR_params->cache->cache_params))->hashtable;
    if (!g_hash_table_contains(h, key)){
        printf("ERROR in prefetched, not in cache %ld, %d\n", *(gint64*)key, GPOINTER_TO_INT(value));
        exit(1);
    }
}


void training_hashtable_count_length(gpointer key, gpointer value, gpointer user_data){
    GList* list_node = (GList*) value;
    struct training_data_node *data_node = (struct training_data_node*) (list_node->data);
    guint64 *counter = (guint64*) user_data;
    (*counter) += data_node->length;
}

void mprintHashTable(gpointer key, gpointer value, gpointer userdata){
    printf("key %lu, array length %u\n", *(guint64*)key, ((GPtrArray*)value)->len);
}



static inline gboolean __MIMIR_check_sequential(struct_cache* MIMIR, cache_line *cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    int i;
    if (MIMIR->core->data_type != 'l')
        printf("ERROR sequential prefetching but data is not long int\n");
    gint64 last = *(gint64*)(cp->item_p);
    gboolean sequential = TRUE;
    gpointer old_item_p = cp->item_p;
    cp->item_p = &last;
    gint sequential_K = MIMIR_params->sequential_K;
    if (sequential_K == -1)
        /* when use AMP, this is -1 */
        sequential_K = 1;
    for (i=0; i<sequential_K; i++){
        last --;
        if ( !MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp) ){
//        if (!g_hash_table_contains( ((struct LRU_params*)(MIMIR_params->cache->cache_params))->hashtable, &last)){
            sequential = FALSE;
            break;
        }
    }
    cp->item_p = old_item_p;
    return sequential;
}

void __MIMIR_insert_element(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    MIMIR_params->cache->core->__insert_element(MIMIR_params->cache, cp);
}

static inline void __MIMIR_record_entry(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    
    /** check whether this is a frequent item, if so, don't insert
     *  if the item is not in recording table, but in cache, means it is frequent, discard it   **/
    if (MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp) &&
        !g_hash_table_contains(MIMIR_params->hashtable_for_training, cp->item_p))
        return;

    /* check it is sequtial or not 0925     */
    if (MIMIR_params->sequential_type && __MIMIR_check_sequential(MIMIR, cp))
        return;

    // check the item in hashtable for training
    GList *list_node = (GList*) g_hash_table_lookup(MIMIR_params->hashtable_for_training, cp->item_p);
    if (list_node == NULL){
        // the node is not in the mining data, should be added
        struct training_data_node *data_node = g_new(struct training_data_node, 1);
        data_node->array = g_new(gint32, MIMIR_params->max_support);
        data_node->length = 1;

        gpointer item_p;
        if (cp->type == 'l'){
            item_p = (gpointer)g_new(gint64, 1);
            *(gint64*)item_p = *(gint64*)(cp->item_p);
        }
        else
            item_p = (gpointer)g_strdup((gchar*)(cp->item_p));
        data_node->item_p = item_p;

        
        (data_node->array)[0] = (MIMIR_params->ts) & ((1L<<16)-1);

        MIMIR_params->training_data = g_list_prepend(MIMIR_params->training_data, (gpointer)data_node);
        
        gpointer key;
        if (cp->type == 'l'){
            key = (gpointer)g_new(guint64, 1);
            *(guint64*)key = *(guint64*)(cp->item_p);
        }
        else{
            key = (gpointer)g_strdup((gchar*)(cp->item_p));
        }
        g_hash_table_insert(MIMIR_params->hashtable_for_training, key, MIMIR_params->training_data);
    }
    else{
        /** first check how many entries are there in the list,
         *  if less than max support, add to list 
         *  otherwise recycle the list and add item to frequent hashset. 
         **/
        struct training_data_node* data_node = (struct training_data_node*)(list_node->data);

        if (data_node->length >= MIMIR_params->max_support){
            // we need to delete this list node, connect its precursor and successor, free data node inside it,
            MIMIR_params->training_data = g_list_remove_link(MIMIR_params->training_data, list_node);
            g_hash_table_remove(MIMIR_params->hashtable_for_training, cp->item_p);
            MIMIR_params->num_of_entry_available_for_training --;
        }
        else{
            (data_node->array)[data_node->length] = (MIMIR_params->ts) & ((1L<<16)-1);
            (data_node->length)++;
            if (MIMIR_params->min_support == 1){
                if (data_node->length == MIMIR_params->min_support+1)
                    MIMIR_params->num_of_entry_available_for_training ++;
            }
            else{
                if (data_node->length == MIMIR_params->min_support)
                    MIMIR_params->num_of_entry_available_for_training ++;
            }
        }
    }
}


static inline void __MIMIR_prefetch(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    gint prefetch_table_index = GPOINTER_TO_INT(g_hash_table_lookup(MIMIR_params->prefetch_hashtable, cp->item_p));
    gint dim1 = floor(prefetch_table_index/PREFETCH_TABLE_SHARD_SIZE);
    gint dim2 = prefetch_table_index % PREFETCH_TABLE_SHARD_SIZE * (MIMIR_params->prefetch_list_size+1);
    
    if (prefetch_table_index){
        gpointer old_cp_gp = cp->item_p;
        int i;
        for (i=1; i<MIMIR_params->prefetch_list_size+1; i++){
            if (MIMIR_params->prefetch_table_array[dim1][dim2+i] == 0)
                break;
            if (cp->type == 'l')
                cp->item_p = &(MIMIR_params->prefetch_table_array[dim1][dim2+i]);
            else
                cp->item_p = (char*) (MIMIR_params->prefetch_table_array[dim1][dim2+i]);

            if (MIMIR_params->output_statistics)
                MIMIR_params->num_of_check += 1;
            
            // can't use MIMIR_check_element here
            if (MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp)){
                continue;
            }
            
            MIMIR_params->cache->core->__insert_element(MIMIR_params->cache, cp);
            while (MIMIR_params->cache->core->get_size(MIMIR_params->cache) > MIMIR_params->cache->core->size)
                // use this because we need to record stat when evicting 
                __MIMIR_evict_element(MIMIR, cp);
            
            
            if (MIMIR_params->output_statistics){
                MIMIR_params->num_of_prefetch_mimir += 1;
                
                gpointer item_p;
                if (cp->type == 'l'){
                    item_p = (gpointer)g_new(guint64, 1);
                    *(guint64*)item_p = *(guint64*)(cp->item_p);
                }
                else
                    item_p = (gpointer)g_strdup((gchar*)(cp->item_p));

//                g_hash_table_add(MIMIR_params->prefetched_hashtable_mimir, item_p);
                g_hash_table_insert(MIMIR_params->prefetched_hashtable_mimir, item_p, GINT_TO_POINTER(1));
            }
        }
        cp->item_p = old_cp_gp;
    }
    
    
    
    
    
    
    // prefetch sequential      0925
    if (MIMIR_params->sequential_type == 1 && __MIMIR_check_sequential(MIMIR, cp)){
        gpointer old_gp = cp->item_p;
        gint64 next = *(gint64*) (cp->item_p) + 1;
        cp->item_p = &next;
//        if (MIMIR_params->output_statistics)
//            MIMIR_params->num_of_check += 1;
        
//        next ++;
        if (MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp)){
            MIMIR_params->cache->core->__update_element(MIMIR_params->cache, cp);
            cp->item_p = old_gp;
            return;
        }

        // use this because we need to record stat when evicting
        MIMIR_params->cache->core->__insert_element(MIMIR_params->cache, cp);
        while (MIMIR_params->cache->core->get_size(MIMIR_params->cache) > MIMIR_params->cache->core->size)
            __MIMIR_evict_element(MIMIR, cp);
        
        
        if (MIMIR_params->output_statistics){
            MIMIR_params->num_of_prefetch_sequential += 1;
                
            gpointer item_p;
            item_p = (gpointer)g_new(gint64, 1);
            *(gint64*)item_p = next;
            
            g_hash_table_add(MIMIR_params->prefetched_hashtable_sequential, item_p);
        }
        cp->item_p = old_gp;
    }
}



gboolean MIMIR_check_element(struct_cache* MIMIR, cache_line* cp){
     
    // maybe we can move prefetch to insert, meaning prefetch only when cache miss
    // check element, record entry and prefetch element
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    if (MIMIR_params->output_statistics){
        if (g_hash_table_contains(MIMIR_params->prefetched_hashtable_mimir, cp->item_p)){
            MIMIR_params->hit_on_prefetch_mimir += 1;
            g_hash_table_remove(MIMIR_params->prefetched_hashtable_mimir, cp->item_p);
        }
        if (g_hash_table_contains(MIMIR_params->prefetched_hashtable_sequential, cp->item_p)){
            MIMIR_params->hit_on_prefetch_sequential += 1;
            g_hash_table_remove(MIMIR_params->prefetched_hashtable_sequential, cp->item_p);
        }

    }
    
    gboolean result = MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp);
    
    
    __MIMIR_record_entry(MIMIR, cp);

    // new 0918 change1, record last
    // check whether this is a frequent item, if so, don't insert
//    if (g_hash_table_contains(MIMIR_params->hashset_frequentItem, cp->item_p))
//        return result;

//    if (cp->type == 'l'){
//        if (MIMIR_params->last == NULL)
//            MIMIR_params->last = MIMIR_params->last_request_storage;
//        else
//            mimir_add_to_prefetch_table(MIMIR, MIMIR_params->last, cp->item_p);
//        
//        *(guint64*)(MIMIR_params->last) = *(guint64*)(cp->item_p);
//    }
//    else{
//        if (MIMIR_params->last == NULL)
//            MIMIR_params->last = MIMIR_params->last_request_storage;
//        else
//            mimir_add_to_prefetch_table(MIMIR, MIMIR_params->last, cp->item_p);
//        
//        strncpy(MIMIR_params->last_request_storage, cp->item_p, CACHE_LINE_LABEL_SIZE2);
//    }
    
    // sanity check
    if (MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp) != result)
        printf("ERROR: not same2 %d, %d ************************\n", result, MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp));
    

    return result;
}


void __MIMIR_update_element(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    MIMIR_params->cache->core->__update_element(MIMIR_params->cache, cp);
}


void __MIMIR_evict_element(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
//    g_hash_table_foreach(MIMIR_params->prefetched_hashtable_mimir, check_prefetched_hashtable, MIMIR_params);
    if (MIMIR_params->output_statistics){
        gpointer gp;
        gp = MIMIR_params->cache->core->__evict_element_with_return(MIMIR_params->cache, cp);

        gint type = GPOINTER_TO_INT(g_hash_table_lookup(MIMIR_params->prefetched_hashtable_mimir, gp));
        if (type !=0 && type < MIMIR_params->cycle_time){
//            printf("give one more chance \n");
            gpointer new_key;
            if (cp->type == 'l'){
                new_key = g_new(gint64, 1);
                *(gint64*) new_key = *(gint64*) gp;
            }
            else if (cp->type == 'c'){
                new_key = g_strdup(gp);
            }
            g_hash_table_insert(MIMIR_params->prefetched_hashtable_mimir, new_key, GINT_TO_POINTER(type+1));
            gpointer old_cp = cp->item_p;
            cp->item_p = gp;
            __MIMIR_insert_element(MIMIR, cp);
            cp->item_p = old_cp;
            __MIMIR_evict_element(MIMIR, cp);
        }
        else{

//            if (g_hash_table_contains(MIMIR_params->prefetched_hashtable_mimir, gp)){
//                MIMIR_params->evicted_prefetch ++;
//            }

        g_hash_table_remove(MIMIR_params->prefetched_hashtable_mimir, gp);
        g_hash_table_remove(MIMIR_params->prefetched_hashtable_sequential, gp);
        }
        g_free(gp);
    }
    else
        MIMIR_params->cache->core->__evict_element(MIMIR_params->cache, cp);
}



gboolean MIMIR_add_element(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    // only support virtual time now
    static int c = 0;
    if (MIMIR_params->training_period_type == 'v'){
//        if (MIMIR_params->ts - MIMIR_params->last_train_time > MIMIR_params->training_period){
//        if (g_hash_table_size(MIMIR_params->hashtable_for_training) >= MIMIR_params->training_period){
        if (MIMIR_params->num_of_entry_available_for_training > MIMIR_params->mining_threshold ||
            (MIMIR_params->min_support == 1 && MIMIR_params->num_of_entry_available_for_training > MINING_THRESHOLD/8)){
            gint count = 0;
//            g_hash_table_foreach(MIMIR_params->hashtable_for_training, training_hashtable_count_length, &count);
            printf("mining %u %lf %d\n", g_hash_table_size(MIMIR_params->hashtable_for_training), (double)count/g_hash_table_size(MIMIR_params->hashtable_for_training), c++);
            __MIMIR_mining(MIMIR);
            MIMIR_params->last_train_time = (double)(MIMIR_params->ts);
            MIMIR_params->num_of_entry_available_for_training = 0;
            MIMIR_params->residual_mining_table_size = (MIMIR_params->min_support * 2 + 8 + 4) *
                                                        g_hash_table_size(MIMIR_params->hashtable_for_training);
            MIMIR_params->cache->core->size = MIMIR->core->size -
                                                (gint)((MIMIR_params->current_metadata_size +
                                                        MIMIR_params-> residual_mining_table_size)
                                                       /MIMIR_params->block_size);
                        printf("residual hashtable %u\n", g_hash_table_size(MIMIR_params->hashtable_for_training));

            
//            counter = 0; g_hash_table_foreach(MIMIR_params->prefetch_hashtable, prefetch_hashmap_count_length, &counter);
//            printf("training table size %u, prefetch table size %u, ave len: %lf\n\n\n",
//                   g_hash_table_size(MIMIR_params->hashtable_for_training),
//                   g_hash_table_size(MIMIR_params->prefetch_hashtable),
//                   (double) counter / g_hash_table_size(MIMIR_params->prefetch_hashtable));
        }
    }
    else if (MIMIR_params->training_period_type == 'r'){
        printf("currently don't support real time in MIMIR_add_element\n");
        exit(1);
    }
    else{
        fprintf(stderr, "cannot recognize given training period type: %c in MIMIR_add_element\n",
                MIMIR_params->training_period_type);
        exit(1);
    }

    
    MIMIR_params->ts ++;
    if (MIMIR_params->cache->core->type == e_AMP){
        gboolean result = MIMIR_check_element(MIMIR, cp);
//        MIMIR_params->cache->core->add_element(MIMIR_params->cache, cp);
        AMP_add_element_no_eviction(MIMIR_params->cache, cp);
        __MIMIR_prefetch(MIMIR, cp);
        while ( MIMIR_params->cache->core->get_size(MIMIR_params->cache) > MIMIR->core->size)
            __MIMIR_evict_element(MIMIR, cp);

        return result;
    }
    else{
        if (MIMIR_check_element(MIMIR, cp)){
            __MIMIR_update_element(MIMIR, cp);
            __MIMIR_prefetch(MIMIR, cp);
            return TRUE;
        }
        else{
            __MIMIR_insert_element(MIMIR, cp);
            __MIMIR_prefetch(MIMIR, cp);
            while ( MIMIR_params->cache->core->get_size(MIMIR_params->cache) > MIMIR->core->size)
                __MIMIR_evict_element(MIMIR, cp);
            return FALSE;
        }
    }
}




void MIMIR_destroy(struct_cache* cache){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(cache->cache_params);

    g_hash_table_destroy(MIMIR_params->prefetch_hashtable);
    g_hash_table_destroy(MIMIR_params->hashtable_for_training);
    g_hash_table_destroy(MIMIR_params->hashset_frequentItem);
    int i = 0;
    while (1){
        if (MIMIR_params->prefetch_table_array[i]){
            if (cache->core->data_type == 'c'){
                int j=0;
                for (j=0; j<PREFETCH_TABLE_SHARD_SIZE*(1+MIMIR_params->prefetch_list_size); j++)
                    g_free((char*)MIMIR_params->prefetch_table_array[i][j]);
            }
            g_free(MIMIR_params->prefetch_table_array[i]);
        }
        else
            break;
        i++;
    }
    g_free(MIMIR_params->prefetch_table_array);
    
//    g_list_free(MIMIR_params->training_data);                       // data have already been freed by hashtable
    if (MIMIR_params->output_statistics){
        g_hash_table_destroy(MIMIR_params->prefetched_hashtable_mimir);
        g_hash_table_destroy(MIMIR_params->prefetched_hashtable_sequential);
    }
    MIMIR_params->cache->core->destroy(MIMIR_params->cache);            // 0921
//    g_free( ((struct MIMIR_init_params*)(cache->core->cache_init_params))->cache_type);
    cache_destroy(cache);
}

void MIMIR_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(cache->cache_params);
    g_hash_table_destroy(MIMIR_params->prefetch_hashtable);
    g_hash_table_destroy(MIMIR_params->hashtable_for_training);
    g_hash_table_destroy(MIMIR_params->hashset_frequentItem);
    int i = 0;
    while (1){
        if (MIMIR_params->prefetch_table_array[i]){
            if (cache->core->data_type == 'c'){
                int j=0;
                for (j=0; j<PREFETCH_TABLE_SHARD_SIZE*(1+MIMIR_params->prefetch_list_size); j++)
                    g_free((char*)MIMIR_params->prefetch_table_array[i][j]);
            }
            g_free(MIMIR_params->prefetch_table_array[i]);
        }
        else
            break;
        i++;
    }
    g_free(MIMIR_params->prefetch_table_array);

    if (MIMIR_params->output_statistics){
        g_hash_table_destroy(MIMIR_params->prefetched_hashtable_mimir);
        g_hash_table_destroy(MIMIR_params->prefetched_hashtable_sequential);
    }
    MIMIR_params->cache->core->destroy_unique(MIMIR_params->cache);         // 0921
    cache_destroy_unique(cache);
}


struct_cache* MIMIR_init(guint64 size, char data_type, void* params){
    struct_cache *cache = cache_init(size, data_type);
    cache->cache_params = g_new0(struct MIMIR_params, 1);
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(cache->cache_params);
    
    cache->core->type               = e_mimir;
    cache->core->cache_init         = MIMIR_init;
    cache->core->destroy            = MIMIR_destroy;
    cache->core->destroy_unique     = MIMIR_destroy_unique;
    cache->core->add_element        = MIMIR_add_element;
    cache->core->check_element      = MIMIR_check_element;
    cache->core->cache_init_params  = params;
    
    struct MIMIR_init_params *init_params = (struct MIMIR_init_params*) params;
    
    
    MIMIR_params->item_set_size     = init_params->item_set_size;
    MIMIR_params->max_support       = init_params->max_support;
    MIMIR_params->min_support       = init_params->min_support;
    MIMIR_params->confidence        = init_params->confidence;
    MIMIR_params->prefetch_list_size= init_params->prefetch_list_size;
    MIMIR_params->sequential_type   = init_params->sequential_type;
    MIMIR_params->sequential_K      = init_params->sequential_K;
    MIMIR_params->training_period   = init_params->training_period;
    MIMIR_params->cycle_time        = init_params->cycle_time;
//    MIMIR_params->mining_threshold  = (gint)(MINING_THRESHOLD / pow(2, MIMIR_params->min_support));
    MIMIR_params->mining_threshold  = (gint)(MINING_THRESHOLD/MIMIR_params->min_support);
    
//    MIMIR_params->prefetch_table_size = init_params->prefetch_table_size;
    MIMIR_params->max_metadata_size = (gint64) (init_params->block_size * size * init_params->max_metadata_size);
    MIMIR_params->block_size = init_params->block_size;
    gint max_num_of_shards_in_prefetch_table = (gint) (MIMIR_params->max_metadata_size /
                                                       (PREFETCH_TABLE_SHARD_SIZE * init_params->prefetch_list_size));
    MIMIR_params->current_prefetch_table_pointer = 0;
    MIMIR_params->prefetch_table_fully_allocatd = FALSE;
    // always save to size+1 position, and enlarge table when size%shards_size == 0
    MIMIR_params->prefetch_table_array = g_new0(gint64*, max_num_of_shards_in_prefetch_table);
    MIMIR_params->prefetch_table_array[0] = g_new0(gint64, PREFETCH_TABLE_SHARD_SIZE*(MIMIR_params->prefetch_list_size+1));
    
    /* now adjust the cache size by deducting current meta data size
        8 is the size of storage for block, 4 is the size of storage for index to array */
    MIMIR_params->current_metadata_size = (init_params->max_support * 2 + 8 + 4) * MIMIR_params->mining_threshold +
                                            max_num_of_shards_in_prefetch_table * 8 +
                                            PREFETCH_TABLE_SHARD_SIZE * (MIMIR_params->prefetch_list_size*8 + 8 + 4) ;
    
    size = size - (gint)(MIMIR_params->current_metadata_size/init_params->block_size);
//    printf("size decrease %d\n", (gint)(MIMIR_params->current_metadata_size/init_params->block_size));
    
    
    MIMIR_params->training_period_type = init_params->training_period_type;
    MIMIR_params->ts = 0;
    MIMIR_params->num_of_entry_available_for_training = 0;
    MIMIR_params->last_train_time = 0;

    MIMIR_params->output_statistics = init_params->output_statistics;
    MIMIR_params->output_statistics = 1;
    
    
    MIMIR_params->evicted_prefetch = 0;
    MIMIR_params->hit_on_prefetch_mimir = 0;
    MIMIR_params->hit_on_prefetch_sequential = 0;
    MIMIR_params->num_of_prefetch_mimir = 0;
    MIMIR_params->num_of_prefetch_sequential = 0;
    MIMIR_params->num_of_check = 0;
    
    
    if (strcmp(init_params->cache_type, "LRU") == 0)
        MIMIR_params->cache = LRU_init(size, data_type, NULL);
    else if (strcmp(init_params->cache_type, "FIFO") == 0)
        MIMIR_params->cache = fifo_init(size, data_type, NULL);
    else if (strcmp(init_params->cache_type, "AMP") == 0){
        struct AMP_init_params *AMP_init_params = g_new0(struct AMP_init_params, 1);
        AMP_init_params->APT = 4;
        AMP_init_params->p_threshold = 256;
        AMP_init_params->read_size = 1;
        MIMIR_params->cache = AMP_init(size, data_type, AMP_init_params);
    }
    else if (strcmp(init_params->cache_type, "Optimal") == 0){
        struct optimal_init_params *optimal_init_params = g_new(struct optimal_init_params, 1);
        optimal_init_params->reader = NULL;
        MIMIR_params->cache = NULL;
        ;
    }
    else{
        fprintf(stderr, "can't recognize cache type: %s\n", init_params->cache_type);
        MIMIR_params->cache = LRU_init(size, data_type, NULL);
    }

    

    if (data_type == 'l'){
        MIMIR_params->hashtable_for_training = g_hash_table_new_full(g_int64_hash, g_int_equal,
                                                                     simple_g_key_value_destroyer,
                                                                     training_node_destroyer);
        MIMIR_params->hashset_frequentItem = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                                   simple_g_key_value_destroyer,
                                                                   NULL);
        MIMIR_params->prefetch_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, NULL, NULL);
//                                                                 simple_g_key_value_destroyer, NULL);
//                                                                 prefetch_node_destroyer);
        
        if (MIMIR_params->output_statistics){
            MIMIR_params->prefetched_hashtable_mimir = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                                             simple_g_key_value_destroyer,
                                                                             NULL);
            MIMIR_params->prefetched_hashtable_sequential = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                                             simple_g_key_value_destroyer,
                                                                             NULL);
        
        }
    }
    else if (data_type == 'c'){
        MIMIR_params->hashtable_for_training = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                     simple_g_key_value_destroyer,
                                                                     training_node_destroyer);
        MIMIR_params->hashset_frequentItem = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                   simple_g_key_value_destroyer,
                                                                   NULL);
        MIMIR_params->prefetch_hashtable = g_hash_table_new_full(g_str_hash, g_str_equal, NULL, NULL);
//                                                                 simple_g_key_value_destroyer, NULL);
//                                                                 prefetch_node_destroyer);
        if (MIMIR_params->output_statistics){
            MIMIR_params->prefetched_hashtable_mimir = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                             simple_g_key_value_destroyer,
                                                                             NULL);
            MIMIR_params->prefetched_hashtable_sequential = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                                  simple_g_key_value_destroyer,
                                                                                  NULL);
        }
    }
    else{
        g_error("does not support given data type: %c\n", data_type);
    }
    
    return cache;
}


void __MIMIR_mining(struct_cache* MIMIR){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
#ifdef PROFILING
    GTimer *timer = g_timer_new();
    gulong microsecond;
    g_timer_start(timer);
#endif 
    
    __MIMIR_aging(MIMIR);
#ifdef PROFILING
    printf("ts: %lu, aging takes %lf seconds\n", MIMIR_params->ts, g_timer_elapsed(timer, &microsecond));
#endif
    GList* list_node1 = MIMIR_params->training_data;
    GList* list_node2;
    struct training_data_node* data_node1, *data_node2;
    while (list_node1){
        data_node1 = (struct training_data_node*)(list_node1->data);
        if (data_node1->length < MIMIR_params->min_support){
            list_node1 = list_node1->next;
            continue;
        }
        list_node2 = list_node1->next;
        gboolean first_flag = TRUE;
        while (list_node2){
            data_node2 = (struct training_data_node*)(list_node2->data);
            if (data_node1->array[0] - data_node2->array[0] > MIMIR_params->item_set_size)
                break;
            /** We can have better approximation for association rule mining here, maybe using bitwise op, 
             *  and capture the situation where sequence 1 and 2 are of different length, but 
             *  1->2 is 100% confident, 2->1 is not.
             **/
            gint8 smaller_length = data_node1->length;
            if (data_node2->length < smaller_length)
                smaller_length = data_node2->length;
            if (data_node2->length < MIMIR_params->min_support ||
                        abs(data_node2->length - data_node1->length) > MIMIR_params->confidence){
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
                    if ( labs(data_node2->array[i] - data_node1->array[i]) > MIMIR_params->item_set_size){
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
                    mimir_add_to_prefetch_table(MIMIR, data_node2->item_p, data_node1->item_p);
//                    mimir_add_to_prefetch_table(MIMIR, data_node1->item_p, data_node2->item_p);
                }
                
            }       // checking associated
            
            list_node2 = list_node2->next;
            
        }           // while list_node2
        GList* old_node = list_node1;
        list_node1 = list_node1->next;
        /* now delete node1 because len >= min_support */
        MIMIR_params->training_data = g_list_remove_link(MIMIR_params->training_data, old_node);
        struct training_data_node* t_data_node = old_node->data;
        g_hash_table_remove(MIMIR_params->hashtable_for_training, t_data_node->item_p);
    }               // while list_node1

#ifdef PROFILING
    printf("ts: %lu, training takes %lf seconds\n", MIMIR_params->ts, g_timer_elapsed(timer, &microsecond));
#endif
    
//    if (g_hash_table_size(MIMIR_params->hashtable_for_training) > MIMIR->core->size * MIMIR_params->block_size
//        * 0.02 / (MIMIR_params->min_support * 2 + 8 + 4) ){
//        MIMIR_params->ts = 0;
//        MIMIR_params->ts = 0;
//        guint len = g_hash_table_size(MIMIR_params->hashtable_for_training);
//        int i;
//        GList *next_list_node, *list_node = MIMIR_params->training_data;
//        // jump over newsest 10 percent
//        for (i=0; i<len / 2; i++){
//            list_node = list_node->next;
//        }
//        while (list_node){
//            next_list_node = list_node->next;
//            MIMIR_params->training_data = g_list_remove_link(MIMIR_params->training_data, list_node);
//            struct training_data_node* t_data_node = list_node->data;
//            g_hash_table_remove(MIMIR_params->hashtable_for_training, t_data_node->item_p);
//            list_node = next_list_node;
//        }
//    }
    
    
    // maximum 2% mining meta data
    if (g_hash_table_size(MIMIR_params->hashtable_for_training) > MIMIR->core->size * MIMIR_params->block_size
            * 0.02 / (MIMIR_params->min_support * 2 + 8 + 4) ){
        MIMIR_params->ts = 0;
        g_hash_table_remove_all(MIMIR_params->hashtable_for_training);
        MIMIR_params->training_data = NULL;
    }

    
    /* clearing training hashtable and training list */
//    GHashTable *old_hashtable = MIMIR_params->hashtable_for_training;
//    
//    // add the last record into new record table
//    if (MIMIR->core->data_type == 'l'){
//        MIMIR_params->hashtable_for_training = g_hash_table_new_full(g_int64_hash, g_int64_equal,
//                                                                     simple_g_key_value_destroyer,
//                                                                     training_node_destroyer);
//        struct training_data_node *last_data_node = (struct training_data_node*) (MIMIR_params->training_data->data);
//        struct training_data_node *data_node = g_new(struct training_data_node, 1);
//        data_node->array = g_new(gint32, MIMIR_params->max_support);
//        data_node->length = 1;
//        
//        gpointer item_p;
//        item_p = (gpointer)g_new(guint64, 1);
//        *(guint64*)item_p = *(guint64*)(last_data_node->item_p);
//        data_node->item_p = item_p;
//        
//        
//        (data_node->array)[0] = last_data_node->array[last_data_node->length-1];
//        
//        MIMIR_params->training_data = NULL;
//        MIMIR_params->training_data = g_list_prepend(MIMIR_params->training_data, (gpointer)data_node);
//        
//        
//        gpointer key;
//        key = (gpointer)g_new(guint64, 1);
//        *(guint64*)key = *(guint64*)(last_data_node->item_p);
//        g_hash_table_insert(MIMIR_params->hashtable_for_training, key, MIMIR_params->training_data);
//
//    
//    }
//    else{
//        MIMIR_params->hashtable_for_training = g_hash_table_new_full(g_str_hash, g_str_equal,
//                                                                     simple_g_key_value_destroyer,
//                                                                     training_node_destroyer);
//        struct training_data_node *last_data_node = (struct training_data_node*) (MIMIR_params->training_data->data);
//        struct training_data_node *data_node = g_new(struct training_data_node, 1);
//        data_node->array = g_new(gint32, MIMIR_params->max_support);
//        data_node->length = 1;
//        
//        gpointer item_p;
//        item_p = (gpointer)g_strdup((gchar*)(last_data_node->item_p));
//        data_node->item_p = item_p;
//        
//        
//        (data_node->array)[0] = last_data_node->array[last_data_node->length-1];
//        
//        MIMIR_params->training_data = NULL;
//        MIMIR_params->training_data = g_list_prepend(MIMIR_params->training_data, (gpointer)data_node);
//        
//        gpointer key;
//        key = (gpointer)g_strdup((gchar*)(last_data_node->item_p));
//        g_hash_table_insert(MIMIR_params->hashtable_for_training, key, MIMIR_params->training_data);
//
//        
//    }
//    g_hash_table_destroy(old_hashtable);                                // do not need to free the list again
//        printf("\ndone clearing table\n");
//    }
   
#ifdef PROFILING
    printf("ts: %lu, clearing training data takes %lf seconds\n", MIMIR_params->ts, g_timer_elapsed(timer, &microsecond));
    g_timer_stop(timer);
    g_timer_destroy(timer);
#endif
}



void mimir_add_to_prefetch_table(struct_cache* MIMIR, gpointer gp1, gpointer gp2){
    /** currently prefetch table can only support up to 2^31 entries, and this function assumes the platform is 64 bit */
    
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    
    gint prefetch_table_index = GPOINTER_TO_INT(g_hash_table_lookup(MIMIR_params->prefetch_hashtable, gp1));
    gint dim1 = floor(prefetch_table_index/PREFETCH_TABLE_SHARD_SIZE);
    gint dim2 = prefetch_table_index % PREFETCH_TABLE_SHARD_SIZE * (MIMIR_params->prefetch_list_size+1);
    
    
    // insert into prefetch hashtable
    int i;
    if (prefetch_table_index){
        gboolean insert = TRUE;
        if (MIMIR->core->data_type == 'l'){
            for (i=1; i<MIMIR_params->prefetch_list_size+1; i++){
                // if this element is already in the array, then don't need prefetch again
//                if (*(guint64*)(g_ptr_array_index(pArray, i)) == *(guint64*)(gp2)){
                // ATTENTION: the following assumes a 64 bit platform
                if (MIMIR_params->prefetch_table_array[dim1][dim2] != *(gint64*)(gp1)){
                    printf("ERROR prefetch table pos wrong %ld %ld, dim %d %d\n", *(gint64*)gp1, MIMIR_params->prefetch_table_array[dim1][dim2], dim1, dim2);
                    exit(1);
                }
                if ((MIMIR_params->prefetch_table_array[dim1][dim2+i]) == 0)
                    break;
                if ((MIMIR_params->prefetch_table_array[dim1][dim2+i]) == *(gint64*)(gp2))
                    /* update score here, not implemented yet */
                    insert = FALSE;
            }
        }
        else{
            for (i=1; i<MIMIR_params->prefetch_list_size+1; i++){
                // if this element is already in the cache, then don't need prefetch again
                if ( strcmp((char*)(MIMIR_params->prefetch_table_array[dim1][dim2]), gp1) != 0){
                    printf("ERROR prefetch table pos wrong\n");
                    exit(1);
                }
                if ((MIMIR_params->prefetch_table_array[dim1][dim2+i]) == 0)
                    break;
                if ( strcmp((gchar*)(MIMIR_params->prefetch_table_array[dim1][dim2+1]), (gchar*)gp2) == 0 )
                    /* update score here, not implemented yet */
                    insert = FALSE;
            }
        }
        
        if (insert){
            if (i == MIMIR_params->prefetch_list_size+1){
                // list full, randomly pick one for replacement
                i = rand()%MIMIR_params->prefetch_list_size + 1;
                if (MIMIR->core->data_type == 'c'){
                    g_free((gchar*)(MIMIR_params->prefetch_table_array[dim1][dim2+i]));
                }
            }
            // new add at position i
            if (MIMIR->core->data_type == 'c'){
                char* key2 = g_strdup((gchar*)gp2);
                MIMIR_params->prefetch_table_array[dim1][dim2+i] = (gint64)key2;
            }
            else{
                MIMIR_params->prefetch_table_array[dim1][dim2+i] = *(gint64*)(gp2);
            }
        }
    }
    else{
        MIMIR_params->current_prefetch_table_pointer ++;
        dim1 = floor(MIMIR_params->current_prefetch_table_pointer/PREFETCH_TABLE_SHARD_SIZE);
        dim2 = MIMIR_params->current_prefetch_table_pointer % PREFETCH_TABLE_SHARD_SIZE *
                                (MIMIR_params->prefetch_list_size + 1);
        
        /* check whether prefetch table is fully allocated, if True, we are going to replace the 
            entry at current_prefetch_table_pointer by set the entry it points to as 0, 
            delete from prefetch_hashtable and add new entry */
        if (MIMIR_params->prefetch_table_fully_allocatd){
            g_hash_table_remove(MIMIR_params->prefetch_hashtable, &(MIMIR_params->prefetch_table_array[dim1][dim2]));
            memset(&(MIMIR_params->prefetch_table_array[dim1][dim2]), 0,
                   sizeof(gint64) * (MIMIR_params->prefetch_list_size+1));
        }
        
        gpointer gp1_dup;
        if (MIMIR->core->data_type == 'l'){
//            gp1_dup = (gpointer)g_new(guint64, 1);
//            *(guint64*)gp1_dup = *(guint64*)(gp1);
            gp1_dup = &(MIMIR_params->prefetch_table_array[dim1][dim2]);
        }
        else{
            gp1_dup = (gpointer)g_strdup((gchar*)(gp1));
        }
        
        if (MIMIR->core->data_type == 'c'){
            char* key2 = g_strdup((gchar*)gp2);
            MIMIR_params->prefetch_table_array[dim1][dim2+1] = (gint64)key2;
            MIMIR_params->prefetch_table_array[dim1][dim2]   = (gint64)gp1_dup;
        }
        else{
            MIMIR_params->prefetch_table_array[dim1][dim2+1] = *(gint64*)(gp2);
            MIMIR_params->prefetch_table_array[dim1][dim2]   = *(gint64*)(gp1);
        }


        if (g_hash_table_contains(MIMIR_params->prefetch_hashtable, gp1)){
            gpointer gp = g_hash_table_lookup(MIMIR_params->prefetch_hashtable, gp1);
            printf("ERROR contains %ld, value %d, %d\n", *(gint64*)gp1, GPOINTER_TO_INT(gp), prefetch_table_index);
            
        }
        g_hash_table_insert(MIMIR_params->prefetch_hashtable, gp1_dup,
                            GINT_TO_POINTER(MIMIR_params->current_prefetch_table_pointer));
        
        
        if ( (MIMIR_params->current_prefetch_table_pointer +1) % PREFETCH_TABLE_SHARD_SIZE == 0){
            /* need to allocate a new shard for prefetch table */
            if (MIMIR_params->current_metadata_size + PREFETCH_TABLE_SHARD_SIZE * (MIMIR_params->prefetch_list_size * 8 +8 + 4)
                            < MIMIR_params->max_metadata_size){
                MIMIR_params->prefetch_table_array[dim1+1] = g_new0(gint64, PREFETCH_TABLE_SHARD_SIZE *
                                                                    (MIMIR_params->prefetch_list_size + 1));
                gint required_meta_data_size = PREFETCH_TABLE_SHARD_SIZE * (MIMIR_params->prefetch_list_size * 8 + 8 + 4);
                MIMIR_params->current_metadata_size += required_meta_data_size;
                MIMIR_params->cache->core->size = MIMIR->core->size -
                                                    (gint)((MIMIR_params->current_metadata_size + MIMIR_params-> residual_mining_table_size) /MIMIR_params->block_size);
            }
            else{
                MIMIR_params->prefetch_table_fully_allocatd = TRUE;
                MIMIR_params->current_prefetch_table_pointer = 0;
            }
        }
    }
}





void __MIMIR_aging(struct_cache* MIMIR){
    
    ;
}


void training_node_destroyer(gpointer data){
    GList* list_node = (GList*)data;
    struct training_data_node *data_node = (struct training_data_node*) (list_node->data);
    g_free(data_node->array);
    g_free(data_node->item_p);
    g_free(data_node);
    g_list_free_1(list_node);
}

void prefetch_node_destroyer(gpointer data){
//    struct prefetch_hashtable_valuelist_element *node = (struct prefetch_hashtable_valuelist_element*) data;
//    g_free(node->item_p);
//    g_free(node);
    g_ptr_array_free((GPtrArray*)data, TRUE);
//    g_free(data);
}

void prefetch_array_node_destroyer(gpointer data){
    g_free(data);
}
