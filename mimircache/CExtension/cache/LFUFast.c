//
//  LFU_fast.c
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

/* this module uses linkedlist to order cache lines,
 * which is O(1) at each request, which should be preferred in most cases
 * the drawback of this implementation is the memory usage, because two pointers 
 * are associated with each item 
 *
 * this LFU_fast clear the frequenct of an item after evicting from cache 
 * this LFU_fast is LFU_LRU, which choose LRU end when more than one items have 
 * the same smallest freq 
 */ 


#include "LFUFast.h"

#ifdef __cplusplus
extern "C"
{
#endif



/********************************** LFU_fast ************************************/
gboolean __LFU_fast_verify(struct_cache* LFU_fast){

    LFU_fast_params_t* LFU_fast_params = (LFU_fast_params_t*)(LFU_fast->cache_params);
    GList *mnode_list = g_queue_peek_head_link(LFU_fast_params->main_list);
    main_list_node_data_t *mnode_data;
//    branch_list_node_data_t *bnode_data;
    guint64 current_size = 0;
    while (mnode_list){
        mnode_data = mnode_list->data;
        current_size += g_queue_get_length(mnode_data->queue);
        printf("%u\t", g_queue_get_length(mnode_data->queue));
        mnode_list = mnode_list->next;
    }
    printf("\n");
    if (g_hash_table_size(LFU_fast_params->hashtable) == current_size)
        return TRUE;
    else{
        ERROR("hashtable size %u, queue accu size %lu\n",
              g_hash_table_size(LFU_fast_params->hashtable), current_size);
        return FALSE;
    }
}





void __LFU_fast_insert_element(struct_cache* LFU_fast, cache_line* cp){
    LFU_fast_params_t* LFU_fast_params = (LFU_fast_params_t*)(LFU_fast->cache_params);
    
    gpointer key;
    if (cp->type == 'l'){
        key = (gpointer)g_new(guint64, 1);
        *(guint64*)key = *(guint64*)(cp->item_p);
    }
    else{
        key = (gpointer)g_strdup((gchar*)(cp->item_p));
    }
    branch_list_node_data_t* bnode_data = g_new0(branch_list_node_data_t, 1);
    bnode_data->key = key;
    GList *list_node = g_list_append(NULL, bnode_data);
    
    g_hash_table_insert (LFU_fast_params->hashtable,
                         (gpointer)key, (gpointer)list_node);

    
    main_list_node_data_t *mnode_data;
    if (LFU_fast_params->min_freq != 1){
        // initial
        if (LFU_fast_params->min_freq < 1 &&
                LFU_fast_params->main_list->length != 0){
            WARNING("LFU initialization error\n");
        }
        mnode_data = g_new0(main_list_node_data_t, 1);
        mnode_data->freq = 1;
        mnode_data->queue = g_queue_new();
        g_queue_push_head(LFU_fast_params->main_list, mnode_data);
        LFU_fast_params->min_freq = 1;
    }
    else {
        mnode_data = (main_list_node_data_t*)(g_queue_peek_head(LFU_fast_params->main_list));
#ifdef SANITY_CHECK
        if (mnode_data->freq != 1){
            ERROR("first main node freq is not 1, is %d\n", mnode_data->freq);
            exit(1);
        }
#endif
    }
    
    g_queue_push_tail_link(mnode_data->queue, list_node);
    bnode_data->main_list_node = g_queue_peek_head_link(LFU_fast_params->main_list);
}


gboolean LFU_fast_check_element(struct_cache* cache, cache_line* cp){
    return g_hash_table_contains(
                ((LFU_fast_params_t*)(cache->cache_params))->hashtable,
                (gconstpointer)(cp->item_p));
}


void __LFU_fast_update_element(struct_cache* cache, cache_line* cp){
    /* find the given bnode_data, remove from current list, 
     * insert into next main node */ 
    LFU_fast_params_t* LFU_fast_params = (LFU_fast_params_t*)(cache->cache_params);
    
    GList* list_node = g_hash_table_lookup(LFU_fast_params->hashtable,
                                            (gconstpointer)(cp->item_p));
    
    branch_list_node_data_t* bnode_data = list_node->data;
    main_list_node_data_t* mnode_data = bnode_data->main_list_node->data;
    
    // remove from current main_list_node
    g_queue_unlink(mnode_data->queue, list_node);
    
    // check whether there is next
    gboolean exist_next = (bnode_data->main_list_node->next==NULL)?FALSE:TRUE;
    
    // check whether next is freq+1
    main_list_node_data_t* mnode_next_data =
                exist_next?bnode_data->main_list_node->next->data:NULL;
    if (exist_next && mnode_next_data->freq == mnode_data->freq + 1){
        // insert to this main list node
        g_queue_push_tail_link(mnode_next_data->queue, list_node);
    }
    else{
#ifdef SANITY_CHECK
        if (exist_next && mnode_next_data->freq <= mnode_data->freq){
            ERROR("mnode next freq %d, current freq %d\n",
                  mnode_next_data->freq, mnode_data->freq);
            exit(-1);
        }
#endif
        // create a new main list node, insert in between
        main_list_node_data_t *new_mnode_data = g_new0(main_list_node_data_t, 1);
        new_mnode_data->freq = mnode_data->freq + 1;
        new_mnode_data->queue = g_queue_new();
        g_queue_push_tail_link(new_mnode_data->queue, list_node);
        // insert mnode
        g_queue_insert_after(LFU_fast_params->main_list,
                                bnode_data->main_list_node,
                                new_mnode_data);
    }
    bnode_data->main_list_node = bnode_data->main_list_node->next;
}



void __LFU_fast_evict_element(struct_cache* cache, cache_line* cp){
    LFU_fast_params_t* LFU_fast_params = (LFU_fast_params_t*)(cache->cache_params);
    
    GList* mnode = g_queue_peek_head_link(LFU_fast_params->main_list);
    main_list_node_data_t* mnode_data = g_queue_peek_head(LFU_fast_params->main_list);
    
    // find the first main list node that has an non-empty queue
    while (g_queue_is_empty(mnode_data->queue)){
        mnode = mnode->next;
        mnode_data = mnode->data;
    }
    branch_list_node_data_t* bnode_data = g_queue_pop_head(mnode_data->queue);
    g_hash_table_remove(LFU_fast_params->hashtable, (gconstpointer)(bnode_data->key));
    
    g_free(bnode_data);
}


gpointer __LFU_fast__evict_with_return(struct_cache* cache, cache_line* cp){
    LFU_fast_params_t* LFU_fast_params = (LFU_fast_params_t*)(cache->cache_params);
    
    GList* mnode = g_queue_peek_head_link(LFU_fast_params->main_list);
    main_list_node_data_t* mnode_data = g_queue_peek_head(LFU_fast_params->main_list);
    
    // find the first main list node that has an non-empty queue
    while (g_queue_is_empty(mnode_data->queue)){
        mnode = mnode->next;
        mnode_data = mnode->data;
    }
    
    branch_list_node_data_t* bnode_data = g_queue_pop_head(mnode_data->queue);

    // save evicted key
    gpointer evicted_key;
    if (cp->type == 'l'){
        evicted_key = (gpointer)g_new(guint64, 1);
        *(guint64*)evicted_key = *(guint64*)(bnode_data->key);
    }
    else{
        evicted_key = (gpointer)g_strdup((gchar*)bnode_data->key);
    }
    g_hash_table_remove(LFU_fast_params->hashtable, (gconstpointer)(bnode_data->key));
    return evicted_key;
}



gboolean LFU_fast_add_element(struct_cache* cache, cache_line* cp){
    LFU_fast_params_t* LFU_fast_params = (LFU_fast_params_t*)(cache->cache_params);

    if (LFU_fast_check_element(cache, cp)){
        __LFU_fast_update_element(cache, cp);
        return TRUE;
    }
    else{
        __LFU_fast_insert_element(cache, cp);
        if ( (long)g_hash_table_size(LFU_fast_params->hashtable) > cache->core->size)
            __LFU_fast_evict_element(cache, cp);
        return FALSE;
    }
}


gboolean LFU_fast_add_element_only(struct_cache* cache, cache_line* cp){
    return LFU_fast_add_element(cache, cp);
}


gboolean LFU_fast_add_element_withsize(struct_cache* cache, cache_line* cp){
    int i, n = 0;
    gint64 original_lbn = *(gint64*)(cp->item_p);
    gboolean ret_val;
    
    if (cache->core->block_unit_size != 0 && cp->disk_sector_size != 0){
        *(gint64*)(cp->item_p) = (gint64) (*(gint64*)(cp->item_p) *
                                           cp->disk_sector_size /
                                           cache->core->block_unit_size);
        n = (int)ceil((double) cp->size/cache->core->block_unit_size);
    }
    
    ret_val = LFU_fast_add_element(cache, cp);
    
    
    if (cache->core->block_unit_size != 0 && cp->disk_sector_size != 0){
        if (cache->core->block_unit_size != 0){
            for (i=0; i<n-1; i++){
                (*(guint64*)(cp->item_p)) ++;
                LFU_fast_add_element_only(cache, cp);
            }
        }
    }
    
    *(gint64*)(cp->item_p) = original_lbn;
    return ret_val;
}




void free_main_list_node_data(gpointer data){
    main_list_node_data_t* mnode_data = data;
    g_queue_free_full(mnode_data->queue, simple_g_key_value_destroyer);
    g_free(data);
}




void LFU_fast_destroy(struct_cache* cache){
    LFU_fast_params_t* LFU_fast_params = (LFU_fast_params_t*)(cache->cache_params);
    
    g_queue_free_full(LFU_fast_params->main_list, free_main_list_node_data);
    g_hash_table_destroy(LFU_fast_params->hashtable);
    
    cache_destroy(cache);
}


void LFU_fast_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in LFU_fast, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    LFU_fast_destroy(cache);
}



struct_cache* LFU_fast_init(guint64 size, char data_type, int block_size, void* params){
    struct_cache* cache = cache_init(size, data_type, block_size);
    LFU_fast_params_t* LFU_fast_params = g_new0(LFU_fast_params_t, 1);
    cache->cache_params = (void*) LFU_fast_params;
    
    cache->core->type               =   e_LFU_fast;
    cache->core->cache_init         =   LFU_fast_init;
    cache->core->destroy            =   LFU_fast_destroy;
    cache->core->destroy_unique     =   LFU_fast_destroy_unique;
    cache->core->add_element        =   LFU_fast_add_element;
    cache->core->check_element      =   LFU_fast_check_element;

    cache->core->__insert_element   =   __LFU_fast_insert_element;
    cache->core->__update_element   =   __LFU_fast_update_element;
    cache->core->__evict_element    =   __LFU_fast_evict_element;
    cache->core->__evict_with_return=   __LFU_fast__evict_with_return;
    cache->core->get_size           =   LFU_fast_get_size;
    cache->core->cache_init_params  =   NULL;
    cache->core->add_element_only   =   LFU_fast_add_element;
    cache->core->add_element_withsize = LFU_fast_add_element_withsize; 
    

    if (data_type == 'l'){
        LFU_fast_params->hashtable =
            g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                  simple_g_key_value_destroyer,
                                  NULL);
    }
    
    else if (data_type == 'c'){
        LFU_fast_params->hashtable =
            g_hash_table_new_full(g_str_hash, g_str_equal,
                                  simple_g_key_value_destroyer,
                                  NULL);
    }
    else{
        ERROR("does not support given data type: %c\n", data_type);
    }
    
    LFU_fast_params->min_freq = 0; 
    LFU_fast_params->main_list = g_queue_new();
    
    return cache;
}


gint64 LFU_fast_get_size(struct_cache* cache){
    LFU_fast_params_t* LFU_fast_params = (LFU_fast_params_t*)(cache->cache_params);
    return (guint64) g_hash_table_size(LFU_fast_params->hashtable);
}

void LFU_fast_remove_element(struct_cache* cache, void* data_to_remove){
    LFU_fast_params_t* LFU_fast_params = (LFU_fast_params_t*)(cache->cache_params);
    GList* blist_node = g_hash_table_lookup(LFU_fast_params->hashtable, data_to_remove);
    
    branch_list_node_data_t* bnode_data = blist_node->data;
    main_list_node_data_t* mnode_data = bnode_data->main_list_node->data;
    g_queue_unlink(mnode_data->queue, blist_node); 
    
    g_hash_table_remove(LFU_fast_params->hashtable, (gconstpointer)(data_to_remove));
}



#ifdef __cplusplus
}
#endif