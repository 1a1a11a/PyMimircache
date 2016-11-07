////
////  PG.h
////  mimircache
////
////  Created by Juncheng on 6/2/16.
////  Copyright © 2016 Juncheng. All rights reserved.
////
//
//
//#include "cache.h" 
//#include "PG.h"
//
//
//
//
//struct PG_page* __PG_insert_element(struct_cache* PG, cache_line* cp){
//    struct PG_params* PG_params = (struct PG_params*)(PG->cache_params);
//    
//    struct PG_page* page = g_new0(struct PG_page, 1);
//    page->block_number = block;
//    
//    GList* node = g_list_alloc();
//    node->data = page;
//    
//    g_queue_push_tail_link(PG_params->list, node);
//    g_hash_table_insert(PG_params->hashtable, (gpointer)&(page->block_number), (gpointer)node);
//    return page;
//}
//
//
//void __PG_insert_element(struct_cache* PG, cache_line* cp){
//    gint64 block = GET_BLOCK(cp);
//    __PG_insert_element_int(PG, block); 
//}
//
//
//gboolean PG_check_element_int(struct_cache* cache, gint64 block){
//    struct PG_params* PG_params = (struct PG_params*)(cache->cache_params);
//    return g_hash_table_contains( PG_params->hashtable, &block );
//}
//
//
//gboolean PG_check_element(struct_cache* cache, cache_line* cp){
//    gint64 block = GET_BLOCK(cp);
//    return PG_check_element_int(cache, block);
//}
//
//
//
//struct PG_page* __PG_update_element_int(struct_cache* cache, gint64 block){
//    struct PG_params* PG_params = (struct PG_params*)(cache->cache_params);
//    GList* node = g_hash_table_lookup(PG_params->hashtable, &block);
//    
//    g_queue_unlink(PG_params->list, node);
//    g_queue_push_tail_link(PG_params->list, node);
//    return (struct PG_page*) (node->data);
//}
//
//void __PG_update_element(struct_cache* cache, cache_line* cp){
//    gint64 block = GET_BLOCK(cp);
//    __PG_update_element_int(cache, block);
//}
//
//
//void __PG_evict_element(struct_cache* PG, cache_line* cp){
//    struct PG_params* PG_params = (struct PG_params*)(PG->cache_params);
//
//    struct PG_page *page = (struct PG_page*) g_queue_pop_head(PG_params->list);
//    if (page->old || page->accessed){
//        gboolean result = g_hash_table_remove(PG_params->hashtable, (gconstpointer)&(page->block_number));
//        if (result == FALSE){
//            fprintf(stderr, "ERROR nothing removed, block %ld\n", page->block_number);
//            exit(1);
//        }
//        g_hash_table_remove(PG_params->prefetched, &(page->block_number));
//        page->block_number = -1;
//        g_free(page);
//    }
//    else{
//        page->old = TRUE;
//        page->tag = FALSE;
//        GList* node = g_list_alloc();
//        node->data = page;
//        g_queue_push_tail_link(PG_params->list, node); 
//        g_hash_table_insert(PG_params->hashtable, (gpointer)&(page->block_number), (gpointer)node);
//        
//        struct PG_page* last_sequence_page = lastInSequence(PG, page);
//        if (last_sequence_page){
//            last_sequence_page->p --;
//            if (last_sequence_page->p < 1)
//                last_sequence_page->p = 1;
//            last_sequence_page->g = MIN(last_sequence_page->p-1, last_sequence_page->g-1);
//            if (last_sequence_page->g < 0)
//                last_sequence_page->g = 0;
//        }
//    }
//}
//
//gpointer __PG_evict_element_with_return(struct_cache* PG, cache_line* cp){
//    /** return a pointer points to the data that being evicted, 
//     *  it can be a pointer pointing to gint64 or a pointer pointing to char*
//     *  it is the user's responsbility to g_free the pointer 
//     **/
//     
//    struct PG_params* PG_params = (struct PG_params*)(PG->cache_params);
//    
//    struct PG_page *page = (struct PG_page*) g_queue_pop_head(PG_params->list);
//    if (page->old || page->accessed){
//        gboolean result = g_hash_table_remove(PG_params->hashtable, (gconstpointer)&(page->block_number));
//        if (result == FALSE){
//            fprintf(stderr, "ERROR nothing removed, block %ld\n", page->block_number);
//            exit(1);
//        }
//        gint64* return_data = g_new(gint64, 1);
//        *(gint64*) return_data = page->block_number;
//
//        g_hash_table_remove(PG_params->prefetched, &(page->block_number));
//        page->block_number = -1;
//        g_free(page);
//        return return_data;
//    }
//    else{
//        page->old = TRUE;
//        page->tag = FALSE;
//        GList* node = g_list_alloc();
//        node->data = page;
//        g_queue_push_tail_link(PG_params->list, node);
//        g_hash_table_insert(PG_params->hashtable, (gpointer)&(page->block_number), (gpointer)node);
//        
//        struct PG_page* last_sequence_page = lastInSequence(PG, page);
//        if (last_sequence_page){
//            last_sequence_page->p --;
//            if (last_sequence_page->p < 1)
//                last_sequence_page->p = 1;
//            last_sequence_page->g = MIN(last_sequence_page->p-1, last_sequence_page->g-1);
//            if (last_sequence_page->g < 0)
//                last_sequence_page->g = 0;
//        }
//        return __PG_evict_element_with_return(PG, cp); 
//    }
//}
//
//
//
//
//gboolean PG_add_element_no_eviction(struct_cache* PG, cache_line* cp){
//    struct PG_params* PG_params = (struct PG_params*)(PG->cache_params);
//    gint64 block;
//    if (cp->type == 'l')
//        block = *(gint64*) (cp->item_p);
//    else{
//        block = atoll(cp->item);
//    }
//
//    struct PG_page* page = PG_lookup(PG, block);
//
//    if (PG_check_element_int(PG, block)){
//        // sanity check
//        if (page == NULL)
//            fprintf(stderr, "ERROR page is NULL\n");
//
//        if (g_hash_table_contains(PG_params->prefetched, &block)){
//            PG_params->num_of_hit ++;
//            g_hash_table_remove(PG_params->prefetched, &block);
//        }
//
//        if (page->accessed)
//            __PG_update_element_int(PG, block);
//        page->accessed = 1;
//
//        gint length = 0;
//        if (page->tag){
//            // hit in the trigger and prefetch
//            struct PG_page* page_last = PG_last(PG, page);
//            length =  PG_params->read_size;
//            if (page_last && page_last->p)
//                length = page_last->p;
//        }
//        
//        gboolean check_result = PG_isLast(page) && !(page->old);
//        if (check_result){
//            struct PG_page* page_new = lastInSequence(PG, page);
//            if (page_new)
//                if (page_new->p + PG_params->read_size < PG_params->p_threshold)
//                    page_new->p = page_new->p + PG_params->read_size;
//            ;
//        }
//
//        /* this is done here, because the page may be evicted when createPages */
//        if (page->tag){
//            page->tag = FALSE;
//            createPages_no_eviction(PG, page->last_block_number+1, length);
//        }
//        return TRUE;
//    }
//    else{
//        if (page != NULL)
//            fprintf(stderr, "ERROR page is not NULL\n");
//        page = __PG_insert_element_int(PG, block);
//        page->accessed = 1;
//        
//        struct PG_page* page_prev = PG_lookup(PG, block - 1);
//        int length = PG_params->read_size;
//        if (page_prev && page_prev->p)
//            length = page_prev->p;
//        // miss -> prefetch
//        if (page_prev)
//            createPages_no_eviction(PG, block+1, length);
//        
//        return FALSE;
//    }
//}
//
//gboolean PG_add_element(struct_cache* PG, cache_line* cp){
//    struct PG_params* PG_params = (struct PG_params*)(PG->cache_params);
//    gboolean result = PG_add_element_no_eviction(PG, cp);
//    while ( (long)g_hash_table_size( PG_params->hashtable) > PG->core->size)
//        __PG_evict_element(PG, NULL);
//    return result;
//}
//
//
//
//
//void PG_destroy(struct_cache* cache){
//    struct PG_params* PG_params = (struct PG_params*)(cache->cache_params);
//
//    g_queue_free_full(PG_params->list, simple_g_key_value_destroyer);
//    g_hash_table_destroy(PG_params->hashtable);
//    g_hash_table_destroy(PG_params->prefetched);
//    cache_destroy(cache);
//}
//
//void PG_destroy_unique(struct_cache* cache){
//    /* the difference between destroy_unique and destroy
//     is that the former one only free the resources that are
//     unique to the cache, freeing these resources won't affect
//     other caches copied from original cache
//     in Optimal, next_access should not be freed in destroy_unique,
//     because it is shared between different caches copied from the original one.
//     */
//    struct PG_params* PG_params = (struct PG_params*)(cache->cache_params);
//
//    g_queue_free(PG_params->list);
//    g_hash_table_destroy(PG_params->hashtable);
//    g_hash_table_destroy(PG_params->prefetched);
//}
//
//
//struct_cache* PG_init(guint64 size, char data_type, void* params){
//    struct_cache *cache = cache_init(size, data_type);
//    cache->cache_params = g_new0(struct PG_params, 1);
//    struct PG_params* PG_params = (struct PG_params*)(cache->cache_params);
//    struct PG_init_params* init_params = (struct PG_init_params*) params;
//    
//    cache->core->type = e_PG;
//    cache->core->cache_init = PG_init;
//    cache->core->destroy = PG_destroy;
//    cache->core->destroy_unique = PG_destroy_unique;
//    cache->core->add_element = PG_add_element;
//    cache->core->check_element = PG_check_element;
//    cache->core->__insert_element = __PG_insert_element;
//    cache->core->__update_element = __PG_update_element;
//    cache->core->__evict_element = __PG_evict_element;
//    cache->core->__evict_element_with_return = __PG_evict_element_with_return;
//
//    cache->core->get_size = PG_get_size;
//    cache->core->cache_init_params = params;
//    
//    PG_params->APT = init_params->APT;
//    PG_params->read_size = init_params->read_size;
//    PG_params->p_threshold = init_params->p_threshold;
//    PG_params->temp = 0;
//    
//
//    PG_params->hashtable = g_hash_table_new_full(
//                                                  g_int64_hash,
//                                                  g_int64_equal,
//                                                  NULL,
//                                                  NULL);
//    PG_params->prefetched = g_hash_table_new_full(
//                                                  g_int64_hash,
//                                                  g_int64_equal,
//                                                  NULL,
//                                                  NULL);
//    PG_params->list = g_queue_new();
//    
//    
//    return cache;
//}
//
//
//
//guint64 PG_get_size(struct_cache* cache){
//    struct PG_params* PG_params = (struct PG_params*)(cache->cache_params);
//    return (guint64) g_hash_table_size(PG_params->hashtable);
//}
