//
//  LRUPage.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h" 
#include "LRUPage.h"

#ifdef __cplusplus
extern "C"
{
#endif



void __LRUPage_insert_element(struct_cache* LRUPage, cache_line* cp){
    struct LRUPage_params* LRUPage_params = (struct LRUPage_params*)(LRUPage->cache_params);
    
    page_t *page    =   g_new0(page_t, 1);
    page->content   =   g_new(char, cp->size_of_content);
    memcpy(page->content, cp->content, cp->size_of_content);

    if (cp->type == 'l'){
        page->key = (gpointer)g_new(guint64, 1);
        *(guint64*)(page->key) = *(guint64*)(cp->item_p);
    }
    else{
        page->key = (gpointer)g_strdup((gchar*)(cp->item_p));
    }
    
    GList* node = g_list_alloc();
    node->data = page;
    
    g_queue_push_tail_link(LRUPage_params->list, node);
    g_hash_table_insert(LRUPage_params->hashtable, page->key, (gpointer)node);
    
}

gboolean LRUPage_check_element(struct_cache* cache, cache_line* cp){
    struct LRUPage_params* LRUPage_params = (struct LRUPage_params*)(cache->cache_params);
    return g_hash_table_contains( LRUPage_params->hashtable, cp->item_p );
}


void __LRUPage_update_element(struct_cache* cache, cache_line* cp){
    struct LRUPage_params* LRUPage_params = (struct LRUPage_params*)(cache->cache_params);
    GList* node = (GList* ) g_hash_table_lookup(LRUPage_params->hashtable, cp->item_p);
    g_queue_unlink(LRUPage_params->list, node);
    memcpy(((page_t*)(node->data))->content, cp->content, cp->size_of_content);
    g_queue_push_tail_link(LRUPage_params->list, node);
}


void __LRUPage_evict_element(struct_cache* LRUPage, cache_line* cp){
    struct LRUPage_params* LRUPage_params = (struct LRUPage_params*)(LRUPage->cache_params);

    page_t* page = g_queue_pop_head(LRUPage_params->list);
    g_hash_table_remove(LRUPage_params->hashtable, (gconstpointer)(page->key));
    destroy_LRUPage_t(page);
}


void* __LRUPage__evict_with_return(struct_cache* LRUPage, cache_line* cp){
    /** evict one element and return the key of evicted element, 
     needs to free the memory of returned data **/
    
    struct LRUPage_params* LRUPage_params = (struct LRUPage_params*)(LRUPage->cache_params);

    page_t* page = g_queue_pop_head(LRUPage_params->list);
    gpointer evicted_key;
    if (cp->type == 'l'){
        evicted_key = (gpointer)g_new(guint64, 1);
        *(guint64*)evicted_key = *(guint64*)(page->key);
    }
    else{
        evicted_key = (gpointer)g_strdup((gchar*)(page->key));
    }

    g_hash_table_remove(LRUPage_params->hashtable, (gconstpointer)(page->key));
    destroy_LRUPage_t(page);
    return evicted_key;
}




gboolean LRUPage_add_element(struct_cache* cache, cache_line* cp){
    struct LRUPage_params* LRUPage_params = (struct LRUPage_params*)(cache->cache_params);
    if (LRUPage_check_element(cache, cp)){
        __LRUPage_update_element(cache, cp);
        return TRUE;
    }
    else{
        __LRUPage_insert_element(cache, cp);
        if ( (long)g_hash_table_size( LRUPage_params->hashtable) > cache->core->size)
            __LRUPage_evict_element(cache, cp);
        return FALSE;
    }
}




void LRUPage_destroy(struct_cache* cache){
    struct LRUPage_params* LRUPage_params = (struct LRUPage_params*)(cache->cache_params);

    g_queue_free_full(LRUPage_params->list, destroy_LRUPage_t);
    g_hash_table_destroy(LRUPage_params->hashtable);
    cache_destroy(cache);
}

void LRUPage_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    LRUPage_destroy(cache);
}


struct_cache* LRUPage_init(guint64 size, char data_type, int block_size, void* params){
    struct_cache *cache = cache_init(size, data_type, block_size);
    cache->cache_params = g_new0(struct LRUPage_params, 1);
    struct LRUPage_params* LRUPage_params = (struct LRUPage_params*)(cache->cache_params);
    
    cache->core->type               =       e_LRUPage;
    cache->core->cache_init         =       LRUPage_init;
    cache->core->destroy            =       LRUPage_destroy;
    cache->core->destroy_unique     =       LRUPage_destroy_unique;
    cache->core->add_element        =       LRUPage_add_element;
    cache->core->check_element      =       LRUPage_check_element;
    cache->core->__insert_element   =       __LRUPage_insert_element;
    cache->core->__update_element   =       __LRUPage_update_element;
    cache->core->__evict_element    =       __LRUPage_evict_element;
    cache->core->__evict_with_return=       __LRUPage__evict_with_return;
    cache->core->get_size           =       LRUPage_get_size;
    cache->core->cache_init_params  =       NULL;
    cache->core->add_element_only   =       LRUPage_add_element;

    if (data_type == 'l'){
        LRUPage_params->hashtable   =   g_hash_table_new_full(g_int64_hash,
                                                              g_int64_equal,
                                                              NULL,
                                                              NULL);
    }
    else if (data_type == 'c'){
        LRUPage_params->hashtable   =   g_hash_table_new_full(g_str_hash,
                                                              g_str_equal,
                                                              NULL,
                                                              NULL);
    }
    else{
        ERROR("does not support given data type: %c\n", data_type);
    }
    LRUPage_params->list = g_queue_new();
    
    return cache;
}




void LRUPage_remove_element(struct_cache* cache, void* data_to_remove){
    struct LRUPage_params* LRUPage_params = (struct LRUPage_params*)(cache->cache_params);
    
    GList* node = g_hash_table_lookup(LRUPage_params->hashtable, data_to_remove);
    g_queue_delete_link(LRUPage_params->list, node);
    g_hash_table_remove(LRUPage_params->hashtable, data_to_remove);
}

gint64 LRUPage_get_size(struct_cache* cache){
    struct LRUPage_params* LRUPage_params = (struct LRUPage_params*)(cache->cache_params);
    return (guint64) g_hash_table_size(LRUPage_params->hashtable);
}


void destroy_LRUPage_t(gpointer data){
    page_t *page = (page_t*) data; 
    if (page->key != NULL)
        g_free(page->key);
    if (page->content != NULL)
        g_free(page->content);
    g_free(page); 
}



#ifdef __cplusplus
}
#endif