//
//  AMP.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h" 
#include "AMP.h"
#include <assert.h>



/** because we use direct_hash in hashtable, meaning we direct use pointer as int, so be careful
 ** we require the block number to be within 32bit uint 
 **/


void checkHashTable(gpointer key, gpointer value, gpointer user_data); 


/* AMP functions */
struct AMP_page* AMP_lookup(struct_cache* AMP, gint64 block){
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);
    GList* list_node = (GList* ) g_hash_table_lookup(AMP_params->hashtable, &block);
    if (list_node)
        return (struct AMP_page*)(list_node->data);
    else
        return NULL;
}


static struct AMP_page* AMP_prev(struct_cache* AMP, struct AMP_page* page){
    return AMP_lookup(AMP, page->block_number - 1);
}

static struct AMP_page* AMP_prev_int(struct_cache* AMP, gint64 block){
    return AMP_lookup(AMP, block - 1);
}

static struct AMP_page* AMP_last(struct_cache* AMP, struct AMP_page* page){
    return AMP_lookup(AMP, page->last_block_number);
}


static gboolean AMP_isLast(struct AMP_page* page){
    return (page->block_number == page->last_block_number); 
}


void createPages(struct_cache* AMP, gint64 block_begin, gint length){
    /* this function currently is used for prefetching */
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);
//    g_hash_table_foreach(AMP_params->hashtable, checkHashTable, NULL);
    if (length <= 0)
        fprintf(stderr, "prefetch length %d\n", length);
//    printf("prefetch %ld, to %ld\n", block_begin, block_begin+length-1);
    gint64 i;
    gint64 lastblock = block_begin + length -1;
    struct AMP_page* page_new;
    for (i=block_begin; i<block_begin+length; i+=1){
        if (AMP_check_element(AMP, i))
            page_new = __AMP_update_element(AMP, i);
        else
            page_new = __AMP_insert_element(AMP, i);
        page_new->last_block_number = lastblock;
        page_new->accessed = 0;
        page_new->old = 0;
        g_hash_table_add(AMP_params->prefetched, &(page_new->block_number));
    }
//    lastblock = block_begin + length -1;
//    if (lastblock < 0)
//        printf("last block %ld, block begin %ld, length %d\n", lastblock, block_begin, length);
//    if (i!=lastblock+1)
//        printf("i %ld, last block %ld\n", i, lastblock);
//    if (!g_hash_table_contains(AMP_params->hashtable, &lastblock))
//        printf("after insert, last block disappear\n");

    struct AMP_page* last_page = AMP_lookup(AMP, lastblock);
    struct AMP_page* prev_page = AMP_lookup(AMP, block_begin - 1);
    if (last_page == NULL || prev_page == NULL)
        printf("ERROR got NULL for page %p %p\n", prev_page, last_page);
    

    last_page->p = MAX(prev_page->p, last_page->g +1);
    last_page->g = prev_page->g;
    AMP_lookup(AMP, last_page->block_number-prev_page->g)->tag = TRUE; 
    
    
    AMP_params->num_of_prefetch += length;

    while ( (long)g_hash_table_size( AMP_params->hashtable) > AMP->core->size)
        __AMP_evict_element(AMP);
}

struct AMP_page* lastInSequence(struct_cache* AMP, struct AMP_page *page){
    struct AMP_page* l1 = AMP_last(AMP, page);
    if (!l1)
        return 0;
    if (!AMP_lookup(AMP, page->last_block_number+1))
        return l1;
    struct AMP_page* l2 = AMP_lookup(AMP, page->last_block_number + l1->p);
    if (l2)
        return l2;
    else
        return l1;
}





/* end of AMP functions */


void printer(gpointer data){
    printf("in printer %ld\n", *(guint64*)data);
}

void checkHashTable(gpointer key, gpointer value, gpointer user_data){
    GList *node = (GList*) value;
    struct AMP_page* page = node->data;
    if (page->block_number != *(gint64*)key || page->block_number < 0)
        printf("find error in page, page block %ld, key %ld\n", page->block_number, *(gint64*)key);
}



struct AMP_page* __AMP_insert_element(struct_cache* AMP, gint64 block){
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);
    
    if (g_hash_table_contains(AMP_params->hashtable, &block))
        fprintf(stderr, "ERROR element adding is in the cache\n");
    
    
    struct AMP_page* page = g_new0(struct AMP_page, 1);
    page->block_number = block;
    
    
    GList* node = g_list_alloc();
    node->data = page;
    
    g_queue_push_tail_link(AMP_params->list, node);
    
    g_hash_table_insert(AMP_params->hashtable, (gpointer)&(page->block_number), (gpointer)node);

    
    return page;
}

gboolean AMP_check_element(struct_cache* cache, gint64 block){
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);
    if (g_hash_table_contains(AMP_params->prefetched, &block)){
        AMP_params->num_of_hit ++;
        g_hash_table_remove(AMP_params->prefetched, &block);
    }
    return g_hash_table_contains( AMP_params->hashtable, &block );
}


struct AMP_page* __AMP_update_element(struct_cache* cache, gint64 block){
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);
    GList* node = g_hash_table_lookup(AMP_params->hashtable, &block);
    
    g_queue_unlink(AMP_params->list, node);
    g_queue_push_tail_link(AMP_params->list, node);
    return (struct AMP_page*) (node->data);
}


void __AMP_evict_element(struct_cache* AMP){
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);

    struct AMP_page *page = (struct AMP_page*) g_queue_pop_head(AMP_params->list);
//    if (page->block_number == 34104){
//        printf("******************I am here*****************\n");
//        printf("old %d accessed %d\n", page->old, page->accessed);
//        exit(1);
//    }
    if (page->old || page->accessed){
        gboolean result = g_hash_table_remove(AMP_params->hashtable, (gconstpointer)&(page->block_number));
        if (result == FALSE){
            printf("ERROR nothing removed\n");
            exit(1);
        }
        g_hash_table_remove(AMP_params->prefetched, &(page->block_number));
        page->block_number = -1;
        g_free(page);
    }
    else{
        page->old = TRUE;
        page->tag = FALSE;
        GList* node = g_list_alloc();
        node->data = page;
        g_queue_push_tail_link(AMP_params->list, node); 
        g_hash_table_insert(AMP_params->hashtable, (gpointer)&(page->block_number), (gpointer)node);
        
        struct AMP_page* last_sequence_page = lastInSequence(AMP, page);
        if (last_sequence_page){
            last_sequence_page->p --;
            if (last_sequence_page->p < 1)
                last_sequence_page->p = 1;
            last_sequence_page->g = MIN(last_sequence_page->p-1, last_sequence_page->g-1);
            if (last_sequence_page->g < 0)
                last_sequence_page->g = 0;
        }
    }
}



gboolean AMP_add_element(struct_cache* AMP, cache_line* cp){
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);
//    g_hash_table_foreach(AMP_params->hashtable, checkHashTable, NULL);
    
    gint64 block;
    if (cp->type == 'l')
        block = *(gint64*) (cp->item_p);
    else{
        block = atoll(cp->item);
    }
    struct AMP_page* page = AMP_lookup(AMP, block);
//    struct AMP_page* page_t = AMP_lookup(AMP, 25060);
//    if (page_t && page_t->p < 0){
//        printf("page %ld, p %d\n", page_t->block_number, page_t->p);
//        exit(1);
//    }
//    if (page && (page->block_number == 34104 || page->block_number == 34098)){
//        printf("add ******************I am here***************** %ld\n", page->block_number);
//        printf("old %d accessed %d\n", page->old, page->accessed);
//    }
//    if (g_hash_table_size(AMP_params->hashtable) != AMP_params->list->length){
//        printf("ERROR size not same, %ld %u %u\n", AMP->core->size, g_hash_table_size(AMP_params->hashtable), AMP_params->list->length);
//        exit(1);
//    }
    if (AMP_check_element(AMP, block)){
        // sanity check
        if (page == NULL)
            fprintf(stderr, "ERROR page is NULL\n");
        if (page->accessed)
            __AMP_update_element(AMP, block);
        page->accessed = 1;

        gint length = 0;
        if (page->tag){
            // hit in the trigger and prefetch
            struct AMP_page* page_last = AMP_last(AMP, page);
            length =  AMP_params->read_size;
            if (page_last && page_last->p)
                length = page_last->p;
        }
        
        gboolean check_result = AMP_isLast(page) && !(page->old);
        if (check_result){
            struct AMP_page* page_new = lastInSequence(AMP, page);
            if (page_new)
                if (page_new->p + AMP_params->read_size < AMP_params->p_threshold)
                    page_new->p = page_new->p + AMP_params->read_size;
            ;
        }

        /* this is done here, because the page may be evicted when createPages */
        if (page->tag){
//            printf("inside tag, page %p tag block %ld, tag %d last block number %ld, page %p\n", page, page->block_number, page->tag, page->last_block_number, AMP_last(AMP, page));
            page->tag = FALSE;
            createPages(AMP, page->last_block_number+1, length);
        }
        return TRUE;
    }
    else{
        if (page != NULL)
            fprintf(stderr, "ERROR page is not NULL\n");
        page = __AMP_insert_element(AMP, block);
        page->accessed = 1;
        
        struct AMP_page* page_prev = AMP_lookup(AMP, block - 1);
        int length = AMP_params->read_size;
        if (page_prev && page_prev->p)
            length = page_prev->p;
        // miss -> prefetch
//        if (length < 0)
//            printf("in insert, length %d, read_size %d, block %ld, p %d\n", length, AMP_params->read_size, page_prev->block_number, page_prev->p); 
        if (page_prev)
            createPages(AMP, block+1, length);
        
        while ( (long)g_hash_table_size( AMP_params->hashtable) > AMP->core->size)
            __AMP_evict_element(AMP);
        
        return FALSE;
    }
}




void AMP_destroy(struct_cache* cache){
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);

    g_queue_free_full(AMP_params->list, simple_g_key_value_destroyer);
    g_hash_table_destroy(AMP_params->hashtable);
    g_hash_table_destroy(AMP_params->prefetched);
    cache_destroy(cache);
}

void AMP_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);

    g_queue_free(AMP_params->list);
    g_hash_table_destroy(AMP_params->hashtable);
    g_hash_table_destroy(AMP_params->prefetched);
}


struct_cache* AMP_init(guint64 size, char data_type, void* params){
    struct_cache *cache = cache_init(size, data_type);
    cache->cache_params = g_new0(struct AMP_params, 1);
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);
    struct AMP_init_params* init_params = (struct AMP_init_params*) params;
    
    cache->core->type = e_AMP;
    cache->core->cache_init = AMP_init;
    cache->core->destroy = AMP_destroy;
    cache->core->destroy_unique = AMP_destroy_unique;
    cache->core->add_element = AMP_add_element;

    cache->core->get_size = AMP_get_size;
    cache->core->cache_init_params = params;
    
    AMP_params->APT = init_params->APT;
    AMP_params->read_size = init_params->read_size;
    AMP_params->p_threshold = init_params->p_threshold;
    

    AMP_params->hashtable = g_hash_table_new_full(
                                                  g_int64_hash,
                                                  g_int64_equal,
                                                  NULL,
                                                  NULL);
    AMP_params->prefetched = g_hash_table_new_full(
                                                  g_int64_hash,
                                                  g_int64_equal,
                                                  NULL,
                                                  NULL);
    AMP_params->list = g_queue_new();
    
    
    return cache;
}



guint64 AMP_get_size(struct_cache* cache){
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);
    return (guint64) g_hash_table_size(AMP_params->hashtable);
}
