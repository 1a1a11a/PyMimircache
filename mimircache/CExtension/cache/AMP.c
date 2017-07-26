//
//  AMP.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "AMP.h"

#ifdef __cplusplus
extern "C"
{
#endif


/** because we use direct_hash in hashtable, meaning we direct use pointer as int, so be careful
 ** we require the block number to be within 32bit uint 
 **/


#define GET_BLOCK( cp )   ( ((cache_line*) (cp))->type == 'l'? *(gint64*) (((cache_line*) (cp))->item_p): atoll(((cache_line*) (cp))->item))

//#define TRACKED_BLOCK 13113277 




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


//static struct AMP_page* AMP_prev(struct_cache* AMP, struct AMP_page* page){
//    return AMP_lookup(AMP, page->block_number - 1);
//}
//
//static struct AMP_page* AMP_prev_int(struct_cache* AMP, gint64 block){
//    return AMP_lookup(AMP, block - 1);
//}

static struct AMP_page* AMP_last(struct_cache* AMP, struct AMP_page* page){
    return AMP_lookup(AMP, page->last_block_number);
}


static gboolean AMP_isLast(struct AMP_page* page){
    return (page->block_number == page->last_block_number); 
}


void createPages_no_eviction(struct_cache* AMP, gint64 block_begin, gint length){
    /* this function currently is used for prefetching */
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);
    if (length <= 0 || block_begin <= 1){
        fprintf(stderr, "error AMP prefetch length %d, begin from %ld\n", length, block_begin);
        abort();
    }
//    printf("prefetch %d pages\n", length);
    
#ifdef TRACKED_BLOCK
    if (block_begin <= TRACKED_BLOCK && TRACKED_BLOCK < block_begin + length)
        printf("prefetch %ld, length %d\n", block_begin, length);
#endif
    
    gint64 i;
    gint64 lastblock = block_begin + length -1;
    struct AMP_page* page_new;
    for (i=block_begin; i<block_begin+length; i+=1){
        if (AMP_check_element_int(AMP, i))
            page_new = __AMP_update_element_int(AMP, i);
        else{
            AMP_params->num_of_prefetch ++;
            page_new = __AMP_insert_element_int(AMP, i);
            g_hash_table_add(AMP_params->prefetched, &(page_new->block_number));
        }
        page_new->last_block_number = lastblock;
        page_new->accessed = 0;
        page_new->old = 0;
    }

    struct AMP_page* last_page = AMP_lookup(AMP, lastblock);
    struct AMP_page* prev_page = AMP_lookup(AMP, block_begin - 1);
    if (last_page == NULL){ // || prev_page == NULL){       // prev page be evicted?
        ERROR("got NULL for page %p %p, block %ld %ld\n", prev_page, last_page,
               block_begin-1, lastblock);
    }
    
    // new 1704
    if (prev_page == NULL){
        last_page->p = length;
        last_page->g = (int) (last_page->p/2);
        AMP_lookup(AMP, last_page->block_number)->tag = TRUE;
    }
    // end
    else{
        last_page->p = MAX(prev_page->p, last_page->g +1);
        last_page->g = prev_page->g;
        AMP_lookup(AMP, last_page->block_number-prev_page->g)->tag = TRUE;
    }
}

void createPages(struct_cache* AMP, gint64 block_begin, gint length){
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);
    createPages_no_eviction(AMP, block_begin, length);
    while ( (long)g_hash_table_size( AMP_params->hashtable) > AMP->core->size)
        __AMP_evict_element(AMP, NULL);
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




void checkHashTable(gpointer key, gpointer value, gpointer user_data){
    GList *node = (GList*) value;
    struct AMP_page* page = node->data;
    if (page->block_number != *(gint64*)key || page->block_number < 0)
        printf("find error in page, page block %ld, key %ld\n", page->block_number, *(gint64*)key);
    if (page->p < page->g){
        ERROR("page %ld, p %d, g %d\n", page->block_number, page->p, page->g);
        abort(); 
    }
}



struct AMP_page* __AMP_insert_element_int(struct_cache* AMP, gint64 block){
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);
    
    struct AMP_page* page = g_new0(struct AMP_page, 1);
    page->block_number = block;
    // accessed is set in add_element
    
    GList* node = g_list_alloc();
    node->data = page;
    
    g_queue_push_tail_link(AMP_params->list, node);
    g_hash_table_insert(AMP_params->hashtable, (gpointer)&(page->block_number), (gpointer)node);
    return page;
}


void __AMP_insert_element(struct_cache* AMP, cache_line* cp){
    gint64 block = GET_BLOCK(cp);
    __AMP_insert_element_int(AMP, block); 
}


gboolean AMP_check_element_int(struct_cache* cache, gint64 block){
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);
    return g_hash_table_contains( AMP_params->hashtable, &block );
}


gboolean AMP_check_element(struct_cache* cache, cache_line* cp){
    gint64 block = GET_BLOCK(cp);
    return AMP_check_element_int(cache, block);
}



struct AMP_page* __AMP_update_element_int(struct_cache* cache, gint64 block){
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);
    GList* node = g_hash_table_lookup(AMP_params->hashtable, &block);
    
    g_queue_unlink(AMP_params->list, node);
    g_queue_push_tail_link(AMP_params->list, node);
    return (struct AMP_page*) (node->data);
}

void __AMP_update_element(struct_cache* cache, cache_line* cp){
    gint64 block = GET_BLOCK(cp);
    __AMP_update_element_int(cache, block);
}


void __AMP_evict_element(struct_cache* AMP, cache_line* cp){
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);

    struct AMP_page *page = (struct AMP_page*) g_queue_pop_head(AMP_params->list);
    if (page->old || page->accessed){
#ifdef TRACKED_BLOCK
        if (page->block_number == TRACKED_BLOCK)
            printf("ts %lu, final evict %d, old %d, accessed %d\n", cp->ts, TRACKED_BLOCK, page->old, page->accessed);
#endif
        gboolean result = g_hash_table_remove(AMP_params->hashtable, (gconstpointer)&(page->block_number));
        if (result == FALSE){
            fprintf(stderr, "ERROR nothing removed, block %ld\n", page->block_number);
            exit(1);
        }
        g_hash_table_remove(AMP_params->prefetched, &(page->block_number));
        page->block_number = -1;
        g_free(page);
    }
    else{
#ifdef TRACKED_BLOCK
        if (page->block_number == TRACKED_BLOCK)
            printf("ts %lu, try evict %d, old %d, accessed %d\n", cp->ts, TRACKED_BLOCK, page->old, page->accessed);
#endif 
        
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

void* __AMP__evict_with_return(struct_cache* AMP, cache_line* cp){
    /** return a pointer points to the data that being evicted, 
     *  it can be a pointer pointing to gint64 or a pointer pointing to char*
     *  it is the user's responsbility to g_free the pointer 
     **/
     
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);
    
    struct AMP_page *page = (struct AMP_page*) g_queue_pop_head(AMP_params->list);
    if (page->old || page->accessed){
        gboolean result = g_hash_table_remove(AMP_params->hashtable, (gconstpointer)&(page->block_number));
        if (result == FALSE){
            fprintf(stderr, "ERROR nothing removed, block %ld\n", page->block_number);
            exit(1);
        }
        gint64* return_data = g_new(gint64, 1);
        *(gint64*) return_data = page->block_number;

        g_hash_table_remove(AMP_params->prefetched, &(page->block_number));
        page->block_number = -1;
        g_free(page);
        return return_data;
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
        return __AMP__evict_with_return(AMP, cp); 
    }
}




gboolean AMP_add_element_no_eviction(struct_cache* AMP, cache_line* cp){
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);
    gint64 block;
    if (cp->type == 'l')
        block = *(gint64*) (cp->item_p);
    else{
        block = atoll(cp->item);
    }

    struct AMP_page* page = AMP_lookup(AMP, block);

    if (AMP_check_element_int(AMP, block)){
        // sanity check
        if (page == NULL)
            fprintf(stderr, "ERROR page is NULL\n");

        if (g_hash_table_contains(AMP_params->prefetched, &block)){
            AMP_params->num_of_hit ++;
            g_hash_table_remove(AMP_params->prefetched, &block);
        }

        if (page->accessed)
            __AMP_update_element_int(AMP, block);
        page->accessed = 1;

        gint length = 0;
        if (page->tag){
            // hit in the trigger and prefetch
            struct AMP_page* page_last = AMP_last(AMP, page);
            length = AMP_params->read_size;
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
            page->tag = FALSE;
            createPages_no_eviction(AMP, page->last_block_number+1, length);
        }
        return TRUE;
    }
    else{
        if (page != NULL)
            fprintf(stderr, "ERROR page is not NULL\n");
        page = __AMP_insert_element_int(AMP, block);
        page->accessed = 1;
        
        struct AMP_page* page_prev = AMP_lookup(AMP, block - 1);
        int length = AMP_params->read_size;
        if (page_prev && page_prev->p)
            length = page_prev->p;
        // miss -> prefetch
        if (page_prev){
            gboolean check = TRUE;
            int m;
            for (m=2; m<=AMP_params->K; m++)
                check = check && AMP_lookup(AMP, block-m);
            if (check)
                createPages_no_eviction(AMP, block+1, length);
        }
        
        return FALSE;
    }
}

gboolean AMP_add_element_only_no_eviction(struct_cache* AMP, cache_line* cp){
    
    gint64 block;
    if (cp->type == 'l')
        block = *(gint64*) (cp->item_p);
    else{
        block = atoll(cp->item);
    }
    
    struct AMP_page* page = AMP_lookup(AMP, block);
    
    if (AMP_check_element_int(AMP, block)){         // check is same
        // sanity check
        if (page == NULL)
            fprintf(stderr, "ERROR page is NULL\n");
        
        __AMP_update_element_int(AMP, block);       // update is same
        page->accessed = 1;
        
        return TRUE;
    }
    else{
        if (page != NULL)
            fprintf(stderr, "ERROR page is not NULL\n");
        page = __AMP_insert_element_int(AMP, block);    // insert is same
        page->accessed = 1;

        return FALSE;
    }
}


gboolean AMP_add_element_only(struct_cache* cache, cache_line* cp){
/* only add element, do not trigger other possible operation */
    
#ifdef TRACKED_BLOCK
    if (*(gint64*)(cp->item_p) == TRACKED_BLOCK)
        printf("ts %lu, add_only %d\n", cp->ts, TRACKED_BLOCK);
#endif
    gboolean ret_val;
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);
    ret_val = AMP_add_element_only_no_eviction(cache, cp);
    while ( (long)g_hash_table_size( AMP_params->hashtable) > cache->core->size)
        __AMP_evict_element(cache, cp);             // not sure
    return ret_val;
}



gboolean AMP_add_element_no_eviction_withsize(struct_cache* cache, cache_line* cp){
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);
    gint64 block, i, n = 1;
    
    if (cp->disk_sector_size != 0 && cache->core->block_unit_size !=0){
        *(gint64*)(cp->item_p) = (gint64) (*(gint64*)(cp->item_p) *
                                           cp->disk_sector_size /
                                           cache->core->block_unit_size);
        n = (int)ceil((double)cp->size/cache->core->block_unit_size);
        if (n<1)    // some traces have size zero for some requests
            n = 1;
    }

    block = *(gint64*) (cp->item_p);
    struct AMP_page* page = AMP_lookup(cache, block);
    
    if (AMP_check_element_int(cache, block)){
        // sanity check
        if (page == NULL)
            ERROR("check shows exist, but page is NULL\n");
        
        if (g_hash_table_contains(AMP_params->prefetched, &block)){
            AMP_params->num_of_hit ++;
            g_hash_table_remove(AMP_params->prefetched, &block);
        }
        
        if (page->accessed)
            __AMP_update_element_int(cache, block);
        page->accessed = 1;

        // new for withsize, keep reading the remaining pages
        if (cp->disk_sector_size != 0 && cache->core->block_unit_size !=0){
            gint64 old_block = (*(gint64*)(cp->item_p));
            for (i=0; i<n-1; i++){
                (*(gint64*)(cp->item_p)) ++;
                AMP_add_element_only(cache, cp);
            }
            (*(gint64*)(cp->item_p)) = old_block;
        }
        // end
        
        gint length = 0;
        if (page->tag){
            // hit in the trigger and prefetch
            struct AMP_page* page_last = AMP_last(cache, page);
            length = AMP_params->read_size;
            if (page_last && page_last->p)
                length = page_last->p;
        }
        
        gboolean check_result = AMP_isLast(page) && !(page->old);
        if (check_result){
            struct AMP_page* page_new = lastInSequence(cache, page);
            if (page_new)
                if (page_new->p + AMP_params->read_size < AMP_params->p_threshold)
                    page_new->p = page_new->p + AMP_params->read_size;
            ;
        }

        
        /* this is done here, because the page may be evicted when createPages */
        if (page->tag){
            // this page can come from prefetch or read at miss
            page->tag = FALSE;
            createPages_no_eviction(cache, page->last_block_number+1, length);
        }
        
        return TRUE;
    }
    // cache miss, load from disk
    else{
        if (page != NULL)
            ERROR("check non-exist, page is not NULL\n");
        page = __AMP_insert_element_int(cache, block);
        page->accessed = 1;
        struct AMP_page* page_prev = AMP_lookup(cache, block - 1);
        // NEED TO SET LAST PAGE
        AMP_lookup(cache, block)->last_block_number = block;
        
        
        // new for withsize, keep reading the remaining pages
        if (cp->disk_sector_size != 0 && cache->core->block_unit_size != 0){

            gint64 last_block = (*(gint64*)(cp->item_p)) + n -1;
            // update new last_block_number
            AMP_lookup(cache, block)->last_block_number = last_block;

            for (i=0; i<n-1; i++){
                (*(gint64*)(cp->item_p)) ++;
                
                // added 170505, for calculating precision
//                if (g_hash_table_contains(AMP_params->prefetched, cp->item_p)){
//                    AMP_params->num_of_hit ++;
//                    g_hash_table_remove(AMP_params->prefetched, cp->item_p);
//                }

                
                AMP_add_element_only(cache, cp);
                AMP_lookup(cache, (*(gint64*)(cp->item_p)))->last_block_number = last_block;
            }
            *(gint64*)(cp->item_p) -= (n-1);
        
#ifdef SANITY_CHECK
            if (*(gint64*)(cp->item_p) != block)
                ERROR("current block %ld, original %ld, n %d\n", *(gint64*)(cp->item_p), block, n);
            if (AMP_lookup(cache, block) == NULL)
                ERROR("requested block is not in cache after inserting, n %d, cache size %ld\n", n, cache->core->size);
#endif 
        
            // if n==1, then page_last is the same only page
            struct AMP_page* page_last = AMP_lookup(cache, last_block);
            page_last->p = (page_prev? page_prev->p: 0) + AMP_params->read_size;
            if (page_last->p > (int) (AMP_params->APT) ){
                page_last->g = (int) (AMP_params->APT / 2);
                struct AMP_page* tag_page =
                AMP_lookup(cache, page_last->block_number - (int)(AMP_params->APT/2) );
                if (tag_page){
                    tag_page->tag = TRUE;
                    tag_page->last_block_number = last_block;
                }
            }
            // this is possible, but not mentioned in the paper, anything wrong?
            if (page_last->p < page_last->g)
                page_last->g = page_last->p - 1 > 0 ? page_last->p - 1 : 0;
            
        }
        // end
        
        
        // prepare for prefetching on miss
        int length = AMP_params->read_size;
        if (page_prev && page_prev->p)
            length = page_prev->p;
        
        // miss -> prefetch
        if (page_prev){
            gboolean check = TRUE;
            int m;  // m begins with 2, because page_prev already exists
            for (m=2; m<=AMP_params->K; m++)
                check = check && AMP_lookup(cache, block-m);
            if (check){
                // new 170505, for calculating precision
//                if (cp->disk_sector_size != 0 && cache->core->block_unit_size != 0){
//                    block = (*(gint64*)(cp->item_p)) + n -1;
//                    length -= (long)(n - 1);
//                }
//                if (length > 0)
//                    createPages_no_eviction(cache, block + 1, length);
                // end
                createPages_no_eviction(cache, block + 1, length);
            }
        }
        
        return FALSE;
    }
}


gboolean AMP_add_element_withsize(struct_cache* cache, cache_line* cp){
#ifdef TRACKED_BLOCK
    // debug
    if (*(gint64*)(cp->item_p) == TRACKED_BLOCK)
        printf("ts %lu, add %d\n", cp->ts, TRACKED_BLOCK);
#endif
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);

    gboolean ret_val = AMP_add_element_no_eviction_withsize(cache, cp);
    while ( (long)g_hash_table_size( AMP_params->hashtable) > cache->core->size)
        __AMP_evict_element(cache, cp);
    
    return ret_val;
}








gboolean AMP_add_element(struct_cache* AMP, cache_line* cp){
#ifdef TRACKED_BLOCK
    // debug
    if (*(gint64*)(cp->item_p) == TRACKED_BLOCK)
        printf("ts %lu, add %d\n", cp->ts, TRACKED_BLOCK);
#endif
    
    struct AMP_params* AMP_params = (struct AMP_params*)(AMP->cache_params);
    gboolean result = AMP_add_element_no_eviction(AMP, cp);
    while ( (long)g_hash_table_size( AMP_params->hashtable) > AMP->core->size)
        __AMP_evict_element(AMP, cp);
    return result;
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


struct_cache* AMP_init(guint64 size, char data_type, int block_size, void* params){
    struct_cache *cache = cache_init(size, data_type, block_size);
    cache->cache_params = g_new0(struct AMP_params, 1);
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);
    struct AMP_init_params* init_params = (struct AMP_init_params*) params;
    
    cache->core->type                           =       e_AMP;
    cache->core->cache_init                     =       AMP_init;
    cache->core->destroy                        =       AMP_destroy;
    cache->core->destroy_unique                 =       AMP_destroy_unique;
    cache->core->add_element                    =       AMP_add_element;
    cache->core->check_element                  =       AMP_check_element;
    cache->core->__insert_element               =       __AMP_insert_element;
    cache->core->__update_element               =       __AMP_update_element;
    cache->core->__evict_element                =       __AMP_evict_element;
    cache->core->__evict_with_return            =       __AMP__evict_with_return; 

    cache->core->get_size                       =       AMP_get_size;
    cache->core->cache_init_params              =       params;
    cache->core->add_element_only               =       AMP_add_element_only;
    cache->core->add_element_withsize           =       AMP_add_element_withsize; 
    
    AMP_params->K           =   init_params->K;
    AMP_params->APT         =   init_params->APT;
    AMP_params->read_size   =   init_params->read_size;
    AMP_params->p_threshold =   init_params->p_threshold;
    

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



gint64 AMP_get_size(struct_cache* cache){
    struct AMP_params* AMP_params = (struct AMP_params*)(cache->cache_params);
    return (guint64) g_hash_table_size(AMP_params->hashtable);
}



#ifdef __cplusplus
}
#endif


