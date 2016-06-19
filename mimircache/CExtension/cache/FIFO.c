//
//  FIFO.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h" 
#include "FIFO.h"

/* need add support for p and c type of data 
 
 */



//
//typedef struct{
//    cache_type type;
//    long size;
//    char data_type;
//    long long hit_count;
//    long long miss_count;
//    void* cache_init_params;
//    struct cache* (*cache_init)(long long, char, void*);
//    void (*destroy)(struct_cache* );
//    gboolean (*add_element)(struct_cache*, cache_line* cp);
//    gboolean (*check_element)(struct_cache*, cache_line* cp);
//    
//    union{
//        struct{
//            GHashTable *hashtable;
//            GSList *list;
//        };
//        char cache_params[1024];
//    };
//    
//    
//}FIFO;
//



inline void __fifo_insert_element_long(FIFO* fifo, cache_line* cp){
    gint64* key = g_new(gint64, 1);
    if (key == NULL){
        printf("not enough memory\n");
        exit(1);
    }
    *key = cp->long_content;
    g_hash_table_add(fifo->hashtable, (gpointer)key);
    fifo->list = g_slist_append(fifo->list, (gpointer)key);
}

inline gboolean fifo_check_element_long(struct_cache* cache, cache_line* cp){
    return g_hash_table_contains( ((FIFO*)cache)->hashtable, (gconstpointer)(&(cp->long_content)) );
}


inline void __fifo_update_element_long(FIFO* fifo, cache_line* cp){
    return;
}


inline void __fifo_evict_element(FIFO* fifo){
    gpointer data = fifo->list->data;
    fifo->list = g_slist_remove (fifo->list, (gconstpointer)data);
    g_hash_table_remove(fifo->hashtable, (gconstpointer)data);
}




inline gboolean fifo_add_element_long(struct_cache* cache, cache_line* cp){
    if (fifo_check_element_long(cache, cp)){
        return TRUE;
    }
    else{
        __fifo_insert_element_long((FIFO*)cache, cp);
        if ( (long)g_hash_table_size( ((FIFO*)cache)->hashtable ) > cache->size)
            __fifo_evict_element((FIFO*)cache);
        return FALSE;
    }
}




 inline void fifo_destroy(struct_cache* cache){
    FIFO* fifo = (FIFO* )cache;
    g_slist_free(fifo->list);
    g_hash_table_destroy(fifo->hashtable);
    free(fifo);
}

inline void fifo_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    FIFO* fifo = (FIFO* )cache;
    g_slist_free(fifo->list);
    g_hash_table_destroy(fifo->hashtable);
    free(fifo);
}


struct_cache* fifo_init(long long size, char data_type, void* params){
    FIFO* fifo = (FIFO*) calloc(1, sizeof(FIFO));
    
//    strcpy(fifo->name, "FIFO");
    fifo->type = e_FIFO;
    fifo->size = size;
    fifo->data_type = data_type;
    fifo->cache_init = fifo_init;
    fifo->destroy = fifo_destroy;
    fifo->destroy_unique = fifo_destroy_unique;
    fifo->cache_init_params = NULL;

    if (data_type == 'v'){
        fifo->add_element = fifo_add_element_long;
        fifo->check_element = fifo_check_element_long;
        fifo->hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_key_value_destroyed, NULL);
    }
    
    else if (data_type == 'p'){
        printf("not supported yet\n");
    }
    else if (data_type == 'c'){
        printf("not supported yet\n");
    }
    else{
        g_error("does not support given data type: %c\n", data_type);
    }
    fifo->list = NULL;
    
    
    return (struct_cache*)fifo;
}



