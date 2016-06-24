//
//  FIFO.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef fifo_h
#define fifo_h


#include "cache.h" 

/* need add support for p and c type of data 
 
 */




//typedef struct{
//    cache_type type;
//    long size;
//    char data_type;
//    long long hit_count;
//    long long miss_count;
//    void* cache_init_params;
//    struct cache* (*cache_init)(long long, char, void*);
//    void (*destroy)(struct_cache* );
//    void (*destroy_unique)(struct cache* );
//    gboolean (*add_element)(struct_cache*, cache_line* cp);
//    gboolean (*check_element)(struct_cache*, cache_line* cp);
//    
//    union{
//                char cache_params[1024];
//    };
//}FIFO;

struct FIFO_params{
    GHashTable *hashtable;
    GQueue *list;
};



extern inline void __fifo_insert_element_long(struct_cache* fifo, cache_line* cp);

extern inline gboolean fifo_check_element_long(struct_cache* cache, cache_line* cp);

extern inline void __fifo_update_element_long(struct_cache* fifo, cache_line* cp);

extern inline void __fifo_evict_element(struct_cache* fifo);

extern inline gboolean fifo_add_element_long(struct_cache* cache, cache_line* cp);

extern inline void fifo_destroy(struct_cache* cache);
extern inline void fifo_destroy_unique(struct_cache* cache);


struct_cache* fifo_init(long long size, char data_type, void* params);


#endif