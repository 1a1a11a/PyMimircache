//
//  cache.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef cache_h
#define cache_h

#include <stdio.h>
#include <glib.h>
#include "reader.h"
#include <stdlib.h> 
#include <string.h> 
#include "glib_related.h"




typedef enum{
    e_LRU,
    e_Optimal,
    e_FIFO
}cache_type;


struct cache_core{
    cache_type type;
    long size;
    char data_type;     // v, p, c
    long long hit_count;
    long long miss_count;
    void* cache_init_params;
    struct cache* (*cache_init)(long long, char, void*);
    void (*destroy)(struct cache* );
    void (*destroy_unique)(struct cache* );
    gboolean (*add_element)(struct cache*, cache_line* cp);
    gboolean (*check_element)(struct cache*, cache_line* cp);
};


typedef struct cache{
    struct cache_core *core;
    void* cache_params;
}struct_cache;



//#define core->type type;
//#define core->size size;
//#define data_type core->data_type;
//#define hit_count core->hit_count;
//#define miss_count core->miss_count;
//#define cache_init_params core->cache_init_params;
//#define cache_init core->cache_init;
//#define destroy core->destroy;
//#define destroy_unique core->destroy_unique;
//#define add_element core->add_element;
//#define check_element core->check_element;


//    union{
//        char cache_param[1024];
//    };

struct_cache* cache_init(long long size, char data_type);
void cache_destroy(struct_cache* cache);






#endif /* cache_h */
