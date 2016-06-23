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



typedef struct cache{
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
    
//    union{
//        char cache_param[1024];
//    };
    void* cache_param;
    
    
}struct_cache;






#endif /* cache_h */
