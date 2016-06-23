//
//  optimal.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef optimal_h
#define optimal_h


#include "cache.h"
#include "pqueue.h" 
#include "heatmap.h"


/* need add support for p and c type of data
 
 */




//typedef struct optimal{
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
//        char cache_params[1024];
//    };
//    
//    
//}Optimal;


struct optimal_params{
    GHashTable *hashtable;
    pqueue_t *pq;
    GArray* next_access;
    long long ts;       // virtual time stamp
    READER* reader;
};


struct optimal_init_params{
    READER* reader;
    GArray* next_access;
    long long ts;
};






inline void __optimal_insert_element_long(struct_cache* optimal, cache_line* cp);

inline gboolean optimal_check_element_long(struct_cache* cache, cache_line* cp);

inline void __optimal_update_element_long(struct_cache* optimal, cache_line* cp);

inline void __optimal_evict_element(struct_cache* optimal);

inline gboolean optimal_add_element_long(struct_cache* cache, cache_line* cp);

inline void optimal_destroy(struct_cache* cache);
inline void optimal_destroy_unique(struct_cache* cache);

struct_cache* optimal_init(long long size, char data_type, void* params);



#endif