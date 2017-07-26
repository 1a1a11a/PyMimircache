//
//  Random.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef Random_h
#define Random_h

#include "cache.h"



#ifdef __cplusplus
extern "C"
{
#endif


struct Random_params{
    GHashTable *hashtable;
    GArray* array;
};





extern  void __Random_insert_element(struct_cache* Random, cache_line* cp);

extern  gboolean Random_check_element(struct_cache* cache, cache_line* cp);

extern  void __Random_update_element(struct_cache* Random, cache_line* cp);

extern  void __Random_evict_element(struct_cache* Random, cache_line* cp);

extern  gboolean Random_add_element(struct_cache* cache, cache_line* cp);

extern  void Random_destroy(struct_cache* cache);
extern  void Random_destroy_unique(struct_cache* cache);

struct_cache* Random_init(guint64 size, char data_type, int block_size, void* params);


#ifdef __cplusplus
}

#endif


#endif /* Random_h */
