
#include "cache.h" 






void cache_destroy(struct_cache* cache){
    if (cache->cache_params)
        free(cache->cache_params);
    // cache->core->cache_init_params is on the stack default, if it is on the heap, needs to be freed manually 
    if (cache->core->cache_init_params)
        free(cache->core->cache_init_params);
    free(cache->core);
    free(cache);
}

void cache_destroy_unique(struct_cache* cache){
    if (cache->cache_params)
        free(cache->cache_params);
    free(cache->core);
    free(cache);
}


struct_cache* cache_init(long long size, char data_type){
    struct_cache* cache = (struct_cache*) calloc(1, sizeof(struct_cache));
    cache->core = (struct cache_core*) calloc(1, sizeof(struct cache_core));
    cache->core->size = size;
    cache->core->data_type = data_type;
    cache->core->cache_init_params = NULL;
    
    return cache;
}