
#include "cache.h" 






void cache_destroy(struct_cache* cache){
    if (cache->cache_params)
        g_free(cache->cache_params);
    // cache->core->cache_init_params is on the stack default, if it is on the heap, needs to be freed manually 
    if (cache->core->cache_init_params)
        g_free(cache->core->cache_init_params);
    g_free(cache->core);
    g_free(cache);
}

void cache_destroy_unique(struct_cache* cache){
    if (cache->cache_params)
        g_free(cache->cache_params);
    g_free(cache->core);
    g_free(cache);
}


struct_cache* cache_init(long long size, char data_type){
    struct_cache *cache = g_new0(struct_cache, 1);
    cache->core = g_new0(struct cache_core, 1);
    cache->core->size = size;
    cache->core->cache_init_params = NULL;
    cache->core->data_type = data_type;
    
    
    return cache;
}