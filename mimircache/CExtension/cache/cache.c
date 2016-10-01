
#include "cache.h" 






void cache_destroy(struct_cache* cache){
    if (cache->cache_params){
        g_free(cache->cache_params);
        cache->cache_params = NULL;
    }
    
    /* cache->core->cache_init_params is on the stack default, 
     if it is on the heap, needs to be freed manually
     */
    if (cache->core->cache_init_params){
        g_free(cache->core->cache_init_params);
        cache->core->cache_init_params = NULL;
    }
    
    // This should not be freed, because it points to other's eviction_array, which should be freed only by others
//    if (cache->core->oracle){
//        g_free(cache->core->oracle);
//        cache->core->oracle = NULL;
//    }

    
    if (cache->core->eviction_array){
        
        if (cache->core->data_type == 'l')
            g_free((guint64*)cache->core->eviction_array);
        else{
            guint64 i;
            for (i=0; i<cache->core->eviction_array_len; i++)
                if ( ((gchar**)cache->core->eviction_array)[i] != 0 )
                    g_free(((gchar**)cache->core->eviction_array)[i]);
            g_free( (gchar**)cache->core->eviction_array );
        }
        cache->core->eviction_array = NULL;
    }
    
    if (cache->core->evict_err_array){
        g_free(cache->core->evict_err_array);
        cache->core->evict_err_array = NULL;
    }
    
    g_free(cache->core);
    cache->core = NULL;
    g_free(cache);
}

void cache_destroy_unique(struct_cache* cache){
    if (cache->cache_params){
        g_free(cache->cache_params);
        cache->cache_params = NULL;
    }
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

guint64 get_current_size(struct_cache* cache){
    return 0;
}