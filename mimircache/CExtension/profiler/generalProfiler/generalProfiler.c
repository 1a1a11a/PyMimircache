

#include "generalProfiler.h" 

/* this module is not reentrant-safe */


static void profiler_thread(gpointer data, gpointer user_data){
    struct multithreading_params_generalProfiler* params = (struct multithreading_params_generalProfiler*) user_data;

    int order = GPOINTER_TO_UINT(data);
    guint64 begin_pos = params->begin_pos;
    guint64 end_pos = params->end_pos;
    guint64 pos = begin_pos;
    guint bin_size = params->bin_size;
    
    struct_cache* cache = params->cache->core->cache_init(bin_size * order,
                                                          params->cache->core->data_type,
                                                          params->cache->core->cache_init_params);

    return_res** result = params->result;
        
    READER* reader_thread = copy_reader(params->reader);
    
    skip_N_elements(reader_thread, begin_pos);
    
    
    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = params->cache->core->data_type;
    
    guint64 hit_count=0, miss_count=0;
    gboolean (*add_element)(struct cache*, cache_line* cp);
    add_element = cache->core->add_element;

    
    read_one_element(reader_thread, cp);
    while (cp->valid && pos<end_pos){
        if (add_element(cache, cp))
            hit_count ++;
        else
            miss_count ++;
        pos++;
        read_one_element(reader_thread, cp);
    }
    
    result[order]->hit_count = hit_count;
    result[order]->miss_count = miss_count;
    result[order]->total_count = hit_count + miss_count;
    result[order]->hit_rate = (double) hit_count / (hit_count + miss_count);
    result[order]->miss_rate = 1 - result[order]->hit_rate;
    
    
    // clean up
    g_free(cp);
    if (reader_thread->type != 'v')
        fclose(reader_thread->file);
    g_free(reader_thread);
    cache->core->destroy_unique(cache);

}

    
    
return_res** profiler(READER* reader_in, struct_cache* cache_in, int num_of_threads_in, int bin_size_in, gint64 begin_pos, gint64 end_pos){
    /**
     if profiling from the very beginning, then set begin_pos=0, 
     if porfiling till the end of trace, then set end_pos=-1 or the length of trace+1; 
     return results do not include size 0 
     **/
    
    int i;
    
    if (end_pos<=begin_pos && end_pos!=-1){
        printf("end pos <= beigin pos in general profiler, please check\n");
        exit(1);
    }
    
    
    // initialization
    int num_of_threads = num_of_threads_in;
    int bin_size = bin_size_in;
    
    long num_of_bins = ceil(cache_in->core->size/bin_size)+1;
    
    if (end_pos==-1)
        if (reader_in->total_num == -1)
            end_pos = get_num_of_cache_lines(reader_in);
    
    // create the result storage area and caches of varying sizes
    return_res** result = g_new(return_res*, num_of_bins);

    for (i=0; i<num_of_bins; i++){
        result[i] = g_new0(return_res, 1);
        result[i]->cache_size = bin_size * (i+1);
    }
    result[0]->miss_rate = 1;
    
    
    // build parameters and send to thread pool
    struct multithreading_params_generalProfiler* params = g_new(struct multithreading_params_generalProfiler, 1);
        params->reader = reader_in;
        params->cache = cache_in;
        params->result = result;
        params->bin_size = (guint) bin_size;
        params->begin_pos = begin_pos;
        params->end_pos = end_pos;

    // build the thread pool
    GThreadPool * gthread_pool = g_thread_pool_new ( (GFunc) profiler_thread, (gpointer)params, num_of_threads, TRUE, NULL);
    if (gthread_pool == NULL)
        g_error("cannot create thread pool in general profiler\n");
    
    
    for (i=1; i<num_of_bins; i++){
        if ( g_thread_pool_push (gthread_pool, GUINT_TO_POINTER(i), NULL) == FALSE)
            g_error("cannot push data into thread in generalprofiler\n");
    }

    g_thread_pool_free (gthread_pool, FALSE, TRUE);

    // clean up
    g_free(params);
    // needs to free result later
        
    return result;
}





//#include "reader.h"
//#include "FIFO.h"
//#include "Optimal.h"
//#include "LRU_K.h"
//
//int main(int argc, char* argv[]){
//# define CACHESIZE 2000
//# define BIN_SIZE 200
//    
//    
//    printf("test_begin!\n");
//    
//    READER* reader = setup_reader(argv[1], 'v');
//    
////    struct_cache* cache = fifo_init(CACHESIZE, 'v', NULL);
//    
//    struct optimal_init_params* init_params = g_new0(struct optimal_init_params, 1);
//    init_params->reader = reader;
////    init_params->ts = 0;
//    struct_cache* cache = optimal_init(CACHESIZE, 'l', (void*)init_params);
//    
////    struct LRU_K_init_params *LRU_K_init_params = (struct LRU_K_init_params*) malloc(sizeof(struct LRU_K_init_params));
////    LRU_K_init_params->K = 1;
////    LRU_K_init_params->maxK = 1;
////    struct_cache* cache = LRU_K_init(CACHESIZE, 'v', LRU_K_init_params);
//    
//    
//    printf("after initialization, begin profiling\n");
//    
//
//    return_res** res = profiler(reader, cache, 8, BIN_SIZE, 0, -1);
////    return_res** res = profiler(reader, cache, 1, BIN_SIZE, 23, 43);
//    
//    int i;
//    for (i=0; i<CACHESIZE/BIN_SIZE+1; i++){
//        printf("%lld: %f\n", res[i]->cache_size, res[i]->hit_rate);
//        g_free(res[i]);
//    }
//    
//    cache->core->destroy(cache);
//    g_free(res);
//    printf("after profiling\n");
//
//
//    close_reader(reader);
//    
//    
//    printf("test_finished!\n");
//    return 0;
//}
