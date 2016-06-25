

#include "generalProfiler.h" 

/* this module is not reentrant-safe */


static void profiler_thread(gpointer data, gpointer user_data){
    
    int i;
    struct multithreading_params_generalProfiler* params = (struct multithreading_params_generalProfiler*) data;
    int num_of_threads = params->num_of_threads;
    int order = params->order;
    guint64 begin_pos = params->begin_pos;
    guint64 end_pos = params->end_pos;
    guint64 pos = begin_pos;
    
    struct_cache** caches = params->caches;
    return_res** result = params->result;
        
    READER* reader_thread = copy_reader(params->reader);
    
    skip_N_elements(reader_thread, begin_pos);
    
    
    // create cache lize struct and initialization
    cache_line* cp = (cache_line*)malloc(sizeof(cache_line));
    cp->op = -1;
    cp->size = -1;
    cp->valid = TRUE;
    
    
    read_one_element(reader_thread, cp);
    while (cp->valid && pos<end_pos){
        for (i=0; i<params->num_of_cache; i++){
            if ((caches[i*num_of_threads+order])->core->add_element(caches[i*num_of_threads+order], cp))
                result[i*num_of_threads+order]->hit_count ++;
            else
                result[i*num_of_threads+order]->miss_count ++;
        }
        pos++;
        read_one_element(reader_thread, cp);
    }
    
    for (i=0; i<params->num_of_cache; i++){
        result[i*num_of_threads+order]->total_count = result[i*num_of_threads+order]->hit_count +
                                                        result[i*num_of_threads+order]->miss_count;
        result[i*num_of_threads+order]->hit_rate = (float)(result[i*num_of_threads+order]->hit_count) /
                                                        result[i*num_of_threads+order]->total_count;
        result[i*num_of_threads+order]->miss_rate = 1 - result[i*num_of_threads+order]->hit_rate;
    }
    
    // clean up
    free(cp);
    if (reader_thread->type != 'v')
        fclose(reader_thread->file);
    free(reader_thread);
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
    
    long num_of_bins = ceil(cache_in->core->size/bin_size);
    
    if (end_pos==-1)
        if (reader_in->total_num == -1)
            end_pos = get_num_of_cache_lines(reader_in);
    
    // create the result storage area and caches of varying sizes
    return_res** result = (return_res**) malloc(num_of_bins * sizeof(return_res*));
    struct_cache** caches = (struct_cache**) malloc(sizeof(struct_cache*)*num_of_bins);

    for (i=0; i<num_of_bins; i++){
        result[i] = (return_res*) calloc(1, sizeof(return_res));
        result[i]->cache_size = bin_size * (i+1);
        caches[i] = cache_in->core->cache_init(bin_size * (i+1), cache_in->core->data_type, cache_in->core->cache_init_params);
    }

    
    
    // build the thread pool
    GThreadPool * gthread_pool = g_thread_pool_new ( (GFunc) profiler_thread, NULL, num_of_threads, TRUE, NULL);
    if (gthread_pool == NULL)
        g_error("cannot create thread pool in general profiler\n");
    
    // build parameters and send to thread pool
    struct multithreading_params_generalProfiler** params = malloc(sizeof(struct multithreading_params_generalProfiler*) * num_of_threads);
    int residual = num_of_bins % num_of_threads;
    for (i=0; i<num_of_threads; i++){
        params[i] = malloc(sizeof(struct multithreading_params_generalProfiler));
        params[i]->reader = reader_in;
        params[i]->caches = caches;
        params[i]->result = result;
        params[i]->bin_size = bin_size;
        params[i]->num_of_threads = num_of_threads;
        params[i]->order = i;
        params[i]->num_of_cache = num_of_bins/num_of_threads;
        params[i]->begin_pos = begin_pos;
        params[i]->end_pos = end_pos;
        if (i < residual)
            params[i]->num_of_cache ++ ;
        if ( g_thread_pool_push (gthread_pool, (gpointer) params[i], NULL) == FALSE)
            g_error("cannot push data into thread in generalprofiler\n");
    }
    
    g_thread_pool_free (gthread_pool, FALSE, TRUE);

    // clean up
    for (i=0; i<num_of_bins; i++)
        caches[i]->core->destroy_unique(caches[i]);
    
    for (i=0; i<num_of_threads; i++)
        free(params[i]);
    free(caches);
    free(params);
    // needs to free result later
    
    
    return result;
}



//
//
//#include "reader.h"
//#include "FIFO.h"
//#include "Optimal.h"
//
//int main(int argc, char* argv[]){
//# define CACHESIZE 200
//# define BIN_SIZE 20
//    
//    
//    printf("test_begin!\n");
//    
//    READER* reader = setup_reader(argv[1], 'v');
//    
////    struct_cache* cache = fifo_init(CACHESIZE, 'v', NULL);
//    
//    struct optimal_init_params init_params = {.reader=reader, .next_access=NULL, .ts=0};
//    
//    struct_cache* cache = optimal_init(CACHESIZE, 'v', (void*)&init_params);
//    
//    
//    printf("after initialization, begin profiling\n");
//    return_res** res = profiler(reader, cache, CACHESIZE, BIN_SIZE, 23, 43);
//    
//    int i;
//    for (i=0; i<CACHESIZE/BIN_SIZE; i++){
//        printf("%lld: %f\n", res[i]->cache_size, res[i]->hit_rate);
//        free(res[i]);
//    }
//    
//    cache->core->destroy(cache);
//    free(res);
//    printf("after profiling\n");
//
//    cache = optimal_init(CACHESIZE, 'v', (void*)&init_params);
//    
//    printf("after initialization, begin profiling\n");
//    res = profiler(reader, cache, CACHESIZE, BIN_SIZE, 23, 43);
//    
//    for (i=0; i<CACHESIZE/BIN_SIZE; i++){
//        printf("%lld: %f\n", res[i]->cache_size, res[i]->miss_rate);
//        free(res[i]);
//    }
//    
//    cache->core->destroy(cache);
//    free(res);
//    printf("after profiling\n");
//
//    
//    cache = optimal_init(CACHESIZE, 'v', (void*)&init_params);
//    
//    printf("after initialization, begin profiling\n");
//    res = profiler(reader, cache, CACHESIZE, BIN_SIZE, 23, 43);
//    
//    for (i=0; i<CACHESIZE/BIN_SIZE; i++){
//        printf("%lld: %f\n", res[i]->cache_size, res[i]->miss_rate);
//        free(res[i]);
//    }
//    
//    cache->core->destroy(cache);
//    free(res);
//    
//    close_reader(reader);
//    
//    
//    printf("test_finished!\n");
//    return 0;
//}
