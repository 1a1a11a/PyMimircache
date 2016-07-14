

#include "generalProfiler.h" 


#include "reader.h"
#include "FIFO.h"
#include "Optimal.h"
#include "LRU_K.h"
#include "LRU.h"
#include "python_wrapper.h"
#include "LRU_dataAware.h"




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
    
    if (begin_pos!=0 && begin_pos!=-1)
        skip_N_elements(reader_thread, begin_pos); 
    
    
    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = params->cache->core->data_type;
    
    guint64 hit_count=0, miss_count=0;
    gboolean (*add_element)(struct cache*, cache_line* cp);
    add_element = cache->core->add_element;

    read_one_element(reader_thread, cp);
    while (cp->valid && pos<end_pos){
        if (add_element(cache, cp)){
            hit_count ++;
        }
        else
            miss_count ++;
        pos++;
        read_one_element(reader_thread, cp);
    }
    
//    DEBUG(printf("hit count %lu, total count %lu\n", hit_count, miss_count+hit_count));
    result[order]->hit_count = (long long) hit_count;
    result[order]->miss_count = (long long) miss_count;
    result[order]->total_count = hit_count + miss_count;
    result[order]->hit_rate = (double) hit_count / (hit_count + miss_count);
    result[order]->miss_rate = 1 - result[order]->hit_rate;
    
    // clean up
    g_mutex_lock(&(params->mtx));
    (*(params->progress)) ++ ;
    g_mutex_unlock(&(params->mtx));

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
    
    long i;
    guint64 progress = 0;
    
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
        result[i]->cache_size = bin_size * (i);
    }
    result[0]->miss_rate = 1;
    
    
    // build parameters and send to thread pool
    struct multithreading_params_generalProfiler* params = g_new0(struct multithreading_params_generalProfiler, 1);
    params->reader = reader_in;
    params->cache = cache_in;
    params->result = result;
    params->bin_size = (guint) bin_size;
    params->begin_pos = begin_pos;
    params->end_pos = end_pos;
    params->progress = &progress;
    g_mutex_init(&(params->mtx));

    // build the thread pool
    GThreadPool * gthread_pool = g_thread_pool_new ( (GFunc) profiler_thread, (gpointer)params, num_of_threads, TRUE, NULL);
    if (gthread_pool == NULL)
        g_error("cannot create thread pool in general profiler\n");
    
    
    for (i=1; i<num_of_bins; i++){
        if ( g_thread_pool_push (gthread_pool, GUINT_TO_POINTER(i), NULL) == FALSE)
            g_error("cannot push data into thread in generalprofiler\n");
    }
    
    while (progress < (guint64)num_of_bins-1){
        printf("%.2f%%\n", ((double)progress) / (num_of_bins-1) * 100);
        sleep(1);
        printf("\033[A\033[2K\r");
    }
    
    g_thread_pool_free (gthread_pool, FALSE, TRUE);

    // change hit count, now it is accumulated hit_count, change to real hit_count
    for (i=num_of_bins-1; i>=1; i--)
        result[i]->hit_count -= result[i-1]->hit_count;
    
    // clean up
    g_mutex_clear(&(params->mtx));
    g_free(params);
    // needs to free result later
        
    return result;
}


static void traverse_trace(READER* reader, struct_cache* cache){
    
    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = cache->core->data_type;
    
    gboolean (*add_element)(struct cache*, cache_line* cp);
    add_element = cache->core->add_element;
    
    read_one_element(reader, cp);
    while (cp->valid){
        add_element(cache, cp);
        read_one_element(reader, cp);
    }
    
    // clean up
    g_free(cp);
    reset_reader(reader);
    
}


static void get_evict_err(READER* reader, struct_cache* cache){
    
    cache->core->bp_pos = 1;
    cache->core->evict_err_array = g_new0(gdouble, reader->break_points->array->len-1);

    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = cache->core->data_type;
    
    gboolean (*add_element)(struct cache*, cache_line* cp);
    add_element = cache->core->add_element;
    
    read_one_element(reader, cp);
    while (cp->valid){
        add_element(cache, cp);
        read_one_element(reader, cp);
    }
    
    // clean up
    g_free(cp);
    reset_reader(reader);
    
}





gdouble* LRU_evict_err_statistics(READER* reader_in, struct_cache* cache_in, guint64 time_interval){
    
    gen_breakpoints_realtime(reader_in, time_interval);
    cache_in->core->bp = reader_in->break_points;
    cache_in->core->cache_debug_level = 2;
    
    
    struct optimal_init_params* init_params = g_new0(struct optimal_init_params, 1);
    init_params->reader = reader_in;
    init_params->ts = 0;
    struct_cache* optimal; 
    if (cache_in->core->data_type == 'l')
        optimal = optimal_init(cache_in->core->size, 'l', (void*)init_params);
    else{
        printf("other cache data type not supported in LRU_evict_err_statistics in generalProfiler\n");
        exit(1);
    }
    optimal->core->cache_debug_level = 1;
    optimal->core->eviction_array_len = reader_in->total_num;
    optimal->core->bp = reader_in->break_points;
    
    if (reader_in->total_num == -1)
        get_num_of_cache_lines(reader_in);
    
    if (reader_in->type == 'v')
        optimal->core->eviction_array = g_new0(guint64, reader_in->total_num);
    else
        optimal->core->eviction_array = g_new0(gchar*, reader_in->total_num);
    
    // get oracle
    traverse_trace(reader_in, optimal);

    cache_in->core->oracle = optimal->core->eviction_array;
    
    
    get_evict_err(reader_in, cache_in);
    
    
        
    optimal_destroy(optimal);
    
    
    return cache_in->core->evict_err_array;
}


static void profiler_with_prefetch_thread(gpointer data, gpointer user_data){
    struct multithreading_params_generalProfiler* params = (struct multithreading_params_generalProfiler*) user_data;
    
    int order = GPOINTER_TO_UINT(data);
    guint64 begin_pos = params->begin_pos;
    guint64 end_pos = params->end_pos;
    guint64 pos = begin_pos;
    guint bin_size = params->bin_size;
    GHashTable *prefetch_hashtable = params->prefetch_hashtable;
    
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
        if (add_element(cache, cp)){
            hit_count ++;
        }
        else
            miss_count ++;
        pos++;
        
        // begin prefetch elements
        GSList *list = g_hash_table_lookup(prefetch_hashtable, cp->item_p);
        while (list != NULL){
            if (cp->type == 'l')
                *((guint64*) cp->item_p) = *((guint64*)list->data);
            else
                strcpy((char*)(cp->item_p), (char*)(list->data));
            add_element(cache, cp);
            list = list->next;
        }
        
        read_one_element(reader_thread, cp);
    }
    
    
    result[order]->hit_count = hit_count;
    result[order]->miss_count = miss_count;
    result[order]->total_count = hit_count + miss_count;
    result[order]->hit_rate = (double) hit_count / (hit_count + miss_count);
    result[order]->miss_rate = 1 - result[order]->hit_rate;
    
    
    // clean up
    g_mutex_lock(&(params->mtx));
    (*(params->progress)) ++ ;
    g_mutex_unlock(&(params->mtx));
    
    g_free(cp);
    if (reader_thread->type != 'v')
        fclose(reader_thread->file);
    g_free(reader_thread);
    cache->core->destroy_unique(cache);
    
}



return_res** profiler_with_prefetch(READER* reader_in, struct_cache* cache_in, int num_of_threads_in, int bin_size_in, char* prefetch_file_loc, gint64 begin_pos, gint64 end_pos){
    /**
     if profiling from the very beginning, then set begin_pos=0,
     if porfiling till the end of trace, then set end_pos=-1 or the length of trace+1;
     return results do not include size 0
     **/
    
    long i;
    guint64 progress = 0;
    
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
        result[i]->cache_size = bin_size * (i);
    }
    result[0]->miss_rate = 1;
    
    // build prefetch hashtable
    GHashTable *prefetch_hashtable;
    if (cache_in->core->data_type == 'l'){
        prefetch_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_g_key_value_destroyer, g_slist_destroyer);
        FILE* file = fopen(prefetch_file_loc, "r");
        char buf[1024*10];
        char *token;
        GSList* list = NULL;
        guint64 k;
        while (fscanf(file, "%s", buf) != 0){
            printf("%s\n", buf);
            token = strtok(buf, ",");
            sscanf(token, "%lu", &k);
            
        }
        
    }
    else
        prefetch_hashtable = g_hash_table_new_full(g_str_hash, g_str_equal, simple_g_key_value_destroyer, g_slist_destroyer);
    
    
    

    
    
    // build parameters and send to thread pool
    struct multithreading_params_generalProfiler* params = g_new(struct multithreading_params_generalProfiler, 1);
    params->reader = reader_in;
    params->cache = cache_in;
    params->result = result;
    params->bin_size = (guint) bin_size;
    params->begin_pos = begin_pos;
    params->end_pos = end_pos;
    params->prefetch_hashtable = prefetch_hashtable;
    params->progress = &progress;
    g_mutex_init(&(params->mtx));
    
    // build the thread pool
    GThreadPool * gthread_pool = g_thread_pool_new ( (GFunc) profiler_thread, (gpointer)params, num_of_threads, TRUE, NULL);
    if (gthread_pool == NULL)
        g_error("cannot create thread pool in general profiler\n");
    
    
    for (i=1; i<num_of_bins; i++){
        if ( g_thread_pool_push (gthread_pool, GUINT_TO_POINTER(i), NULL) == FALSE)
            g_error("cannot push data into thread in generalprofiler\n");
    }
    
    while (progress < (guint64)num_of_bins-1){
        printf("%.2f%%\n", ((double)progress) / (num_of_bins-1) * 100);
        sleep(1);
        printf("\033[A\033[2K\r");
    }
    
    g_thread_pool_free (gthread_pool, FALSE, TRUE);
    
    // change hit count, now it is accumulated hit_count, change to real hit_count
    for (i=num_of_bins-1; i>=1; i--)
        result[i]->hit_count -= result[i-1]->hit_count;
    
    // clean up
    g_mutex_clear(&(params->mtx));
    g_free(params);
    // needs to free result later
    
    return result;
}








//
//int main(int argc, char* argv[]){
//# define CACHESIZE 2000
//# define BIN_SIZE 100
//    
//    int i;
//    printf("test_begin!\n");
//    
//    READER* reader = setup_reader(argv[1], 'v');
//    
////    struct_cache* cache = fifo_init(CACHESIZE, 'l', NULL);
////    struct_cache* cache = LRU_init(CACHESIZE, 'l', NULL);
//    
////    struct optimal_init_params* init_params = g_new0(struct optimal_init_params, 1);
////    init_params->reader = reader;
////    init_params->ts = 0;
////    struct_cache* cache = optimal_init(CACHESIZE, 'l', (void*)init_params);
//    
////    struct LRU_K_init_params *LRU_K_init_params = (struct LRU_K_init_params*) malloc(sizeof(struct LRU_K_init_params));
////    LRU_K_init_params->K = 1;
////    LRU_K_init_params->maxK = 1;
////    struct_cache* cache = LRU_K_init(CACHESIZE, 'v', LRU_K_init_params);
//    
////    struct LRU_LFU_init_params *init_params = g_new(struct LRU_LFU_init_params, 1);
////    init_params->LRU_percentage = 0.1;
////    struct_cache* cache = LRU_LFU_init(CACHESIZE, 'l', (void*)init_params);
//    
//    struct_cache *cache = LRU_dataAware_init(CACHESIZE, 'l', NULL);
//
//    
//    printf("after initialization, begin profiling\n");
////    gdouble* err_array = LRU_evict_err_statistics(reader, cache, 1000000);
//
////    for (i=0; i<reader->break_points->array->len-1; i++)
////        printf("%d: %lf\n", i, err_array[i]);
//    
//    return_res** res = profiler(reader, cache, 8, BIN_SIZE, 0, -1);
////    return_res** res = profiler(reader, cache, 1, BIN_SIZE, 23, 43);
//    printf("max i: %d\n", CACHESIZE/BIN_SIZE);
//    for (i=0; i<CACHESIZE/BIN_SIZE+1; i++){
//        printf("%d, %lld: %f\n", i, res[i]->cache_size, res[i]->hit_rate);
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
