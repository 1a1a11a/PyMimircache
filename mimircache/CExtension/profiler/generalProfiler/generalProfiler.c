

#include "generalProfiler.h" 


#include "reader.h"
#include "FIFO.h"
#include "Optimal.h"
#include "LRU_K.h"
#include "LRU.h"
#include "python_wrapper.h"
#include "LRU_dataAware.h"
#include "AMP.h"




/* this module is not reentrant-safe */


struct HR_PE* get_HR_PE(READER* reader_in, guint64 size){
    
    long i;
    int n = 22; 
    
    
    // initialization
    int num_of_threads = 8;
    
    // create the result storage area and caches of varying sizes
    struct HR_PE* hrpe = g_new0(struct HR_PE, 1);
   

    // build parameters and send to thread pool
    struct HR_PE_params* hrpe_params = g_new0(struct HR_PE_params, 1);
    hrpe_params->hrpe = hrpe;
    hrpe_params->reader = reader_in;
    hrpe_params->caches = g_new0(struct_cache*, n);
    
    hrpe_params->caches[0] = LRU_init(size, reader_in->data_type, NULL);

    struct AMP_init_params* AMP_initp = g_new0(struct AMP_init_params, 1);
    AMP_initp->APT = 4;
    AMP_initp->p_threshold = 256;
    AMP_initp->read_size = 1;
    hrpe_params->caches[1] = AMP_init(size, reader_in->data_type, AMP_initp);
    
    struct MIMIR_init_params** mimir_initp = g_new0(struct MIMIR_init_params*, n-2);
    for (i=0; i<n-2; i++){
        mimir_initp[i] = g_new0(struct MIMIR_init_params, 1);
        mimir_initp[i]->block_size = 64 * 1024;
        mimir_initp[i]->cache_type = "LRU";
        mimir_initp[i]->max_support = 20;
        mimir_initp[i]->min_support = 2;
        mimir_initp[i]->confidence = 0;
        mimir_initp[i]->item_set_size = 20;
        mimir_initp[i]->training_period = 0;
        mimir_initp[i]->prefetch_list_size = 2;
        mimir_initp[i]->max_metadata_size = 0.2;
        mimir_initp[i]->training_period_type = 'v';
        mimir_initp[i]->sequential_type = 0;
        mimir_initp[i]->sequential_K = -1;
        mimir_initp[i]->cycle_time = 2;
    }
    
    
    mimir_initp[0]->min_support = 1;
    mimir_initp[0]->max_support = 8;

    mimir_initp[1]->min_support = 2;
    mimir_initp[1]->max_support = 10;

    mimir_initp[2]->min_support = 3;
    mimir_initp[2]->max_support = 15;

    mimir_initp[3]->min_support = 4;
    mimir_initp[3]->max_support = 20;

    mimir_initp[4]->min_support = 5;
    mimir_initp[4]->max_support = 25;

    mimir_initp[5]->min_support = 6;
    mimir_initp[5]->max_support = 30;

    mimir_initp[6]->min_support = 7;
    mimir_initp[6]->max_support = 35;

    mimir_initp[7]->min_support = 8;
    mimir_initp[7]->max_support = 40;

    mimir_initp[8]->min_support = 1;
    mimir_initp[8]->max_support = 8;
    mimir_initp[8]->sequential_type = 1;
    mimir_initp[8]->sequential_K = 1;
    
    mimir_initp[9]->min_support = 2;
    mimir_initp[9]->max_support = 10;
    mimir_initp[9]->sequential_type = 1;
    mimir_initp[9]->sequential_K = 1;
    
    mimir_initp[10]->min_support = 3;
    mimir_initp[10]->max_support = 15;
    mimir_initp[10]->sequential_type = 1;
    mimir_initp[10]->sequential_K = 2;
    
    mimir_initp[11]->min_support = 4;
    mimir_initp[11]->max_support = 20;
    mimir_initp[11]->sequential_type = 1;
    mimir_initp[11]->sequential_K = 2;
    
    mimir_initp[12]->min_support = 6;
    mimir_initp[12]->max_support = 30;
    mimir_initp[12]->sequential_type = 1;
    mimir_initp[12]->sequential_K = 2;
    
    mimir_initp[13]->cache_type = "AMP";
    mimir_initp[13]->min_support = 1;
    mimir_initp[13]->max_support = 8;
    mimir_initp[13]->sequential_type = 2;

    mimir_initp[14]->cache_type = "AMP";
    mimir_initp[14]->min_support = 2;
    mimir_initp[14]->max_support = 10;
    mimir_initp[14]->sequential_type = 2;

    mimir_initp[15]->cache_type = "AMP";
    mimir_initp[15]->min_support = 3;
    mimir_initp[15]->max_support = 15;
    mimir_initp[15]->sequential_type = 2;

    mimir_initp[16]->cache_type = "AMP";
    mimir_initp[16]->min_support = 4;
    mimir_initp[16]->max_support = 20;
    mimir_initp[16]->sequential_type = 2;
    
    mimir_initp[17]->cache_type = "AMP";
    mimir_initp[17]->min_support = 5;
    mimir_initp[17]->max_support = 25;
    mimir_initp[17]->sequential_type = 2;

    mimir_initp[18]->cache_type = "AMP";
    mimir_initp[18]->min_support = 6;
    mimir_initp[18]->max_support = 30;
    mimir_initp[18]->sequential_type = 2;

    mimir_initp[19]->cache_type = "AMP";
    mimir_initp[19]->min_support = 8;
    mimir_initp[19]->max_support = 40;
    mimir_initp[19]->sequential_type = 2;
    
    
    
    for (i=0; i<n-2; i++)
        hrpe_params->caches[i+2] = MIMIR_init(size, reader_in->data_type, mimir_initp[i]);
    
    
    // build the thread pool
    GThreadPool * gthread_pool = g_thread_pool_new ( (GFunc) get_HR_PE_thread, (gpointer)hrpe_params, num_of_threads, TRUE, NULL);
    if (gthread_pool == NULL)
        g_error("cannot create thread pool in general profiler\n");
    
    
    for (i=1; i<n+1; i++){
        if ( g_thread_pool_push (gthread_pool, GUINT_TO_POINTER(i), NULL) == FALSE)
            g_error("cannot push data into thread in generalprofiler\n");
    }
    
    
    g_thread_pool_free (gthread_pool, FALSE, TRUE);
    
    
    // clean up
    LRU_destroy(hrpe_params->caches[0]);
    AMP_destroy(hrpe_params->caches[1]);
    for (i=0; i<n-2; i++)
        MIMIR_destroy(hrpe_params->caches[i+2]);
    
    g_free(hrpe_params);
    // needs to free result later
    return hrpe;
}

void get_HR_PE_thread(gpointer data, gpointer user_data){
    int order = (int) data - 1;
    struct HR_PE_params* hrpe_params = (struct HR_PE_params*)user_data;
    
    
    struct_cache* cache = (hrpe_params->caches[order]);
    struct HR_PE* hrpe = hrpe_params->hrpe;
    READER* reader = hrpe_params->reader;
    READER* reader_thread = copy_reader(reader);

    // create cache line struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = reader->data_type;

    
    
    guint64 hit_count=0, miss_count=0;
    gboolean (*add_element)(struct cache*, cache_line* cp);
    add_element = cache->core->add_element;
    
    read_one_element(reader_thread, cp);
    
    while (cp->valid){
        if (add_element(cache, cp)){
            hit_count ++;
        }
        else
            miss_count ++;
        read_one_element(reader_thread, cp);
    }
    
    
    hrpe->HR[order] = (double)hit_count/(hit_count+miss_count);
    hrpe->real_cache_size[order] = cache->core->size; 

    if (cache->core->type == e_mimir){
        gint64 prefetch = ((struct MIMIR_params*)(cache->cache_params))->num_of_prefetch_mimir;
        gint64 hit = ((struct MIMIR_params*)(cache->cache_params))->hit_on_prefetch_mimir;
        
        struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(cache->cache_params);
        hrpe->PE[order] = (double)hit/prefetch;
        hrpe->real_cache_size[order] = MIMIR_params->cache->core->size;
        
        
        if (MIMIR_params->sequential_type == 1){
            prefetch += ((struct MIMIR_params*)(cache->cache_params))->num_of_prefetch_sequential;
            hit += ((struct MIMIR_params*)(cache->cache_params))->hit_on_prefetch_sequential;
            hrpe->PE[order] = (double)hit/prefetch;
        }
        
        if (MIMIR_params->cache->core->type == e_AMP){
            prefetch += ((struct AMP_params*)(MIMIR_params->cache->cache_params))->num_of_prefetch;
            hit += ((struct AMP_params*)(MIMIR_params->cache->cache_params))->num_of_hit;
            hrpe->PE[order] = (double)hit/prefetch;
        }
        hrpe->prefetch[order] = prefetch;
    }
    if (cache->core->type == e_AMP){
        gint64 prefetch = ((struct AMP_params*)(cache->cache_params))->num_of_prefetch;
        gint64 hit = ((struct AMP_params*)(cache->cache_params))->num_of_hit;
        hrpe->PE[order] = (double)hit/prefetch;
        hrpe->prefetch[order] = prefetch;
    }
    
    // clean up
    g_free(cp);
    if (reader_thread->type != 'v')
        fclose(reader_thread->file);
    g_free(reader_thread);
//    cache->core->destroy_unique(cache);
}







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
    
    
    // create cache line struct and initialization
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
    
    result[order]->hit_count = (long long) hit_count;
    result[order]->miss_count = (long long) miss_count;
    result[order]->total_count = hit_count + miss_count;
    result[order]->hit_rate = (double) hit_count / (hit_count + miss_count);
    result[order]->miss_rate = 1 - result[order]->hit_rate;
    
    
    if (cache->core->type == e_mimir){ // || cache->core->type == e_MS1 || cache->core->type == e_MS2){
        struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(cache->cache_params);
        gint64 prefetch = MIMIR_params->num_of_prefetch_mimir;
        gint64 hit = MIMIR_params->hit_on_prefetch_mimir;
//        gint64 effective_hit = MIMIR_params->effective_hit_on_prefetch_mimir;
        
//        long counter = 0;
//        g_hash_table_foreach(MIMIR_params->prefetch_hashtable, prefetch_hashmap_count_length, &counter);

        printf("\ncache size %ld, real size: %ld, hit rate %lf, total check %lu, mimir prefetch %lu, hit %lu, accuracy: %lf, prefetch table size %u, ave len: %lf, evicted_prefetch %lu\n\n",
               cache->core->size, MIMIR_params->cache->core->size,
               (double)hit_count/(hit_count+miss_count),
               ((struct MIMIR_params*)(cache->cache_params))->num_of_check,
               prefetch, hit, (double)hit/prefetch,
               g_hash_table_size(MIMIR_params->prefetch_hashtable), 0.0, MIMIR_params->evicted_prefetch);
//               (double) counter / g_hash_table_size(MIMIR_params->prefetch_hashtable));
        
        if (MIMIR_params->sequential_type == 1){
            prefetch = ((struct MIMIR_params*)(cache->cache_params))->num_of_prefetch_sequential;
            hit = ((struct MIMIR_params*)(cache->cache_params))->hit_on_prefetch_sequential;
            printf("sequential prefetching, prefetch %lu, hit %lu, accuracy %lf\n", prefetch, hit, (double)hit/prefetch);
        }
        
        if (MIMIR_params->cache->core->type == e_AMP){
            prefetch = ((struct AMP_params*)(MIMIR_params->cache->cache_params))->num_of_prefetch;
            hit = ((struct AMP_params*)(MIMIR_params->cache->cache_params))->num_of_hit;
            printf("AMP cache size %ld, prefetch %lu, hit %lu, accuracy: %lf\n",
                   MIMIR_params->cache->core->size, prefetch, hit, (double)hit/prefetch); 

        }
    }
    if (cache->core->type == e_test1){
        gint64 prefech = ((struct test1_params*)(cache->cache_params))->num_of_prefetch;
        gint64 hit = ((struct test1_params*)(cache->cache_params))->hit_on_prefetch;
        
        printf("\ncache size %ld, hit rate %lf, total check %lu, prefetch %lu, hit %lu, accuracy: %lf\n",
               cache->core->size, (double)hit_count/(hit_count+miss_count),
               ((struct test1_params*)(cache->cache_params))->num_of_check,
               prefech, hit, (double)hit/prefech);
    }
    if (cache->core->type == e_AMP){
        gint64 prefech = ((struct AMP_params*)(cache->cache_params))->num_of_prefetch;
        gint64 hit = ((struct AMP_params*)(cache->cache_params))->num_of_hit;

        printf("\ncache size %ld, hit rate %lf, prefetch %lu, hit %lu, accuracy: %lf\n",
               cache->core->size, (double)hit_count/(hit_count+miss_count),
               prefech, hit, (double)hit/prefech);
    }

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
        
//        // FOR VSCSI AND FETCH BLOCK_NUM + 8
//        *(guint64*)(cp->item_p) = *(guint64*)(cp->item_p) + 8;
//        add_element(cache, cp);
//
        
        
        // begin prefetch elements
        GSList *slist = g_hash_table_lookup(prefetch_hashtable, cp->item_p);
        while (slist != NULL){
            if (cp->type == 'l')
                *((guint64*) cp->item_p) = *((guint64*)slist->data);
            else
                strcpy((char*)(cp->item_p), (char*)(slist->data));
            add_element(cache, cp);
            slist = slist->next;
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
    /** each line in the input file is tab seperated,
     *  the first is treated as key, the rest are values
     **/

    GHashTable *prefetch_hashtable;
    if (cache_in->core->data_type == 'l'){
        prefetch_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, simple_g_key_value_destroyer, g_slist_destroyer);
        FILE* file = fopen(prefetch_file_loc, "r");
        char buf[1024*1024];
        char *token;
        GSList* list = NULL;
        gint64 req;
        gint64 *key = NULL;
        while (fgets(buf, 1024*1024, file) != 0){
            list = NULL;
            key = NULL;
//            printf("line: %s\n", buf);
            token = strtok(buf, "\t");
            while (token!=NULL){
                req = (gint64)(atol(token));
//                printf("parsed: %ld\t", req);
//            sscanf(token, "%lu", &k);
                if (key == NULL){
                    key = (gpointer)g_new(gint64, 1);
                    *key = req;
                }
                else{
                    gint64* value = (gpointer)g_new(gint64, 1);
                    *value = req;

                    list = g_slist_prepend(list, (gpointer)value);
                }
                token = strtok(NULL, "\t");
            }
//            printf("\n");
            g_hash_table_insert(prefetch_hashtable, key, list);
        }
        
    }
    else{
        prefetch_hashtable = g_hash_table_new_full(g_str_hash, g_str_equal, simple_g_key_value_destroyer, g_slist_destroyer);

        
        FILE* file = fopen(prefetch_file_loc, "r");
        char buf[1024*1024];
        char *token;
        GSList* list = NULL;
//        gint64 req;
        gchar *key = NULL, *value = NULL;
        while (fgets(buf, 1024*1024, file) != 0){
            list = NULL;
            key = NULL;
            token = strtok(buf, "\t");
            while (token!=NULL){
                if (key == NULL){
                    key = g_strdup(token);
                }
                else{
                    value = g_strdup(token);
                    list = g_slist_prepend(list, (gpointer)value);
                }
                token = strtok(NULL, "\t");
            }
            g_hash_table_insert(prefetch_hashtable, key, list);
        }

    }
    

    
    
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
    GThreadPool * gthread_pool = g_thread_pool_new ( (GFunc) profiler_with_prefetch_thread, (gpointer)params, num_of_threads, TRUE, NULL);
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



