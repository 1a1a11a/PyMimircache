//
//  partition.c
//  mimircache 
//
//  Created by Juncheng on 11/19/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "generalProfiler.h" 
#include "partition.h"



#ifdef __cplusplus
extern "C"
{
#endif


/************************* Util functions **************************/

partition_t* init_partition_t(uint8_t n_partitions, uint64_t cache_size){
    partition_t *partitions         =   g_new0(partition_t, 1);
    partitions->cache_size          =   cache_size;
    partitions->n_partitions        =   n_partitions;
    partitions->jump_over_count     =   0;
    partitions->partition_history   =   g_new(GArray*, n_partitions);
    partitions->current_partition   =   g_new0(uint64_t, n_partitions);
    int i;
    for (i=0; i<n_partitions; i++)
        partitions->partition_history[i] = g_array_new(FALSE, FALSE, sizeof(double));
    return partitions;
}

void free_partition_t(partition_t *partition){
    int i;
    for (i=0; i<partition->n_partitions; i++){
        g_array_free(partition->partition_history[i], TRUE);
    }
    g_free(partition->current_partition);
    g_free(partition->partition_history);
    g_free(partition);
}

static void
printHashTable (gpointer key, gpointer value, gpointer user_data){
    printf("key %s, value %p\n", (char*)key, value);
}


/*********************** partition related function *************************/
partition_t* get_partition(reader_t* reader, struct cache* cache, uint8_t n_partitions){
    /** this function currently only works under my mixed trace, 
     which is either plainText or csv, and each label has the first letter indicating its source 
     **/
    
    partition_t *partitions = init_partition_t(n_partitions, cache->core->size);
    
    // create cache line struct and initialization
    cache_line* cp      =       new_cacheline();
    cp->type            =       reader->base->data_type;
    
 
    gboolean  (*check_element) (struct cache*, cache_line*)     =   cache->core->check_element;
    
    // newly added 0912, may not work for all cache
    void      (*insert_element)(struct cache*, cache_line*)     =   cache->core->__insert_element;
    void      (*update_element)(struct cache*, cache_line*)     =   cache->core->__update_element;
    gpointer  (*evict_with_return)(struct cache*, cache_line*)  =   cache->core->__evict_with_return;
    gint64    (*get_size)      (struct cache*)                  =   cache->core->get_size;

    
    char* key;
    int i;
    guint64 current_size, cache_size;
    double percent;
    uint64_t counter = 0;
    
    struct optimal_params* optimal_params = NULL;
    if (cache->core->type == e_Optimal)
        optimal_params = (struct optimal_params*)(cache->cache_params);

    
    cache_size = cache->core->size;
    
    read_one_element(reader, cp);
    counter ++;
    
    while (cp->valid){
        if (check_element(cache, cp))
            update_element(cache, cp);
        else{
            insert_element(cache, cp);
            partitions->current_partition[(cp->item)[0]-'A'] ++;
        }
        current_size = get_size(cache);
        if (current_size > cache_size){
            key = evict_with_return(cache, cp);
            partitions->current_partition[key[0]-'A'] --;
            g_free(key);
        }
        if (current_size >= cache_size){
            for (i=0; i<n_partitions; i++){
                percent = (double) (partitions->current_partition[i])/cache_size;
                g_array_append_val(partitions->partition_history[i], percent);
            }
        }
//        printf("cp->ts %ld, current %s\n", cp->ts, cp->item_p);
//        g_hash_table_foreach( ((struct optimal_params*)cache->cache_params)->hashtable, printHashTable, NULL);
        SUPPRESS_FUNCTION_NO_USE_WARNING(printHashTable); 
        
        if (partitions->partition_history[0]->len == 1)
            partitions->jump_over_count = counter;
        
        if (get_size(cache) > (gint64) cache_size)
            fprintf(stderr, "ERROR current size %lu, given size %ld\n", (unsigned long)get_size(cache), cache_size);

        read_one_element(reader, cp);
        counter ++;
        if (cache->core->type == e_Optimal)
            optimal_params->ts ++;
    }

    
    reset_reader(reader);
    destroy_cacheline(cp);
    return partitions;
}


























































static void profiler_partition_thread(gpointer data, gpointer user_data){
    struct multithreading_params_generalProfiler* params = (struct multithreading_params_generalProfiler*) user_data;
    
    int order = GPOINTER_TO_UINT(data);
    guint bin_size = params->bin_size;
    
    return_res** result = params->result;
    reader_t* reader_thread = clone_reader(params->reader);
    
    
    // create cache line struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = params->cache->core->data_type;
    
    
    /*************************** THIS IS NOT GENERAL (BEGIN) **********************/
    int i;
    int n_partition = 2;
    struct optimal_init_params* init_params = g_new0(struct optimal_init_params, 1);
    init_params->reader = reader_thread;
    init_params->ts = 0;
    struct_cache* optimal = optimal_init(bin_size*order, reader_thread->base->data_type, 0, (void*)init_params);

    struct_cache** cache = g_new0(struct_cache*, n_partition);
    partition_t* partition = get_partition(reader_thread, optimal, n_partition);
    for (i=0; i<n_partition; i++){
        partition->current_partition[i] = 0;
        cache[i] = params->cache->core->cache_init(
                            (long)(g_array_index(partition->partition_history[i], double, 0) * bin_size * order),
                            params->cache->core->data_type,
                            params->cache->core->block_unit_size,
                            params->cache->core->cache_init_params);
    }

    optimal->core->destroy(optimal);
    
    
    
    /*************************** THIS IS NOT GENERAL (END) ************************/

    
    uint64_t hit_count=0, miss_count=0;

    gboolean  (*check_element) (struct cache*, cache_line*)     =   cache[0]->core->check_element;
    
    // newly added 0912, may not work for all cache
    void      (*insert_element)(struct cache*, cache_line*)     =   cache[0]->core->__insert_element;
    void      (*update_element)(struct cache*, cache_line*)     =   cache[0]->core->__update_element;
    void      (*evict_element) (struct cache*, cache_line*)     =   cache[0]->core->__evict_element; 
//    gpointer  (*evict_with_return)(struct cache*, cache_line*)  =   cache[0]->core->__evict_with_return;
    gint64    (*get_size)      (struct cache*)                  =   cache[0]->core->get_size;
    
    read_one_element(reader_thread, cp);
    
    while (cp->valid){
        if (check_element(cache[(cp->item)[0]-'A'], cp)){
            update_element(cache[(cp->item)[0]-'A'], cp);
            hit_count++;
        }
        else{
            insert_element(cache[(cp->item)[0]-'A'], cp);
            partition->current_partition[(cp->item)[0]-'A'] ++;
            miss_count++;
        }
        
        
        // now adjust each partition cache size
        if ((gint64)(hit_count + miss_count - partition->jump_over_count) >= 0){
            for(i=0; i<n_partition; i++)
                if ((gint64)(hit_count + miss_count - partition->jump_over_count)
                                        >= partition->partition_history[i]->len)
                    printf("ERROR size over, jump %lu, current %lu, len %u\n",
                           (unsigned long)partition->jump_over_count,
                           (unsigned long)(hit_count + miss_count - partition->jump_over_count),
                           partition->partition_history[i]->len);
                else
                    // the reason of 0.5 + is because default cast is to truncate
                    cache[i]->core->size = (long)(0.5 + bin_size * order *
                                                  g_array_index(partition->partition_history[i], double,
                                                    hit_count + miss_count - partition->jump_over_count));
        }
        
        
        // evict after adjustment
        for(i=0; i<n_partition; i++){
            // can't use while here, because ?
            while ((long)get_size(cache[i]) > cache[i]->core->size){
                evict_element(cache[i], cp);
                partition->current_partition[i] --;
            }
        }

#ifdef SANITY_CHECK
        // sanity check
        for(i=0; i<n_partition; i++)
            if (get_size(cache[i]) != (long) partition->current_partition[i])
                fprintf(stderr, "Sanity check failed, i %d, ERROR size %lu, "
                        "partition %lu\n", i, (unsigned long)get_size(cache[i]),
                        (unsigned long)partition->current_partition[i]);
        
        if (hit_count + miss_count == partition->jump_over_count){
            for(i=0; i<n_partition; i++)
                if (get_size(cache[i]) != cache[i]->core->size &&
                    get_size(cache[i]) != cache[i]->core->size - 1){
                    // the second condition is possible because of the resize
                    fprintf(stderr, "Sanity check failed, cache size full, "
                            "but partition not consistent %d %d\n",
                            get_size(cache[i]) != cache[i]->core->size,
                            (long) partition->current_partition[i] != (long) cache[i]->core->size);
                    fprintf(stderr, "i %d, get size %lu, given size %ld, partition %lu\n",
                            i, (unsigned long)get_size(cache[i]), cache[i]->core->size,
                            (unsigned long)partition->current_partition[i]);
                }
        }
#endif

        
        
        read_one_element(reader_thread, cp);
        if (cache[0]->core->type == e_Optimal)
            for(i=0; i<n_partition; i++)
                ((struct optimal_params*)(cache[i]->cache_params))->ts ++;
    }
    
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
    
    
    close_reader_unique(reader_thread); 
//    if (reader_thread->type != 'v')
//        fclose(reader_thread->file);
//    if (reader_thread->type == 'c'){
//        csv_free(reader_thread->csv_parser);
//        g_free(reader_thread->csv_parser);
//    }
//    g_free(reader_thread);
    
    for (i=0; i<n_partition; i++)
        cache[i]->core->destroy_unique(cache[i]);
    free_partition_t(partition);
}



return_res** profiler_partition(reader_t* reader_in, struct_cache* cache_in, int num_of_threads_in, int bin_size_in){
    /**
     if profiling from the very beginning, then set begin_pos=0,
     if porfiling till the end of trace, then set end_pos=-1 or the length of trace+1;
     return results do not include size 0
     **/
    
    long i;
    guint64 progress = 0;
    
    
    // initialization
    int num_of_threads = num_of_threads_in;
    int bin_size = bin_size_in;
    
    long num_of_bins = ceil((double) cache_in->core->size/bin_size)+1;
    
    
    // create the result storage area and caches of varying sizes
    return_res** result = g_new(return_res*, num_of_bins);
    
    for (i=0; i<num_of_bins; i++){
        result[i] = g_new0(return_res, 1);
        result[i]->cache_size = bin_size * (i);
    }
    result[0]->miss_rate = 1;
    
    
    // build parameters and send to thread pool
    mt_param_gp_t* params = g_new0(mt_param_gp_t, 1);
    params->reader = reader_in;
    params->cache = cache_in;
    params->result = result;
    params->bin_size = (guint) bin_size;
    params->begin_pos = 0;
    params->end_pos = -1;
    params->progress = &progress;
    g_mutex_init(&(params->mtx));
    
    // build the thread pool
    GThreadPool * gthread_pool = g_thread_pool_new ((GFunc) profiler_partition_thread,
                                                    (gpointer)params,
                                                    num_of_threads,
                                                    TRUE, NULL);
    if (gthread_pool == NULL)
        ERROR("cannot create thread pool in general profiler\n");
    
    
    for (i=1; i<num_of_bins; i++){
        if ( g_thread_pool_push (gthread_pool, GUINT_TO_POINTER(i), NULL) == FALSE)
            ERROR("cannot push data into thread in generalprofiler\n");
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




#ifdef __cplusplus
}
#endif