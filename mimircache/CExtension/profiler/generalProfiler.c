//
//  generalProfiler.c
//  generalProfiler
//
//  Created by Juncheng on 5/24/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//



#include "generalProfiler.h" 


#include "reader.h"
#include "FIFO.h"
#include "Optimal.h"
#include "LRU_K.h"
#include "LRU.h"
#include "cacheHeader.h"
#include "AMP.h" 
#include "PG.h"
//#include "SLRUML.h"


#ifdef __cplusplus
extern "C"
{
#endif


static void profiler_thread_temp(gpointer data, gpointer user_data){
    SUPPRESS_FUNCTION_NO_USE_WARNING(profiler_thread_temp); 
    struct multithreading_params_generalProfiler* params = (struct multithreading_params_generalProfiler*) user_data;
    
    int order = GPOINTER_TO_UINT(data);
    gint64 pos = 0;
    guint bin_size = params->bin_size;
    
    struct_cache* cache = params->cache->core->cache_init(bin_size * order,
                                                          params->cache->core->data_type,
                                                          params->cache->core->block_unit_size,
                                                          params->cache->core->cache_init_params);
    return_res** result = params->result;
    
    reader_t* reader_thread = clone_reader(params->reader);
    
    
    // create cache line struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = params->cache->core->data_type;
    cp->block_unit_size = (size_t) reader_thread->base->block_unit_size;    // this is not used, block_unit_size goes with cache
    cp->disk_sector_size = (size_t) reader_thread->base->disk_sector_size;
    
    
    guint64 hit_count=0, miss_count=0;
    gboolean (*add_element)(struct cache*, cache_line* cp);
    add_element = cache->core->add_element;
    
    read_one_element(reader_thread, cp);
    
    // this must happen after read, otherwise cp size and type unknown
    if (cache->core->consider_size && cp->type == 'l' && cp->disk_sector_size != 0){     // && cp->size != 0 is removed due to trace dirty
        add_element = cache->core->add_element_withsize;
        if (add_element == NULL){
            ERROR("using size with profiling, cannot find add_element_withsize\n");
            abort();
        }
    }
    
    char output_filename[128];
    char* dat_name = strrchr(reader_thread->base->file_loc, '/');
    sprintf(output_filename, "/home/jason/special/%s_size%ld", dat_name, cache->core->size);
    FILE* miss_req_output = fopen(output_filename, "w");
//    printf("this is the new one\n"); 
    
    
    while (cp->valid){
        if (add_element(cache, cp))
            hit_count ++;
        else{
            miss_count ++;
            fprintf(miss_req_output, "%ld, %s\n", cp->real_time, cp->item);
        }
        pos++;
        read_one_element(reader_thread, cp);
    }
    
    result[order]->hit_count = (long long) hit_count;
    result[order]->miss_count = (long long) miss_count;
    result[order]->total_count = hit_count + miss_count;
    result[order]->hit_rate = (double) hit_count / (hit_count + miss_count);
    result[order]->miss_rate = 1 - result[order]->hit_rate;
    
    fclose(miss_req_output);

    // clean up
    g_mutex_lock(&(params->mtx));
    (*(params->progress)) ++ ;
    g_mutex_unlock(&(params->mtx));
    
    g_free(cp);
    close_reader_unique(reader_thread);
    cache->core->destroy_unique(cache);
}










/********************************************************************* 
 **  printHashTable
 **  used for output associations found in Mithril
 **
 **
 *********************************************************************/


static void
printHashTable (gpointer key, gpointer value, gpointer user_data){
    // only works for long data type 
    struct MIMIR_params* MIMIR_params = user_data;
    gint prefetch_table_index = GPOINTER_TO_INT(value);
    gint dim1 = (gint)floor(prefetch_table_index/(double)PREFETCH_TABLE_SHARD_SIZE);
    gint dim2 = prefetch_table_index % PREFETCH_TABLE_SHARD_SIZE * (MIMIR_params->prefetch_list_size+1);
    
    int i;
    for (i=0; i<3; i++){
        if (MIMIR_params->prefetch_table_array[dim1][dim2] != *(gint64*)key){
            fprintf(stderr, "ERROR prefetch table pos wrong %ld %ld, dim %d %d\n",
                *(gint64*)key, MIMIR_params->prefetch_table_array[dim1][dim2], dim1, dim2);
            exit(1);
        }
        printf("%ld, ", MIMIR_params->prefetch_table_array[dim1][dim2+i]);
    }

    printf("\n");
}

struct HR_PE* get_HR_PE(reader_t* reader_in, guint64 size){
    
    int i;
    int AMP_n = 1, mimir_n1 = 8, mimir_n2 = 8, PG_n = 1;
    int n = AMP_n + mimir_n1 + mimir_n2 + + PG_n + 1;
    int block_unit_size = 16 * 1024; 
    
    
    // initialization
    int num_of_threads = 4;
    
    // create the result storage area and caches of varying sizes
    struct HR_PE* hrpe = g_new0(struct HR_PE, 1);
   

    // build parameters and send to thread pool
    struct HR_PE_params* hrpe_params = g_new0(struct HR_PE_params, 1);
    hrpe_params->hrpe = hrpe;
    hrpe_params->reader = reader_in;
    hrpe_params->caches = g_new0(struct_cache*, n);
    
    hrpe_params->caches[0] = LRU_init(size, reader_in->base->data_type, block_unit_size, NULL);
    
    
    struct AMP_init_params** AMP_initp = g_new0(struct AMP_init_params*, AMP_n);
    for (i=0; i<AMP_n; i++){
        AMP_initp[i] = g_new0(struct AMP_init_params, 1);
        AMP_initp[i]->APT = 4;
        AMP_initp[i]->read_size = 1;
        AMP_initp[i]->K = i+1;
        AMP_initp[i]->p_threshold = (int)(256/pow(2, i));
        hrpe_params->caches[i+1] = AMP_init(size, reader_in->base->data_type, block_unit_size, AMP_initp[i]);
    }

    PG_init_params_t *PG_initp = g_new0(PG_init_params_t, 1);
    PG_initp->block_size = block_unit_size;
    PG_initp->cache_type = "LRU";
    PG_initp->lookahead = 20;
    PG_initp->max_meta_data = 0.1;
    PG_initp->prefetch_threshold = 0.05;
    
    hrpe_params->caches[1+AMP_n] = PG_init(size, reader_in->base->data_type, block_unit_size, PG_initp);

    
    struct MIMIR_init_params** mimir_initp = g_new0(struct MIMIR_init_params*, mimir_n1 + mimir_n2);
    for (i=0; i<mimir_n1+mimir_n2; i++){
        mimir_initp[i] = g_new0(struct MIMIR_init_params, 1);
        mimir_initp[i]->block_size = block_unit_size;
        mimir_initp[i]->cache_type = "LRU";
        mimir_initp[i]->output_statistics = 1;
        mimir_initp[i]->confidence = 0;
        mimir_initp[i]->item_set_size = 20;
        mimir_initp[i]->prefetch_list_size = 2;
        mimir_initp[i]->max_metadata_size = 0.10;
        mimir_initp[i]->sequential_type = 0; 
        mimir_initp[i]->sequential_K = 0;
        mimir_initp[i]->cycle_time = 2;
        mimir_initp[i]->AMP_pthreshold = 256;
        mimir_initp[i]->mining_threshold = 5120;    // this is the default value used in the past for results
    }
    
    
    mimir_initp[0]->min_support = 1;
    mimir_initp[0]->max_support = 4;

    mimir_initp[1]->min_support = 2;
    mimir_initp[1]->max_support = 8;

    mimir_initp[2]->min_support = 3;
    mimir_initp[2]->max_support = 12;

    mimir_initp[3]->min_support = 4;
    mimir_initp[3]->max_support = 16;

    mimir_initp[4]->min_support = 5;
    mimir_initp[4]->max_support = 20;

    mimir_initp[5]->min_support = 6;
    mimir_initp[5]->max_support = 20;

    mimir_initp[6]->min_support = 7;
    mimir_initp[6]->max_support = 20;

    mimir_initp[7]->min_support = 8;
    mimir_initp[7]->max_support = 20;


    
    mimir_initp[8]->min_support = 1;
    mimir_initp[8]->max_support = 4;
    mimir_initp[8]->cache_type = "AMP";
    mimir_initp[8]->sequential_type = 2;
    mimir_initp[8]->sequential_K = 1;

    mimir_initp[9]->min_support = 2;
    mimir_initp[9]->max_support = 8;
    mimir_initp[9]->cache_type = "AMP";
    mimir_initp[9]->sequential_type = 2;
    mimir_initp[9]->sequential_K = 1;

    mimir_initp[10]->min_support = 3;
    mimir_initp[10]->max_support = 12;
    mimir_initp[10]->cache_type = "AMP";
    mimir_initp[10]->sequential_type = 2;
    mimir_initp[10]->sequential_K = 1;

    mimir_initp[11]->min_support = 4;
    mimir_initp[11]->max_support = 16;
    mimir_initp[11]->cache_type = "AMP";
    mimir_initp[11]->sequential_type = 2;
    mimir_initp[11]->sequential_K = 1;

    mimir_initp[12]->min_support = 5;
    mimir_initp[12]->max_support = 20;
    mimir_initp[12]->cache_type = "AMP";
    mimir_initp[12]->sequential_type = 2;
    mimir_initp[12]->sequential_K = 1;

    mimir_initp[13]->min_support = 6;
    mimir_initp[13]->max_support = 20;
    mimir_initp[13]->cache_type = "AMP";
    mimir_initp[13]->sequential_type = 2;
    mimir_initp[13]->sequential_K = 1;

    mimir_initp[14]->min_support = 7;
    mimir_initp[14]->max_support = 20;
    mimir_initp[14]->cache_type = "AMP";
    mimir_initp[14]->sequential_type = 2;
    mimir_initp[14]->sequential_K = 1;

    mimir_initp[15]->min_support = 8;
    mimir_initp[15]->max_support = 20;
    mimir_initp[15]->cache_type = "AMP";
    mimir_initp[15]->sequential_type = 2;
    mimir_initp[15]->sequential_K = 1;

   
    /**
    mimir_initp[10]->min_support = 1;
    mimir_initp[10]->max_support = 8;
    mimir_initp[10]->cache_type = "AMP";
    mimir_initp[10]->sequential_type = 2;
    mimir_initp[10]->sequential_K = 1;
    mimir_initp[10]->AMP_pthreshold = 256;
    
    mimir_initp[11]->min_support = 2;
    mimir_initp[11]->max_support = 8;
    mimir_initp[11]->cache_type = "AMP";
    mimir_initp[11]->sequential_type = 2;
    mimir_initp[11]->sequential_K = 1;
    mimir_initp[11]->AMP_pthreshold = 256;

    mimir_initp[12]->min_support = 3;
    mimir_initp[12]->max_support = 12;
    mimir_initp[12]->cache_type = "AMP";
    mimir_initp[12]->sequential_type = 2;
    mimir_initp[12]->sequential_K = 1;
    mimir_initp[12]->AMP_pthreshold = 256;
     

    **/
    
    
    
    for (i=0; i<mimir_n1+mimir_n2; i++)
        hrpe_params->caches[i+AMP_n+PG_n+1] = MIMIR_init(size, reader_in->base->data_type, block_unit_size, mimir_initp[i]);
    
    
    // build the thread pool
    GThreadPool * gthread_pool = g_thread_pool_new ( (GFunc) get_HR_PE_thread, (gpointer)hrpe_params, num_of_threads, TRUE, NULL);
    if (gthread_pool == NULL)
        ERROR("cannot create thread pool in general profiler\n");
    
    
    for (i=1; i<n+1; i++){  // i cannot begin with 0
        if ( g_thread_pool_push (gthread_pool, GUINT_TO_POINTER(i), NULL) == FALSE)
            ERROR("cannot push data into thread in generalprofiler\n");
    }
    
    
    g_thread_pool_free (gthread_pool, FALSE, TRUE);
    

    
    
    // clean up
    LRU_destroy(hrpe_params->caches[0]);
    for (i=0; i<AMP_n; i++)
        AMP_destroy(hrpe_params->caches[1+i]);
    for (i=0; i<PG_n; i++)
        PG_destroy(hrpe_params->caches[i+AMP_n+1]);
    for (i=0; i<mimir_n1+mimir_n2; i++)
        MIMIR_destroy(hrpe_params->caches[i+1+AMP_n+PG_n]);
    
    g_free(hrpe_params);
    // needs to free result later
    return hrpe;
}

void get_HR_PE_thread(gpointer data, gpointer user_data){
    int order = GPOINTER_TO_INT(data) - 1;
    struct HR_PE_params* hrpe_params = (struct HR_PE_params*)user_data;
    
    
    struct_cache* cache = (hrpe_params->caches[order]);
    struct HR_PE* hrpe = hrpe_params->hrpe;
    reader_t* reader = hrpe_params->reader;
    reader_t* reader_thread = clone_reader(reader);

    // create cache line struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = reader->base->data_type;
    cp->block_unit_size = (size_t) reader->base->block_unit_size;
    cp->disk_sector_size = (size_t) reader->base->disk_sector_size;
    
    
    guint64 hit_count=0, miss_count=0;
    gboolean (*add_element)(struct cache*, cache_line* cp);
    add_element = cache->core->add_element;
    
    read_one_element(reader_thread, cp);
    // new 
    if (cache->core->consider_size && cp->type == 'l' && cp->disk_sector_size != 0){     // && cp->size != 0 is removed due to trace dirty
        add_element = cache->core->add_element_withsize;
        if (add_element == NULL){
            ERROR("profiling with size, but does not have add_element_withsize\n");
            abort();
        }
    }
    
    while (cp->valid){
        /* new 170428
         add size into consideration, this only affects the traces with cache_size column,
         currently CPHY traces are affected
         the default size for each block is 512 bytes */
        if (add_element(cache, cp))
            hit_count ++;
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
    
    if (cache->core->type == e_PG){
        gint64 prefetch = ((PG_params_t*)(cache->cache_params))->num_of_prefetch;
        gint64 hit = ((PG_params_t*)(cache->cache_params))->num_of_hit;
        hrpe->PE[order] = (double)hit/prefetch;
        hrpe->prefetch[order] = prefetch;
        hrpe->real_cache_size[order] = ((PG_params_t*)(cache->cache_params))->cache->core->size; 
    }
    
    
    
    // clean up
    g_free(cp);
    close_reader_unique(reader_thread);
//    if (reader_thread->base->type != 'v')
//        fclose(reader_thread->base->file);
//    g_free(reader_thread);
//    cache->core->destroy_unique(cache);
}







static void profiler_thread(gpointer data, gpointer user_data){
    struct multithreading_params_generalProfiler* params = (struct multithreading_params_generalProfiler*) user_data;

    int order = GPOINTER_TO_UINT(data);
    gint64 begin_pos = params->begin_pos;
    gint64 end_pos = params->end_pos;
    gint64 pos = begin_pos;
    guint bin_size = params->bin_size;
    
    struct_cache* cache = params->cache->core->cache_init(bin_size * order,
                                                          params->cache->core->data_type,
                                                          params->cache->core->block_unit_size,
                                                          params->cache->core->cache_init_params);        
    return_res** result = params->result;
        
    reader_t* reader_thread = clone_reader(params->reader);
    
    if (begin_pos!=0 && begin_pos!=-1)
        skip_N_elements(reader_thread, begin_pos);
    
    
    // create cache line struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = params->cache->core->data_type;
    cp->block_unit_size = (size_t) reader_thread->base->block_unit_size;    // this is not used, block_unit_size goes with cache 
    cp->disk_sector_size = (size_t) reader_thread->base->disk_sector_size;
    
    
    guint64 hit_count=0, miss_count=0;
    gboolean (*add_element)(struct cache*, cache_line* cp);
    add_element = cache->core->add_element;

    read_one_element(reader_thread, cp);

    // this must happen after read, otherwise cp size and type unknown
    if (cache->core->consider_size && cp->type == 'l' && cp->disk_sector_size != 0){     // && cp->size != 0 is removed due to trace dirty
        add_element = cache->core->add_element_withsize;
        if (add_element == NULL){
            ERROR("using size with profiling, cannot find add_element_withsize\n");
            abort();
        }
    }

    while (cp->valid && pos<end_pos){
        /* new 170428
         add size into consideration, this only affects the traces with cache_size column, 
         currently CPHY traces are affected 
         the default size for each block is 512 bytes */
        if (add_element(cache, cp))
            hit_count ++;
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

    
    if (cache->core->type == e_mimir){ 
        struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(cache->cache_params);
        gint64 prefetch = MIMIR_params->num_of_prefetch_mimir;
        gint64 hit = MIMIR_params->hit_on_prefetch_mimir;

        printf("\ncache size %ld, real size: %ld, hit rate %lf, total check %lu, "
               "mimir prefetch %lu, hit %lu, accuracy: %lf, prefetch table size %u\n",
               cache->core->size, MIMIR_params->cache->core->size,
               (double)hit_count/(hit_count+miss_count),
               ((struct MIMIR_params*)(cache->cache_params))->num_of_check,
               prefetch, hit, (double)hit/prefetch,
               g_hash_table_size(MIMIR_params->prefetch_hashtable));
        
        if (MIMIR_params->sequential_type == 1){
            gint64 prefetch2 = ((struct MIMIR_params*)(cache->cache_params))->num_of_prefetch_sequential;
            gint64 hit2 = ((struct MIMIR_params*)(cache->cache_params))->hit_on_prefetch_sequential;
            printf("sequential prefetching, prefetch %lu, hit %lu, accuracy %lf\n", prefetch2, hit2, (double)hit2/prefetch2);
            printf("overall size %ld, hit rate %lf, efficiency %lf\n", MIMIR_params->cache->core->size,
                   (double)hit_count/(hit_count+miss_count), (double)(hit+hit2)/(prefetch+prefetch2));
        }
        
        if (MIMIR_params->cache->core->type == e_AMP){
            gint64 prefetch2 = ((struct AMP_params*)(MIMIR_params->cache->cache_params))->num_of_prefetch;
            gint64 hit2 = ((struct AMP_params*)(MIMIR_params->cache->cache_params))->num_of_hit;
            printf("Mithril_AMP cache size %ld, prefetch %lu, hit %lu, accuracy: %lf, total prefetch %lu, hit %lu, accuracy: %lf\n",
                   MIMIR_params->cache->core->size, prefetch2, hit2, (double)hit2/prefetch2,
                   prefetch+prefetch2, hit+hit2, (double)(hit+hit2)/(prefetch+prefetch2)); 

        }
        
        // output association
//        printf("output association\n");
//        g_hash_table_foreach (MIMIR_params->prefetch_hashtable,
//                              printHashTable, MIMIR_params);
        SUPPRESS_FUNCTION_NO_USE_WARNING(printHashTable);
    }
    
    if (cache->core->type == e_PG){
        PG_params_t *PG_params = (PG_params_t*)(cache->cache_params);
        printf("\n PG cache size %lu, real size %ld, hit rate %lf, prefetch %lu, "
               "hit %lu, precision %lf\n", (unsigned long)PG_params->init_size,
               PG_params->cache->core->size,
               (double)hit_count/(hit_count+miss_count),
               PG_params->num_of_prefetch, PG_params->num_of_hit,
               (double)(PG_params->num_of_hit)/(PG_params->num_of_prefetch));
    }

    if (cache->core->type == e_AMP){
        gint64 prefech = ((struct AMP_params*)(cache->cache_params))->num_of_prefetch;
        gint64 hit = ((struct AMP_params*)(cache->cache_params))->num_of_hit;

        printf("\nAMP cache size %ld, hit rate %lf, prefetch %lu, hit %lu, accuracy: %lf\n\n",
               cache->core->size, (double)hit_count/(hit_count+miss_count),
               prefech, hit, (double)hit/prefech);
    }
    

    // clean up
    g_mutex_lock(&(params->mtx));
    (*(params->progress)) ++ ;
    g_mutex_unlock(&(params->mtx));

    g_free(cp);
    close_reader_unique(reader_thread);
    cache->core->destroy_unique(cache);
}

    
    
return_res** profiler(reader_t* reader_in,
                      struct_cache* cache_in,
                      int num_of_threads_in,
                      int bin_size_in,
                      gint64 begin_pos,
                      gint64 end_pos){
    /**
     if profiling from the very beginning, then set begin_pos=0, 
     if porfiling till the end of trace, then set end_pos=-1 or the length of trace+1; 
     return results do not include size 0 
     **/
    
    long i;
    guint64 progress = 0;
    if (end_pos<=begin_pos && end_pos!=-1){
        ERROR("end pos <= beigin pos in general profiler, please check\n");
        exit(1);
    }

    
    // initialization
    int num_of_threads = num_of_threads_in;
    int bin_size = bin_size_in;
    
    long num_of_bins = ceil((double) cache_in->core->size/bin_size)+1;
    
    if (end_pos==-1){
        if (reader_in->base->total_num == -1)
            get_num_of_cache_lines(reader_in);
        end_pos = reader_in->base->total_num;
    }
    
    
    // check whether profiling considering size or not
    if (cache_in->core->consider_size && reader_in->base->data_type == 'l'
            && reader_in->base->disk_sector_size != 0 &&
        cache_in->core->block_unit_size != 0){     // && cp->size != 0 is removed due to trace dirty
        INFO("use block size %d, disk sector size %d in profiling\n",
             cache_in->core->block_unit_size,
             reader_in->base->disk_sector_size);
    }
    
    
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
    // GThreadPool * gthread_pool = g_thread_pool_new ( (GFunc) profiler_thread_temp, (gpointer)params, num_of_threads, TRUE, NULL);
    if (gthread_pool == NULL)
        ERROR("cannot create thread pool in general profiler\n");
    
    
    for (i=1; i<num_of_bins; i++){
        if ( g_thread_pool_push (gthread_pool, GUINT_TO_POINTER(i), NULL) == FALSE)
            ERROR("cannot push data into thread in generalprofiler\n");
    }
    
    while (progress < (guint64)num_of_bins-1){
//            fprintf(stderr, "%.2f%%\n", ((double)progress) / (num_of_bins-1) * 100);
            sleep(1);
//            fprintf(stderr, "\033[A\033[2K\r");
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


static void traverse_trace(reader_t* reader, struct_cache* cache){
    
    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = cache->core->data_type;
    cp->block_unit_size = (size_t) reader->base->block_unit_size;
    
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


static void get_evict_err(reader_t* reader, struct_cache* cache){
    
    cache->core->bp_pos = 1;
    cache->core->evict_err_array = g_new0(gdouble, reader->sdata->break_points->array->len-1);

    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = cache->core->data_type;
    cp->block_unit_size = (size_t) reader->base->block_unit_size;
    
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





gdouble* LRU_evict_err_statistics(reader_t* reader_in, struct_cache* cache_in, guint64 time_interval){
    
    gen_breakpoints_realtime(reader_in, time_interval, -1);
    cache_in->core->bp = reader_in->sdata->break_points;
    cache_in->core->cache_debug_level = 2;
    
    
    struct optimal_init_params* init_params = g_new0(struct optimal_init_params, 1);
    init_params->reader = reader_in;
    init_params->ts = 0;
    struct_cache* optimal; 
    if (cache_in->core->data_type == 'l')
        optimal = optimal_init(cache_in->core->size, 'l', 0, (void*)init_params);
    else{
        printf("other cache data type not supported in LRU_evict_err_statistics in generalProfiler\n");
        exit(1);
    }
    optimal->core->cache_debug_level = 1;
    optimal->core->eviction_array_len = reader_in->base->total_num;
    optimal->core->bp = reader_in->sdata->break_points;
    
    if (reader_in->base->total_num == -1)
        get_num_of_cache_lines(reader_in);
    
    if (reader_in->base->type == 'v')
        optimal->core->eviction_array = g_new0(guint64, reader_in->base->total_num);
    else
        optimal->core->eviction_array = g_new0(gchar*, reader_in->base->total_num);
    
    // get oracle
    traverse_trace(reader_in, optimal);

    cache_in->core->oracle = optimal->core->eviction_array;
    
    
    get_evict_err(reader_in, cache_in);
    
    optimal_destroy(optimal);
    
    
    return cache_in->core->evict_err_array;
}


#ifdef __cplusplus
}
#endif
