//
//  heatmap_thread.c
//  mimircache  
//
//  Created by Juncheng on 5/24/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "heatmap.h"


#ifdef __cplusplus
extern "C"
{
#endif


/**
 * thread function for computing nonLRU heatmap of type start_time_end_time 
 * 
 * @param data: contains order+1 
 * @param user_data: passed in param 
 */
void heatmap_nonLRU_hit_rate_start_time_end_time_thread(gpointer data, gpointer user_data){
    guint64 i, j, hit_count, miss_count;
    mt_params_hm_t* params = (mt_params_hm_t*) user_data;
    reader_t* reader_thread = clone_reader(params->reader);
    GArray* break_points = params->break_points;
    guint64* progress = params->progress;
    draw_dict* dd = params->dd;
    struct cache* cache = params->cache->core->cache_init(params->cache->core->size,
                                                          params->cache->core->data_type,
                                                          params->cache->core->block_unit_size, 
                                                          params->cache->core->cache_init_params);
    
    int order = GPOINTER_TO_INT(data)-1;
    
    hit_count = 0;
    miss_count = 0;


    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = cache->core->data_type;
    cp->block_unit_size = (size_t) reader_thread->base->block_unit_size;

    guint64 N = g_array_index(break_points, guint64, order);
    if (N != skip_N_elements(reader_thread, N)){
        ERROR("failed to skip %lu requests\n", N);
        exit(1); 
    };
    
    // this is for synchronizing ts in cache, which is used as index for access next_access array 
    if (cache->core->type == e_Optimal)
        ((struct optimal_params*)(cache->cache_params))->ts = g_array_index(break_points, guint64, order);
    
    for (i=order; i<break_points->len-1; i++){
        for(j=0; j< g_array_index(break_points, guint64, i+1) - g_array_index(break_points, guint64, i); j++){
            read_one_element(reader_thread, cp);
            if (cache->core->add_element(cache, cp))
                hit_count++;
            else
                miss_count++;
        }
        dd->matrix[order][i] = (double)(hit_count)/(hit_count+miss_count);
    }


    // clean up
    g_mutex_lock(&(params->mtx));
    (*progress) ++ ;
    g_mutex_unlock(&(params->mtx));
    g_free(cp);
    
    
    close_reader_unique(reader_thread);
    cache->core->destroy_unique(cache);
}


/**
 * thread function for computing nonLRU heatmap of type start_time_end_time 
 * 
 * @param data: contains order+1 
 * @param user_data: passed in param 
 */
void heatmap_LRU_hit_rate_start_time_end_time_thread(gpointer data, gpointer user_data){

    guint64 i, j, hit_count, miss_count;
    mt_params_hm_t* params = (mt_params_hm_t*) user_data;
    reader_t* reader_thread = clone_reader(params->reader);
    GArray* break_points = params->break_points;
    guint64* progress = params->progress;
    draw_dict* dd = params->dd;
    guint64 cache_size = (guint64)params->cache->core->size;
    gint* last_access = reader_thread->sdata->last_access;
    gint64* reuse_dist = reader_thread->sdata->reuse_dist;
    
    int order = GPOINTER_TO_INT(data)-1;
    guint64 real_start = g_array_index(break_points, guint64, order);
    
    hit_count = 0;
    miss_count = 0;
    

    // unnecessary ? 
    skip_N_elements(reader_thread, g_array_index(break_points, guint64, order));

    for (i=order; i<break_points->len-1; i++){
        for(j=g_array_index(break_points, guint64, i); j< g_array_index(break_points, guint64, i+1); j++){
            if (reuse_dist[j] == -1)
                miss_count ++;
            else if (last_access[j] - (long long)(j - real_start) <= 0 && reuse_dist[j] < (long long)cache_size)
                hit_count ++;
            else
                miss_count ++;
        }
        dd->matrix[order][i] = (double)(hit_count)/(hit_count+miss_count);
    }
    
    // clean up
    g_mutex_lock(&(params->mtx));
    (*progress) ++ ;
    g_mutex_unlock(&(params->mtx));
    close_reader_unique(reader_thread);
}



/**
 * thread function for computing nonLRU heatmap of type start_time_end_time 
 * 
 * @param data: contains order+1 
 * @param user_data: passed in param 
 */
void heatmap_LRU_hitrate_start_time_cachesize_thread(gpointer data, gpointer user_data){

    guint64 i, j, hit_count, miss_count; 
    mt_params_hm_t* params = (mt_params_hm_t*) user_data;
    reader_t* reader_thread = clone_reader(params->reader);
    GArray* break_points = params->break_points;
    guint64* progress = params->progress;
    draw_dict* dd = params->dd;
    guint64 cache_size = (guint64)params->cache->core->size;
    gint* last_access = reader_thread->sdata->last_access;
    gint64* reuse_dist = reader_thread->sdata->reuse_dist;
    
    int order = GPOINTER_TO_INT(data)-1;
    guint64 real_start = g_array_index(break_points, guint64, order);
    
    hit_count = 0;
    miss_count = 0;
    

    // unnecessary ? 
    skip_N_elements(reader_thread, g_array_index(break_points, guint64, order));

    for (i=order; i<break_points->len-1; i++){
        
        for(j=g_array_index(break_points, guint64, i); j< g_array_index(break_points, guint64, i+1); j++){
            if (reuse_dist[j] == -1)
                miss_count ++;
            else if (last_access[j] - (long long)(j - real_start) <= 0 && reuse_dist[j] < (long long)cache_size)
                hit_count ++;
            else
                miss_count ++;
        }
        dd->matrix[order][i] = (double)(hit_count)/(hit_count+miss_count);
    }
    
    // clean up
    g_mutex_lock(&(params->mtx));
    (*progress) ++ ;
    g_mutex_unlock(&(params->mtx));
    close_reader_unique(reader_thread);
}






void heatmap_rd_distribution_thread(gpointer data, gpointer user_data){
    
    guint64 j;
    mt_params_hm_t* params = (mt_params_hm_t*) user_data;
    GArray* break_points = params->break_points;
    guint64* progress = params->progress;
    draw_dict* dd = params->dd;
    gint64* reuse_dist = params->reader->sdata->reuse_dist;
    double log_base = params->log_base;
    
    guint64 order = (guint64)GPOINTER_TO_INT(data)-1;
    double* array = dd->matrix[order];
    
    if (order != break_points->len-1){
        for(j=g_array_index(break_points, guint64, order); j< g_array_index(break_points, guint64, order+1); j++){
            if (reuse_dist[j] == 0 ||reuse_dist[j] == 1)
                array[0] += 1;
            else
                array[(long)(log(reuse_dist[j])/(log(log_base)))] += 1;
        }
    }

    // clean up
    g_mutex_lock(&(params->mtx));
    (*progress) ++ ;
    g_mutex_unlock(&(params->mtx));
}

void heatmap_rd_distribution_CDF_thread(gpointer data, gpointer user_data){
    
    guint64 j;
    mt_params_hm_t* params = (mt_params_hm_t*) user_data;
    GArray* break_points = params->break_points;
    guint64* progress = params->progress;
    draw_dict* dd = params->dd;
    gint64* reuse_dist = params->reader->sdata->reuse_dist;
    double log_base = params->log_base;
    
    guint64 order = (guint64)GPOINTER_TO_INT(data)-1;
    double* array = dd->matrix[order];
    
    
    if (order != break_points->len-1){
        for(j=g_array_index(break_points, guint64, order); j< g_array_index(break_points, guint64, order+1); j++){
            if (reuse_dist[j] == 0 ||reuse_dist[j] == 1)
                array[0] += 1;
            else
                array[(long)(log(reuse_dist[j])/(log(log_base)))] += 1;
        }
    }

    for (j=1; j<dd->ylength; j++)
        array[j] += array[j-1];
    
    for (j=0; j<dd->ylength; j++)
        array[j] = array[j]/array[dd->ylength-1];
    
    // clean up
    g_mutex_lock(&(params->mtx));
    (*progress) ++ ;
    g_mutex_unlock(&(params->mtx));
}


#ifdef __cplusplus
}
#endif