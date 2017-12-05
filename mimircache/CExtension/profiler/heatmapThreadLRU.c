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
     * thread function for computing LRU heatmap of type start_time_end_time
     *
     * @param data: contains order+1
     * @param user_data: passed in param
     */
    void hm_LRU_hr_st_et_thread(gpointer data, gpointer user_data){
        
        guint64 i, j, hit_count_all, miss_count_all, hit_count_interval, miss_count_interval;
        mt_params_hm_t* params = (mt_params_hm_t*) user_data;
        reader_t* reader_thread = clone_reader(params->reader);
        GArray* break_points = params->break_points;
        guint64* progress = params->progress;
        draw_dict* dd = params->dd;
        guint64 cache_size = (guint64)params->cache->core->size;
        // distance to last access
        gint* last_access = reader_thread->sdata->last_access;
        gint64* reuse_dist = reader_thread->sdata->reuse_dist;
        
        guint64 order = GPOINTER_TO_INT(data)-1;
        guint64 real_start = g_array_index(break_points, guint64, order);
        int interval_hit_ratio_b = params->interval_hit_ratio_b;
        double decay_coefficient_lf = params->ewma_coefficient_lf;
        
        hit_count_all = 0;
        miss_count_all = 0;
        
        
        // unnecessary ?
        skip_N_elements(reader_thread, g_array_index(break_points, guint64, order));
        
        for (i=order; i<break_points->len-1; i++){
            hit_count_interval  = 0;
            miss_count_interval = 0;
            for(j=g_array_index(break_points, guint64, i); j< g_array_index(break_points, guint64, i+1); j++){
                if (reuse_dist[j] == -1){
                    miss_count_interval ++;
                }
                else if (last_access[j] - (long long)(j - real_start) <= 0 && reuse_dist[j] < (long long)cache_size)
                    hit_count_interval ++;
                else
                    miss_count_interval ++;
            }
            hit_count_all += hit_count_interval;
            miss_count_all += miss_count_interval;
            if (interval_hit_ratio_b){
                if (i == order)
                    // no decay for first pixel
                    dd->matrix[order][i] = (double)(hit_count_all)/(hit_count_all + miss_count_all);
                else {
                    dd->matrix[order][i] = dd->matrix[order][i-1] * decay_coefficient_lf +
                    (1 - decay_coefficient_lf) * (double)(hit_count_interval)/(hit_count_interval + miss_count_interval);
                }
            }
            else
                dd->matrix[order][i] = (double)(hit_count_all)/(hit_count_all + miss_count_all);
        }
        
        // clean up
        g_mutex_lock(&(params->mtx));
        (*progress) ++ ;
        g_mutex_unlock(&(params->mtx));
        close_reader_unique(reader_thread);
    }
    
    
    
    /**
     * thread function for computing LRU heatmap of type interval_cacheSize
     *
     * @param data: contains order+1
     * @param user_data: passed in param
     */
    void hm_LRU_hr_interval_size_thread(gpointer data, gpointer user_data){
        
        guint64 i, j, hit_count_interval, miss_count_interval;
        
        mt_params_hm_t* params = (mt_params_hm_t*) user_data;
        reader_t* reader_thread = clone_reader(params->reader);
        GArray* break_points = params->break_points;
        guint64* progress = params->progress;
        draw_dict* dd = params->dd;
        
        
        guint64 order = GPOINTER_TO_INT(data)-1;
        // plus 1 because we don't need cache size 0
        gint64 cache_size = (gint64)params->bin_size * order;
        if (cache_size == 0){
            for (i=0; i<break_points->len - 1; i++)
                dd->matrix[i][order] = 0;
        }
        else {
            gint64* reuse_dist = reader_thread->sdata->reuse_dist;
            double ewma_coefficient_lf = params->ewma_coefficient_lf;
            
            
            for (i=0; i<break_points->len-1; i++){
                hit_count_interval  = 0;
                miss_count_interval = 0;
                for(j=g_array_index(break_points, guint64, i); j< g_array_index(break_points, guint64, i+1); j++){
                    if (reuse_dist[j] == -1){
                        miss_count_interval ++;
                    }
                    else if (reuse_dist[j] < (long long)cache_size)
                        hit_count_interval ++;
                    else
                        miss_count_interval ++;
                }
                
                if (i == 0)
                    // no decay for first pixel
                    dd->matrix[i][order] = (double)(hit_count_interval)/(hit_count_interval + miss_count_interval);
                else {
                    dd->matrix[i][order] = dd->matrix[i-1][order] * ewma_coefficient_lf +
                    (1 - ewma_coefficient_lf) * (double)(hit_count_interval)/(hit_count_interval + miss_count_interval);
                }
            }
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
    void hm_LRU_hr_st_size_thread(gpointer data, gpointer user_data){
        
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
    
    
    
    
    
    
    void hm_rd_distribution_thread(gpointer data, gpointer user_data){
        
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
    
    void hm_rd_distribution_CDF_thread(gpointer data, gpointer user_data){
        
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
