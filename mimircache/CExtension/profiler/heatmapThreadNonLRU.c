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
    void hm_nonLRU_hr_st_et_thread(gpointer data, gpointer user_data){
        guint64 i, j, hit_count_all, miss_count_all, hit_count_interval, miss_count_interval;
        mt_params_hm_t* params = (mt_params_hm_t*) user_data;
        reader_t* reader_thread = clone_reader(params->reader);
        GArray* break_points = params->break_points;
        guint64* progress = params->progress;
        draw_dict* dd = params->dd;
        struct cache* cache = params->cache->core->cache_init(params->cache->core->size,
                                                              params->cache->core->data_type,
                                                              params->cache->core->block_unit_size,
                                                              params->cache->core->cache_init_params);
        
        guint64 order = GPOINTER_TO_INT(data)-1;
        int interval_hit_ratio_b = params->interval_hit_ratio_b;
        double ewma_coefficient_lf = params->ewma_coefficient_lf;
        
        hit_count_all = 0;
        miss_count_all = 0;
        
        
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
            hit_count_interval = 0;
            miss_count_interval = 0;
            for(j=0; j< g_array_index(break_points, guint64, i+1) - g_array_index(break_points, guint64, i); j++){
                read_one_element(reader_thread, cp);
                if (cache->core->add_element(cache, cp))
                    hit_count_interval++;
                else
                    miss_count_interval++;
            }
            hit_count_all += hit_count_interval;
            miss_count_all += miss_count_interval;
            if (interval_hit_ratio_b){
                if (i == order)
                    // no decay for first pixel
                    dd->matrix[order][i] = (double)(hit_count_all)/(hit_count_all + miss_count_all);
                else {
                    dd->matrix[order][i] = dd->matrix[order][i-1] * ewma_coefficient_lf +
                    (1 - ewma_coefficient_lf) * (double)(hit_count_interval)/(hit_count_interval + miss_count_interval);
                }
            }
            else
                dd->matrix[order][i] = (double)(hit_count_all)/(hit_count_all + miss_count_all);
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
     * thread function for computing nonLRU heatmap of type interval_size
     *
     * @param data: contains order+1
     * @param user_data: passed in param
     */
    void hm_nonLRU_hr_interval_size_thread(gpointer data, gpointer user_data){
        guint64 i, j, hit_count_interval, miss_count_interval;
        mt_params_hm_t* params = (mt_params_hm_t*) user_data;
        reader_t* reader_thread = clone_reader(params->reader);
        GArray* break_points = params->break_points;
        guint64* progress = params->progress;
        draw_dict* dd = params->dd;
        
        struct cache* cache = NULL;
        cache_line* cp = NULL;
        
        guint64 order = GPOINTER_TO_INT(data)-1;
        guint64 cache_size = params->bin_size * order;
        if (cache_size == 0){
            for (i=0; i<break_points->len-1; i++)
                dd->matrix[i][order] = 0;
            
        }
        else {
            cache = params->cache->core->cache_init(cache_size,
                                                    params->cache->core->data_type,
                                                    params->cache->core->block_unit_size,
                                                    params->cache->core->cache_init_params);
            
            double ewma_coefficient_lf = params->ewma_coefficient_lf;
            
            
            // create cache lize struct and initialization
            cp = new_cacheline();
            cp->type = cache->core->data_type;
            cp->block_unit_size = (size_t) reader_thread->base->block_unit_size;
            
                        
            for (i=0; i<break_points->len-1; i++){
                hit_count_interval = 0;
                miss_count_interval = 0;
                for(j=0; j< g_array_index(break_points, guint64, i+1) - g_array_index(break_points, guint64, i); j++){
                    read_one_element(reader_thread, cp);
                    if (cache->core->add_element(cache, cp))
                        hit_count_interval++;
                    else
                        miss_count_interval++;
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
        if (cp != NULL)
            g_free(cp);
        
        close_reader_unique(reader_thread);
        if (cache != NULL)
            cache->core->destroy_unique(cache);
    }
    
    
#ifdef __cplusplus
}
#endif
