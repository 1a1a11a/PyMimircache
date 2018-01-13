//
//  heatmap.c
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
    
    
    //draw_dict* heatmap_LRU(reader_t* reader,
    //                       struct_cache* cache,
    //                       char time_mode,
    //                       gint64 bin_size,
    //                       heatmap_type_e plot_type,
    //                       int interval_hit_ratio_b,
    //                       double decay_coefficient_lf,
    //                       int num_of_threads);
    //
    //draw_dict* heatmap_nonLRU(reader_t* reader,
    //                          struct_cache* cache,
    //                          char time_mode,
    //                          gint64 bin_size,
    //                          heatmap_type_e plot_type,
    //                          int interval_hit_ratio_b,
    //                          double decay_coefficient_lf,
    //                          int num_of_threads);
    
    
    draw_dict* heatmap_computation(reader_t* reader,
                                   struct_cache* cache,
                                   char time_mode,
                                   gint64 bin_size,
                                   heatmap_type_e plot_type,
                                   int interval_hit_ratio_b,
                                   double decay_coefficient_lf,
                                   int num_of_threads);
    
    
    draw_dict* hm_hr_st_et(reader_t* reader,
                           struct_cache* cache,
                           int interval_hit_ratio_b,
                           double decay_coefficient_lf,
                           int num_of_threads);
    
    draw_dict* hm_hr_interval_size(reader_t* reader,
                                   struct_cache* cache,
                                   gint64 bin_size,
                                   double decay_coefficient_lf,
                                   int num_of_threads);
    
    
    
    
    
    
    
    
    /*-----------------------------------------------------------------------------
     *
     * heatmap --
     *      the entrance for heatmap computation
     *
     * Input:
     *      reader:                 the reader for data
     *      cache:                  cache which heatmap is based on
     *      time_mode:              real time (r) or virtual time (v)
     *      time_interval:          real time interval or vritual time interval
     *      num_of_pixels:          the number of pixels in x/y dimension,
     *                                  this is optional, if time_interval is specified,
     *                                  then this one is not needed
     *      plot_type:              the type of plot
     *
     *      interval_hit_ratio_b:   used in hit_ratio_start_time_end_time,
     *                                  if it is True, then the hit ratio of each pixel is not
     *                                  the average hit ratio from beginning,
     *                                  instead it is a combined hit ratio of exponentially decayed
     *                                  average hit ratio plus hit ratio in current interval
     *      decay_coefficient_lf:      used only when interval_hit_ratio_b is True
     *
     *      num_of_threads: the maximum number of threads can use
     *
     * Return:
     *      draw_dict
     *
     *-----------------------------------------------------------------------------
     */
    
    draw_dict* heatmap(reader_t* reader,
                       struct_cache* cache,
                       char time_mode,
                       gint64 time_interval,
                       gint64 bin_size,
                       gint64 num_of_pixel_for_time_dim,
                       heatmap_type_e plot_type,
                       int interval_hit_ratio_b,
                       double decay_coefficient_lf,
                       int num_of_threads){
        
        if (time_mode == 'v')
            get_bp_vtime(reader, time_interval, num_of_pixel_for_time_dim);
        else if (time_mode == 'r')
            get_bp_rtime(reader, time_interval, num_of_pixel_for_time_dim);
        else{
            ERROR("unsupported mode: %c\n", time_mode);
            abort();
        }
        
        
        // check cache is LRU or not
        if (cache==NULL || cache->core->type == e_LRU){
            
            if (plot_type != future_rd_distribution &&
                plot_type != dist_distribution &&
                plot_type != rt_distribution) {
                if (reader->sdata->reuse_dist_type != NORMAL_REUSE_DISTANCE){
                    get_reuse_dist_seq(reader, 0, -1);
                }
            }
            
            //        return heatmap_LRU(reader, cache, time_mode, bin_size, plot_type,
            //                           interval_hit_ratio_b, decay_coefficient_lf, num_of_threads);
        }
        //    else{
        //        return heatmap_nonLRU(reader, cache, time_mode, bin_size, plot_type,
        //                              interval_hit_ratio_b, decay_coefficient_lf, num_of_threads);
        //    }
        
        return heatmap_computation(reader, cache, time_mode, bin_size, plot_type,
                                   interval_hit_ratio_b, decay_coefficient_lf, num_of_threads);
        
    }
    
    
    
    /*-----------------------------------------------------------------------------
     *
     * heatmap_LRU --
     *      heatmap computation for LRU
     *
     * Input:
     *      reader:         the reader for data
     *      cache:          cache which heatmap is based on, contains information
     *                      including cache_size, etc.
     *      time_mode:           real time (r) or virtual time (v)
     *      time_interval:  real time interval or vritual time interval
     *      num_of_pixels:  the number of pixels in x/y dimension,
     *                      this is optional, if time_interval is specified,
     *                      then this one is not needed
     *      plot_type:      the type of plot
     *
     *      interval_hit_ratio_b:   used in hit_ratio_start_time_end_time,
     *                                  if it is True, then the hit ratio of each pixel is not
     *                                  the average hit ratio from beginning,
     *                                  instead it is a combined hit ratio of exponentially decayed
     *                                  average hit ratio plus hit ratio in current interval
     *      decay_coefficient_lf:      used only when interval_hit_ratio_b is True
     *
     *      num_of_threads: the maximum number of threads can use
     *
     * Return:
     *      draw_dict
     *
     *-----------------------------------------------------------------------------
     */
    
    draw_dict* heatmap_computation(reader_t* reader,
                                   struct_cache* cache,
                                   char time_mode,
                                   gint64 bin_size,
                                   heatmap_type_e plot_type,
                                   int interval_hit_ratio_b,
                                   double decay_coefficient_lf,
                                   int num_of_threads){
        
        //    if (plot_type != future_rd_distribution &&
        //        plot_type != dist_distribution &&
        //        plot_type != rt_distribution) {
        //        if (reader->sdata->reuse_dist_type != NORMAL_REUSE_DISTANCE)
        //            get_reuse_dist_seq(reader, 0, -1);
        //    }
        
        if (plot_type == hr_st_et){
            GSList* last_access_gslist = get_last_access_dist_seq(reader, read_one_element);
            if (reader->sdata->last_access != NULL){
                g_free(reader->sdata->last_access);
                reader->sdata->last_access = NULL;
            }
            reader->sdata->last_access = g_new(gint, reader->base->total_num);
            GSList* sl_node = last_access_gslist;
            guint64 counter = reader->base->total_num-1;
            reader->sdata->last_access[counter--] = GPOINTER_TO_INT(sl_node->data);
            while(sl_node->next){
                sl_node = sl_node->next;
                reader->sdata->last_access[counter--] = GPOINTER_TO_INT(sl_node->data);
            }
            g_slist_free(last_access_gslist);
            return hm_hr_st_et(reader,
                               cache,
                               interval_hit_ratio_b,
                               decay_coefficient_lf,
                               num_of_threads);
        }
        
        
        else if (plot_type == hr_interval_size){
            GSList* last_access_gslist = get_last_access_dist_seq(reader, read_one_element);
            if (reader->sdata->last_access != NULL){
                g_free(reader->sdata->last_access);
                reader->sdata->last_access = NULL;
            }
            reader->sdata->last_access = g_new(gint, reader->base->total_num);
            GSList* sl_node = last_access_gslist;
            guint64 counter = reader->base->total_num-1;
            reader->sdata->last_access[counter--] = GPOINTER_TO_INT(sl_node->data);
            while(sl_node->next){
                sl_node = sl_node->next;
                reader->sdata->last_access[counter--] = GPOINTER_TO_INT(sl_node->data);
            }
            g_slist_free(last_access_gslist);
            return hm_hr_interval_size(reader,
                                       cache,
                                       bin_size,
                                       decay_coefficient_lf,
                                       num_of_threads);
            
            
        }
        
        else if (plot_type == hr_st_size){
            WARNING("not implemented \n");
            
            
        }
        
        else if (plot_type == avg_rd_st_et){
            WARNING("not implemented \n");
            
            
            
        }
        
        
        else if (plot_type == rd_distribution){
            return heatmap_rd_distribution(reader, time_mode, num_of_threads, 0);
        }
        else if (plot_type == rd_distribution_CDF){
            return heatmap_rd_distribution(reader, time_mode, num_of_threads, 1);
        }
        else if (plot_type == future_rd_distribution){
            if (reader->sdata->reuse_dist_type != FUTURE_REUSE_DISTANCE){
                g_free(reader->sdata->reuse_dist);
                reader->sdata->reuse_dist = NULL;
                reader->sdata->reuse_dist = get_future_reuse_dist(reader, 0, -1);
            }
            
            draw_dict* dd = heatmap_rd_distribution(reader, time_mode, num_of_threads, 0);
            
            reader->sdata->max_reuse_dist = 0;
            g_free(reader->sdata->reuse_dist);
            reader->sdata->reuse_dist = NULL;
            
            return dd;
        }
        else if (plot_type == dist_distribution){
            if (reader->sdata->reuse_dist_type != NORMAL_DISTANCE){
                g_free(reader->sdata->reuse_dist);
                reader->sdata->reuse_dist = NULL;
                reader->sdata->reuse_dist = get_dist_to_last_access(reader);
            }
            
            draw_dict* dd = heatmap_rd_distribution(reader, time_mode, num_of_threads, 0);
            reader->sdata->max_reuse_dist = 0;
            g_free(reader->sdata->reuse_dist);
            reader->sdata->reuse_dist = NULL;
            
            return dd;
        }
        else if (plot_type == rt_distribution){         // real time distribution, time msut be integer
            if (reader->sdata->reuse_dist_type != REUSE_TIME){
                g_free(reader->sdata->reuse_dist);
                reader->sdata->reuse_dist = NULL;
                reader->sdata->reuse_dist = get_reuse_time(reader);
            }
            
            draw_dict* dd = heatmap_rd_distribution(reader, time_mode, num_of_threads, 0);
            reader->sdata->max_reuse_dist = 0;
            g_free(reader->sdata->reuse_dist);
            reader->sdata->reuse_dist = NULL;
            
            return dd;
        }
        
        else {
            ERROR("unknown plot type\n");
            exit(1);
        }
        
        return NULL;
    }
    
    
    //draw_dict* heatmap_nonLRU(reader_t* reader,
    //                          struct_cache* cache,
    //                          char time_mode,
    //                          gint64 bin_size,
    //                          heatmap_type_e plot_type,
    //                          int interval_hit_ratio_b,
    //                          double decay_coefficient_lf,
    //                          int num_of_threads){
    //
    //    if (plot_type == hr_st_et){
    //        return hm_hr_st_et(reader, cache, time_mode, bin_size,
    //                                                     plot_type, interval_hit_ratio_b,
    //                                                     decay_coefficient_lf,
    //                                                     num_of_threads);
    //    }
    //
    //    else if (plot_type == hr_interval_size){
    //
    //
    //    }
    //
    //    else if (plot_type == hr_st_size){
    //
    //
    //
    //    }
    //
    //    else if (plot_type == avg_rd_st_et){
    //
    //
    //
    //    }
    //
    //    else {
    //        ERROR("unknown plot type\n");
    //        exit(1);
    //    }
    //
    //    return NULL;
    //}
    
    
    
    /** heatmap_hit_ratio_start_time_end_time **/
    
    draw_dict* hm_hr_st_et(reader_t* reader,
                           struct_cache* cache,
                           int interval_hit_ratio_b,
                           double ewma_coefficient_lf,
                           int num_of_threads){
        
        gint i;
        guint64 progress = 0;
        
        GArray* break_points;
        break_points = reader->sdata->break_points->array;
        
        // create draw_dict storage
        draw_dict* dd = g_new(draw_dict, 1);
        dd->xlength = break_points->len - 1;
        dd->ylength = break_points->len - 1;
        dd->matrix = g_new(double*, dd->xlength);
        for (i=0; i< (gint) (dd->xlength); i++)
            dd->matrix[i] = g_new0(double, dd->ylength);
        
        
        // build parameters
        mt_params_hm_t* params = g_new(mt_params_hm_t, 1);
        params->reader = reader;
        params->break_points = break_points;
        params->cache = cache;
        params->dd = dd;
        params->interval_hit_ratio_b = interval_hit_ratio_b;
        params->ewma_coefficient_lf = ewma_coefficient_lf;
        params->progress = &progress;
        g_mutex_init(&(params->mtx));
        
        
        // build the thread pool
        GThreadPool * gthread_pool;
        if (cache->core->type == e_LRU)
            gthread_pool = g_thread_pool_new ( (GFunc) hm_LRU_hr_st_et_thread,
                                              (gpointer)params, num_of_threads, TRUE, NULL);
        else
            gthread_pool = g_thread_pool_new ( (GFunc) hm_nonLRU_hr_st_et_thread,
                                              (gpointer)params, num_of_threads, TRUE, NULL);
        
        if (gthread_pool == NULL)
            ERROR("cannot create thread pool in heatmap\n");
        
        
        // send data to thread pool and begin computation
        //    for (i=0; i<break_points->len-1; i++){
        for (i=break_points->len-2; i>=0; i--){
            if ( g_thread_pool_push (gthread_pool, GINT_TO_POINTER(i+1), NULL) == FALSE)    // +1 otherwise, 0 will be a problem
                ERROR("cannot push data into thread in generalprofiler\n");
        }
        
        while ( progress < break_points->len-1 ){
            fprintf(stderr, "%.2f%%\n", ((double)progress)/break_points->len*100);
            sleep(5);
//            fprintf(stderr, "\033[A\033[2K\r");
        }
        
        g_thread_pool_free (gthread_pool, FALSE, TRUE);
        
        
        g_mutex_clear(&(params->mtx));
        g_free(params);
        // needs to free draw_dict later
        return dd;
    }
    
    
    draw_dict* heatmap_rd_distribution(reader_t* reader, char mode, int num_of_threads, int CDF){
        /* Do NOT call this function directly,
         * call top level heatmap, which setups some data for this function */
        
        /* this one, the result is in the log form */
        
        guint i;
        guint64 progress = 0;
        
        GArray* break_points;
        break_points = reader->sdata->break_points->array;
        
        // this is used to make sure length of x and y are approximate same, not different by too much
        if (reader->sdata->max_reuse_dist == 0 || break_points->len == 0){
            ERROR("did you call top level function? max reuse distance %ld, bp len %u\n",
                  reader->sdata->max_reuse_dist, break_points->len);
            exit(1);
        }
        double log_base = get_log_base(reader->sdata->max_reuse_dist, break_points->len);
        reader->udata->log_base = log_base;
        
        // create draw_dict storage
        draw_dict* dd = g_new(draw_dict, 1);
        dd->xlength = break_points->len - 1;
        
        // the last one is used for store cold miss; rd=0 and rd=1 are combined at first bin (index=0)
        dd->ylength = (long) ceil(log(reader->sdata->max_reuse_dist)/log(log_base));
        dd->matrix = g_new(double*, break_points->len);
        for (i=0; i<dd->xlength; i++)
            dd->matrix[i] = g_new0(double, dd->ylength);
        
        
        
        // build parameters
        mt_params_hm_t* params = g_new(mt_params_hm_t, 1);
        params->reader = reader;
        params->break_points = break_points;
        params->dd = dd;
        params->progress = &progress;
        params->log_base = log_base;
        g_mutex_init(&(params->mtx));
        
        
        // build the thread pool
        GThreadPool * gthread_pool;
        if (!CDF)
            gthread_pool = g_thread_pool_new ( (GFunc) hm_rd_distribution_thread,
                                              (gpointer)params, num_of_threads, TRUE, NULL);
        else
            gthread_pool = g_thread_pool_new ( (GFunc) hm_rd_distribution_CDF_thread,
                                              (gpointer)params, num_of_threads, TRUE, NULL);
        
        if (gthread_pool == NULL)
            ERROR("cannot create thread pool in heatmap rd_distribution\n");
        
        
        // send data to thread pool and begin computation
        for (i=0; i<break_points->len-1; i++){
            if ( g_thread_pool_push (gthread_pool, GINT_TO_POINTER(i+1), NULL) == FALSE)    // +1, otherwise, 0 will be a problem
                ERROR("cannot push data into thread in generalprofiler\n");
        }
        
        while ( progress < break_points->len-1 ){
            fprintf(stderr, "%.2lf%%", ((double)progress)/break_points->len*100);
            sleep(5);
//            fprintf(stderr, "\033[A\033[2K\r");
        }
        
        g_thread_pool_free (gthread_pool, FALSE, TRUE);
        
        
        g_mutex_clear(&(params->mtx));
        g_free(params);
        
        return dd;
    }
    
    
    
    draw_dict* differential_heatmap(reader_t* reader,
                                    struct_cache* cache1,
                                    struct_cache* cache2,
                                    char time_mode,
                                    gint64 bin_size,
                                    gint64 time_interval,
                                    gint64 num_of_pixel_for_time_dim,
                                    heatmap_type_e plot_type,
                                    int interval_hit_ratio_b,
                                    double ewma_coefficient_lf,
                                    int num_of_threads){
        
        if (time_mode == 'v')
            get_bp_vtime(reader, time_interval, num_of_pixel_for_time_dim);
        else if (time_mode == 'r')
            get_bp_rtime(reader, time_interval, num_of_pixel_for_time_dim);
        else{
            ERROR("unsupported mode: %c\n", time_mode);
            exit(1);
        }

        if (cache1==NULL || cache1->core->type == e_LRU \
            || cache2==NULL || cache2->core->type == e_LRU){
            if (plot_type != future_rd_distribution &&
                plot_type != dist_distribution &&
                plot_type != rt_distribution) {
                if (reader->sdata->reuse_dist_type != NORMAL_REUSE_DISTANCE){
                    get_reuse_dist_seq(reader, 0, -1);
                }
            }
        }
        
        draw_dict *draw_dict1, *draw_dict2;
        draw_dict1 = heatmap_computation(reader,
                                         cache1,
                                         time_mode,
                                         bin_size,
                                         plot_type,
                                         interval_hit_ratio_b,
                                         ewma_coefficient_lf,
                                         num_of_threads);
        
        draw_dict2 = heatmap_computation(reader,
                                         cache2,
                                         time_mode,
                                         bin_size,
                                         plot_type,
                                         interval_hit_ratio_b,
                                         ewma_coefficient_lf,
                                         num_of_threads);
        
        // check cache is LRU or not
        //    if (cache1 == NULL || cache1->core->type == e_LRU){
        //        draw_dict1 = heatmap_LRU(reader,
        //                                 cache1,
        //                                 time_mode,
        //                                 bin_size,
        //                                 plot_type,
        //                                 interval_hit_ratio_b,
        //                                 ewma_coefficient_lf,
        //                                 num_of_threads);
        //    }
        //    else{
        //        draw_dict1 = heatmap_nonLRU(reader,
        //                                    cache1,
        //                                    time_mode,
        //                                    bin_size,
        //                                    plot_type,
        //                                    interval_hit_ratio_b,
        //                                    ewma_coefficient_lf,
        //                                    num_of_threads);
        //    }
        //
        //    if (cache2 == NULL || cache2->core->type == e_LRU){
        //        draw_dict2 = heatmap_LRU(reader,
        //                                 cache2,
        //                                 time_mode,
        //                                 bin_size,
        //                                 plot_type,
        //                                 interval_hit_ratio_b,
        //                                 ewma_coefficient_lf,
        //                                 num_of_threads);
        //    }
        //    else{
        //        draw_dict2 = heatmap_nonLRU(reader,
        //                                    cache2,
        //                                    time_mode,
        //                                    bin_size,
        //                                    plot_type,
        //                                    interval_hit_ratio_b,
        //                                    ewma_coefficient_lf,
        //                                    num_of_threads);
        //    }
        
        
        guint64 i, j;
        for (i=0; i<draw_dict1->xlength; i++)
            for (j=0; j<draw_dict1->ylength; j++){
                draw_dict2->matrix[i][j] = (draw_dict2->matrix[i][j] -
                                            draw_dict1->matrix[i][j]) /
                draw_dict1->matrix[i][j];
            }
        free_draw_dict(draw_dict1);
        return draw_dict2;
    }
    
    
    draw_dict* hm_hr_interval_size(reader_t* reader,
                                   struct_cache* cache,
                                   gint64 bin_size,
                                   double ewma_coefficient_lf,
                                   int num_of_threads){
        
        gint i;
        guint64 progress = 0;
        
        GArray* break_points;
        break_points = reader->sdata->break_points->array;
        
        gint num_of_bins = (int) ceil(cache->core->size / bin_size) + 1;
        
        if (num_of_bins <= 1)
            // the first bin is size 0
            num_of_bins = 2;
        
        // create draw_dict storage
        draw_dict* dd = g_new(draw_dict, 1);
        dd->xlength = break_points->len - 1;
        dd->ylength = num_of_bins;
        
        //    // number of bins
        //    dd->ylength = num_of_bins;
        
        dd->matrix = g_new(double*, dd->xlength);
        for (i=0; i<(gint) (dd->xlength); i++)
            dd->matrix[i] = g_new0(double, dd->ylength);
        
        
        // build parameters
        mt_params_hm_t* params = g_new(mt_params_hm_t, 1);
        params->reader = reader;
        params->break_points = break_points;
        params->cache = cache;
        params->bin_size = bin_size;
        params->dd = dd;
        
        params->ewma_coefficient_lf = ewma_coefficient_lf;
        params->progress = &progress;
        g_mutex_init(&(params->mtx));
        
        
        // build the thread pool
        GThreadPool * gthread_pool;
        if (cache->core->type == e_LRU)
            gthread_pool = g_thread_pool_new ( (GFunc) hm_LRU_hr_interval_size_thread,
                                              (gpointer)params, num_of_threads, TRUE, NULL);
        else
            gthread_pool = g_thread_pool_new ( (GFunc) hm_nonLRU_hr_interval_size_thread,
                                              (gpointer)params, num_of_threads, TRUE, NULL);
        
        if (gthread_pool == NULL)
            ERROR("cannot create thread pool in heatmap\n");
        
        
        // send data to thread pool and begin computation
        for (i=num_of_bins-1; i>=0; i--){
            if ( g_thread_pool_push (gthread_pool, GINT_TO_POINTER(i+1), NULL) == FALSE)    // +1 otherwise, 0 will be a problem
                ERROR("cannot push data into thread in generalprofiler\n");
        }
        
        while ( progress < (guint) num_of_bins ){
            fprintf(stderr, "%.2f%%\n", ((double)progress)/num_of_bins*100);
            sleep(5);
//            fprintf(stderr, "\033[A\033[2K\r");
        }
        
        g_thread_pool_free (gthread_pool, FALSE, TRUE);
        
        
        g_mutex_clear(&(params->mtx));
        g_free(params);
        // needs to free draw_dict later
        return dd;
    }
    
    
    
    
    
    
    void free_draw_dict(draw_dict* dd){
        guint64 i;
        for (i=0; i<dd->xlength; i++){
            g_free(dd->matrix[i]);
        }
        g_free(dd->matrix);
        g_free(dd);
    }
    
    
    
    
    
    
    
#ifdef __cplusplus
}
#endif
