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


draw_dict* heatmap_LRU(reader_t* reader,
                       struct_cache* cache,
                       char mode,
                       heatmap_type_e plot_type,
                       int num_of_threads);

draw_dict* heatmap_nonLRU(reader_t* reader,
                          struct_cache* cache,
                          char mode,
                          heatmap_type_e plot_type,
                          int num_of_threads);

draw_dict* heatmap_hit_rate_start_time_end_time(reader_t* reader,
                                                struct_cache* cache,
                                                char mode,
                                                heatmap_type_e plot_type,
                                                int num_of_threads);








/*-----------------------------------------------------------------------------
 *
 * heatmap --
 *      the entrance for heatmap computation
 *
 * Input:
 *      reader:         the reader for data
 *      cache:          cache which heatmap is based on
 *      mode:           real time (r) or virtual time (v)
 *      time_interval:  real time interval or vritual time interval
 *      num_of_pixels:  the number of pixels in x/y dimension,
 *                      this is optional, if time_interval is specified,
 *                      then this one is not needed
 *      plot_type:      the type of plot
 *      num_of_threads: the maximum number of threads can use
 *
 * Return:
 *      draw_dict
 *
 *-----------------------------------------------------------------------------
 */

draw_dict* heatmap(reader_t* reader,
                   struct_cache* cache,
                   char mode,
                   gint64 time_interval,
                   gint64 num_of_pixels,
                   heatmap_type_e plot_type,
                   int num_of_threads){
    
    if (mode == 'v')
        gen_breakpoints_virtualtime(reader, time_interval, num_of_pixels);
    else if (mode == 'r')
        gen_breakpoints_realtime(reader, time_interval, num_of_pixels);
    else{
        ERROR("unsupported mode: %c\n", mode);
        abort();
    }


    // check cache is LRU or not
    if (cache==NULL || cache->core->type == e_LRU){
        return heatmap_LRU(reader, cache, mode, plot_type, num_of_threads);
    }
    else{
        return heatmap_nonLRU(reader, cache, mode, plot_type, num_of_threads);
    }
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
 *      mode:           real time (r) or virtual time (v)
 *      time_interval:  real time interval or vritual time interval
 *      num_of_pixels:  the number of pixels in x/y dimension,
 *                      this is optional, if time_interval is specified,
 *                      then this one is not needed
 *      plot_type:      the type of plot
 *      num_of_threads: the maximum number of threads can use
 *
 * Return:
 *      draw_dict
 *
 *-----------------------------------------------------------------------------
 */

draw_dict* heatmap_LRU(reader_t* reader,
                       struct_cache* cache,
                       char mode,
                       heatmap_type_e plot_type,
                       int num_of_threads){
    
    if (plot_type != future_rd_distribution &&
        plot_type != dist_distribution &&
        plot_type != rt_distribution) {
        if (reader->sdata->reuse_dist_type != NORMAL_REUSE_DISTANCE)
            get_reuse_dist_seq(reader, 0, -1);
    }
    
    if (plot_type == hit_rate_start_time_end_time){
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
        return heatmap_hit_rate_start_time_end_time(reader, cache, mode, plot_type, num_of_threads);
    }
    
    
    
    else if (plot_type == hit_rate_start_time_cache_size){
        
        
        
    }
    
    else if (plot_type == avg_rd_start_time_end_time){
        
        
        
    }

    else if (plot_type == cold_miss_count_start_time_end_time){
        
        
        
    }
    else if (plot_type == rd_distribution){
        return heatmap_rd_distribution(reader, mode, num_of_threads, 0);
    }
    else if (plot_type == rd_distribution_CDF){
        return heatmap_rd_distribution(reader, mode, num_of_threads, 1);
    }
    else if (plot_type == future_rd_distribution){
        if (reader->sdata->reuse_dist_type != FUTURE_REUSE_DISTANCE){
            g_free(reader->sdata->reuse_dist);
            reader->sdata->reuse_dist = NULL; 
            reader->sdata->reuse_dist = get_future_reuse_dist(reader, 0, -1);
        }
        
        draw_dict* dd = heatmap_rd_distribution(reader, mode, num_of_threads, 0);
        
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
        
        draw_dict* dd = heatmap_rd_distribution(reader, mode, num_of_threads, 0);
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
        
        draw_dict* dd = heatmap_rd_distribution(reader, mode, num_of_threads, 0);
        reader->sdata->max_reuse_dist = 0;
        g_free(reader->sdata->reuse_dist);
        reader->sdata->reuse_dist = NULL;
        
        return dd;
    }
    
    else if (plot_type == hit_rate_start_time_cache_size){
        
        
        
    }
    else {
        ERROR("unknown plot type\n");
        exit(1);
    }
    
    return NULL;
}


draw_dict* heatmap_nonLRU(reader_t* reader,
                          struct_cache* cache,
                          char mode,
                          heatmap_type_e plot_type,
                          int num_of_threads){
    
    if (plot_type == hit_rate_start_time_end_time){
        return heatmap_hit_rate_start_time_end_time(reader, cache, mode,
                                                    plot_type, num_of_threads);
    }
    else if (plot_type == hit_rate_start_time_cache_size){
        
        
        
    }
    
    else if (plot_type == avg_rd_start_time_end_time){
        
        
        
    }
    
    else if (plot_type == cold_miss_count_start_time_end_time){
        
        
        
    }

    else if (plot_type == hit_rate_start_time_cache_size){
        
        
        
    }
    else {
        ERROR("unknown plot type\n");
        exit(1);
    }
    
    return NULL;
}




draw_dict* heatmap_hit_rate_start_time_end_time(reader_t* reader,
                                                struct_cache* cache,
                                                char mode,
                                                heatmap_type_e plot_type,
                                                int num_of_threads){

    guint i;
    guint64 progress = 0;
    
    GArray* break_points;
    break_points = reader->sdata->break_points->array;
    

    // create draw_dict storage
    draw_dict* dd = g_new(draw_dict, 1);
    dd->xlength = break_points->len - 1;
    dd->ylength = break_points->len - 1;
    dd->matrix = g_new(double*, break_points->len);
    for (i=0; i<dd->xlength; i++)
        dd->matrix[i] = g_new0(double, break_points->len);
    

    // build parameters
    mt_params_hm_t* params = g_new(mt_params_hm_t, 1);
    params->reader = reader;
    params->break_points = break_points;
    params->cache = cache;
    params->dd = dd;
    params->progress = &progress;
    g_mutex_init(&(params->mtx));

    
    // build the thread pool
    GThreadPool * gthread_pool;
    if (cache->core->type == e_LRU)
        gthread_pool = g_thread_pool_new ( (GFunc) heatmap_LRU_hit_rate_start_time_end_time_thread,
                                          (gpointer)params, num_of_threads, TRUE, NULL);
    else
        gthread_pool = g_thread_pool_new ( (GFunc) heatmap_nonLRU_hit_rate_start_time_end_time_thread,
                                          (gpointer)params, num_of_threads, TRUE, NULL);
    
    if (gthread_pool == NULL)
        ERROR("cannot create thread pool in heatmap\n");
    
    
    // send data to thread pool and begin computation
    for (i=0; i<break_points->len-1; i++){
        if ( g_thread_pool_push (gthread_pool, GINT_TO_POINTER(i+1), NULL) == FALSE)    // +1 otherwise, 0 will be a problem
            ERROR("cannot push data into thread in generalprofiler\n");
    }
    
    while ( progress < break_points->len-1 ){
        fprintf(stderr, "%.2f%%\n", ((double)progress)/break_points->len*100);
        sleep(5);
        fprintf(stderr, "\033[A\033[2K\r");
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
        gthread_pool = g_thread_pool_new ( (GFunc) heatmap_rd_distribution_thread,
                                          (gpointer)params, num_of_threads, TRUE, NULL);
    else
        gthread_pool = g_thread_pool_new ( (GFunc) heatmap_rd_distribution_CDF_thread,
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
        fprintf(stderr, "\033[A\033[2K\r");
    }

    g_thread_pool_free (gthread_pool, FALSE, TRUE);

    
    g_mutex_clear(&(params->mtx));
    g_free(params);
    
    return dd;
}



draw_dict* differential_heatmap(reader_t* reader,
                                struct_cache* cache1,
                                struct_cache* cache2,
                                char mode,
                                gint64 time_interval,
                                gint64 num_of_pixels,
                                heatmap_type_e plot_type,
                                int num_of_threads){
    
    if (mode == 'v')
        gen_breakpoints_virtualtime(reader, time_interval, num_of_pixels);
    else if (mode == 'r')
        gen_breakpoints_realtime(reader, time_interval, num_of_pixels);
    else{
        ERROR("unsupported mode: %c\n", mode);
        exit(1);
    }

    draw_dict *draw_dict1, *draw_dict2;
    // check cache is LRU or not
    if (cache1 == NULL || cache1->core->type == e_LRU){
        draw_dict1 = heatmap_LRU(reader, cache1, mode, plot_type, num_of_threads);
    }
    else{
        draw_dict1 = heatmap_nonLRU(reader, cache1, mode, plot_type, num_of_threads);
    }

    if (cache2 == NULL || cache2->core->type == e_LRU){
        draw_dict2 = heatmap_LRU(reader, cache2, mode, plot_type, num_of_threads);
    }
    else{
        draw_dict2 = heatmap_nonLRU(reader, cache2, mode, plot_type, num_of_threads);
    }

    
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