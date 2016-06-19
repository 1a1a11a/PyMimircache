

#include "heatmap.h"


void heatmap_nonLRU_hit_rate_start_time_end_time(gpointer data, gpointer user_data){
    guint64 i, j, hit_count, miss_count;
    struct multithreading_params_heatmap* params = (struct multithreading_params_heatmap*) user_data;
    READER* reader_thread = copy_reader(params->reader);
    GArray* break_points = params->break_points;
    guint64* progress = params->progress;
    draw_dict* dd = params->dd;
    struct cache* cache = params->cache->cache_init(params->cache->size, params->cache->data_type, params->cache->cache_init_params);
    
    int order = GPOINTER_TO_INT(data)-1;
    
    hit_count = 0;
    miss_count = 0;


    // create cache lize struct and initialization
    cache_line* cp = (cache_line*)malloc(sizeof(cache_line));
    cp->op = -1;
    cp->size = -1;
    cp->valid = TRUE;
    
    skip_N_elements(reader_thread, g_array_index(break_points, guint64, order));
    
    // this is for synchronizing ts in cache, which is used as index for access next_access array 
    if (cache->type == e_Optimal)
        ((Optimal*)cache)->ts = g_array_index(break_points, guint64, order);
    
    for (i=order; i<break_points->len-1; i++){
        for(j=0; j< g_array_index(break_points, guint64, i+1) - g_array_index(break_points, guint64, i); j++){
            read_one_element(reader_thread, cp);
            if (cache->add_element(cache, cp))
                hit_count++;
            else
                miss_count++;
        }
        dd->matrix[order][i] = (float)(hit_count)/(hit_count+miss_count);
    }

    for(j=0; j< reader_thread->total_num - g_array_index(break_points, guint64, break_points->len - 1); j++){
        read_one_element(reader_thread, cp);
        if (!cp->valid)
            printf("detect error in heatmap_nonLRU_hit_rate_start_time_end_time, difference: %llu\n",
                   reader_thread->total_num - g_array_index(break_points, guint64, break_points->len - 1) - j);
        if (cache->add_element(cache, cp))
            hit_count++;
        else
            miss_count++;
    }
    dd->matrix[order][i] = (float)(hit_count)/(hit_count+miss_count);
    

    // clean up
    g_mutex_lock(&(params->mtx));
    (*progress) ++ ;
    g_mutex_unlock(&(params->mtx));
    free(cp);
    if (reader_thread->type != 'v')
        close_reader(reader_thread);
    cache->destroy_unique(cache);
    free(reader_thread);
}


void heatmap_LRU_hit_rate_start_time_end_time(gpointer data, gpointer user_data){

    guint64 i, j, hit_count, miss_count;
    struct multithreading_params_heatmap* params = (struct multithreading_params_heatmap*) user_data;
    READER* reader_thread = copy_reader(params->reader);
    GArray* break_points = params->break_points;
    guint64* progress = params->progress;
    draw_dict* dd = params->dd;
    guint64 cache_size = (guint64)params->cache->size;
    gint* last_access = reader_thread->last_access;
    long long* reuse_dist = reader_thread->reuse_dist;
    
    int order = GPOINTER_TO_INT(data)-1;
    guint64 real_start = g_array_index(break_points, guint64, order);
    
    hit_count = 0;
    miss_count = 0;
    
    
    // create cache line struct and initialization
    cache_line* cp = (cache_line*)malloc(sizeof(cache_line));
    cp->op = -1;
    cp->size = -1;
    cp->valid = TRUE;

    skip_N_elements(reader_thread, g_array_index(break_points, guint64, order));

    for (i=order; i<break_points->len-1; i++){
        
        for(j=g_array_index(break_points, guint64, i); j< g_array_index(break_points, guint64, i+1); j++){
            if (reuse_dist[j] == -1)
                miss_count ++;
            else if (last_access[j] - (long long)(j - real_start) <= 0 && reuse_dist[j] < (long long)cache_size)
                hit_count ++;
            else
                miss_count ++;
//            printf("last access %d, j %lu, real_start %lu, reuse dist %lld, cache_size %lu, hitcount: %lu\n", last_access[j], j, real_start, reuse_dist[j], cache_size, hit_count);
        }
        dd->matrix[order][i] = (float)(hit_count)/(hit_count+miss_count);
    }
    
    for(j=g_array_index(break_points, guint64, break_points->len - 1); j<(guint64)reader_thread->total_num; j++){
        if (reuse_dist[j] == -1)
            miss_count ++;
        else if (last_access[j] - (j - real_start) <=0 && (guint64)reuse_dist[j] < cache_size)
            hit_count ++;
        else
            miss_count ++;
    }
    dd->matrix[order][i] = (float)(hit_count)/(hit_count+miss_count);
    
    // clean up
    g_mutex_lock(&(params->mtx));
    (*progress) ++ ;
    g_mutex_unlock(&(params->mtx));
    free(cp);
    if (reader_thread->type != 'v')
        close_reader(reader_thread);
    free(reader_thread);
}




