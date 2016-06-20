

#include "heatmap.h" 


draw_dict* heatmap_LRU(READER* reader, struct_cache* cache, char mode, long time_interval, int plot_type, int num_of_threads);
draw_dict* heatmap_nonLRU(READER* reader, struct_cache* cache, char mode, long time_interval, int plot_type, int num_of_threads);
draw_dict* heatmap_hit_rate_start_time_end_time(READER* reader, struct_cache* cache, char mode, long time_interval, int plot_type, int num_of_threads);


draw_dict* heatmap(READER* reader, struct_cache* cache, char mode, guint64 time_interval, int plot_type, int num_of_threads){
    
    if (mode == 'v')
        gen_breakpoints_virtualtime(reader, time_interval);
    else if (mode == 'r')
        gen_breakpoints_realtime(reader, time_interval);
    else{
        printf("unsupported mode: %c\n", mode);
        exit(1);
    }

    // check cache is LRU or not
    if (cache->type == e_LRU){
        return heatmap_LRU(reader, cache, mode, time_interval, plot_type, num_of_threads);
    }
    else{
        return heatmap_nonLRU(reader, cache, mode, time_interval, plot_type, num_of_threads);
    }
}


draw_dict* heatmap_LRU(READER* reader, struct_cache* cache, char mode, long time_interval, int plot_type, int num_of_threads){

    get_reuse_dist_seq(reader, 0, -1);
        
    if (plot_type == hit_rate_start_time_end_time){
        GSList* last_access_gslist = get_last_access_dist_seq(reader, read_one_element);
        reader->last_access = (gint*) malloc(sizeof(gint)*reader->total_num);
        GSList* sl_node = last_access_gslist;
        guint64 counter = reader->total_num-1;
        reader->last_access[counter--] = GPOINTER_TO_INT(sl_node->data);
        while(sl_node->next){
            sl_node = sl_node->next;
            reader->last_access[counter--] = GPOINTER_TO_INT(sl_node->data);
        }
        g_slist_free(last_access_gslist);
        return heatmap_hit_rate_start_time_end_time(reader, cache, mode, time_interval, plot_type, num_of_threads);
    }
    
    
    
    else if (plot_type == hit_rate_start_time_cache_size){
        
        
        
    }
    
    else if (plot_type == avg_rd_start_time_end_time){
        
        
        
    }

    else if (plot_type == cold_miss_count_start_time_end_time){
        
        
        
    }
    else if (plot_type == rd_distribution){
        
        
        
    }
    else if (plot_type == hit_rate_start_time_cache_size){
        
        
        
    }
    else {
        printf("unknown plot type\n");
        exit(1);
    }
    
    return NULL;
}


draw_dict* heatmap_nonLRU(READER* reader, struct_cache* cache, char mode, long time_interval, int plot_type, int num_of_threads){
    if (plot_type == hit_rate_start_time_end_time){
        return heatmap_hit_rate_start_time_end_time(reader, cache, mode, time_interval, plot_type, num_of_threads);
    }
    else if (plot_type == hit_rate_start_time_cache_size){
        
        
        
    }
    
    else if (plot_type == avg_rd_start_time_end_time){
        
        
        
    }
    
    else if (plot_type == cold_miss_count_start_time_end_time){
        
        
        
    }
    else if (plot_type == rd_distribution){
        get_reuse_dist_seq(reader, 0, -1);

        
        
    }
    else if (plot_type == hit_rate_start_time_cache_size){
        
        
        
    }
    else {
        printf("unknown plot type\n");
        exit(1);
    }
    
    return NULL;
}




draw_dict* heatmap_hit_rate_start_time_end_time(READER* reader, struct_cache* cache, char mode, long time_interval, int plot_type, int num_of_threads){

    guint i;
    guint64 progress = 0;
    
    GArray* break_points;
    if (mode == 'v')
        break_points = reader->break_points_v;
    else
        break_points = reader->break_points_r;
        

    // create draw_dict storage
    draw_dict* dd = (draw_dict*) malloc(sizeof(draw_dict));
    dd->xlength = break_points->len;
    dd->ylength = break_points->len;
    dd->matrix = (double**) malloc(break_points->len * sizeof(double*));
    for (i=0; i<break_points->len; i++)
        dd->matrix[i] = (double*) calloc(break_points->len, sizeof(double));
    

    // build parameters
    struct multithreading_params_heatmap* params = malloc(sizeof(struct multithreading_params_heatmap) );
    params->reader = reader;
    params->break_points = break_points;
    params->cache = cache;
    params->dd = dd;
    params->progress = &progress;
    g_mutex_init(&(params->mtx));

    
    // build the thread pool
    GThreadPool * gthread_pool;
    if (cache->type == e_LRU)
        gthread_pool = g_thread_pool_new ( (GFunc) heatmap_LRU_hit_rate_start_time_end_time_thread, (gpointer)params, num_of_threads, TRUE, NULL);
    else
        gthread_pool = g_thread_pool_new ( (GFunc) heatmap_nonLRU_hit_rate_start_time_end_time_thread, (gpointer)params, num_of_threads, TRUE, NULL);
    
    if (gthread_pool == NULL)
        g_error("cannot create thread pool in heatmap\n");
    
    
    // send data to thread pool and begin computation
    for (i=0; i<break_points->len; i++){
        if ( g_thread_pool_push (gthread_pool, GINT_TO_POINTER(i+1), NULL) == FALSE)    // +1 otherwise, 0 will be a problem
            g_error("cannot push data into thread in generalprofiler\n");
    }
    
    while ( progress < break_points->len ){
        printf("%.2f%%\n", ((double)progress)/break_points->len*100);
        sleep(1);
        printf("\033[A\033[2K\r");
    }
    
    g_thread_pool_free (gthread_pool, FALSE, TRUE);
    

    g_mutex_clear(&(params->mtx)); 
    free(params);
    // needs to free draw_dict later
    return dd;
}
    

draw_dict* heatmap_rd_distribution(READER* reader, char mode, long time_interval, int num_of_threads){
    guint i;
    guint64 progress = 0;
    
    GArray* break_points;
    if (mode == 'v')
        break_points = reader->break_points_v;
    else
        break_points = reader->break_points_r;
    
    // this is used to make sure length of x and y are approximate same, not different by too much
    double log_base = get_log_base(reader->max_reuse_dist, break_points->len);
    
    
    // create draw_dict storage
    draw_dict* dd = (draw_dict*) malloc(sizeof(draw_dict));
    dd->xlength = break_points->len;
    
    // the last one is used for store cold miss; rd=0 and rd=1 are combined at first bin (index=0) 
    dd->ylength = (long) ceil(log(reader->max_reuse_dist)/log(log_base));
    dd->matrix = (double**) malloc(break_points->len * sizeof(double*));
    for (i=0; i<break_points->len; i++)
        dd->matrix[i] = (double*) calloc(break_points->len, sizeof(double));
    
    
    // build parameters
    struct multithreading_params_heatmap* params = malloc(sizeof(struct multithreading_params_heatmap) );
    params->reader = reader;
    params->break_points = break_points;
    params->dd = dd;
    params->progress = &progress;
    params->log_base = log_base;
    g_mutex_init(&(params->mtx));
    
    
    // build the thread pool
    GThreadPool * gthread_pool;
    gthread_pool = g_thread_pool_new ( (GFunc) heatmap_rd_distribution_thread, (gpointer)params, num_of_threads, TRUE, NULL);
    
    if (gthread_pool == NULL)
        g_error("cannot create thread pool in heatmap rd_distribution\n");
    
    
    // send data to thread pool and begin computation
    for (i=0; i<break_points->len; i++){
        if ( g_thread_pool_push (gthread_pool, GINT_TO_POINTER(i+1), NULL) == FALSE)    // +1 otherwise, 0 will be a problem
            g_error("cannot push data into thread in generalprofiler\n");
    }
    
    while ( progress < break_points->len ){
        printf("%.2lf%%\n", ((double)progress)/break_points->len*100);
        sleep(1);
        printf("\033[A\033[2K\r");
    }
    
    g_thread_pool_free (gthread_pool, FALSE, TRUE);
    
    
    g_mutex_clear(&(params->mtx));
    free(params);
    return dd;
}



draw_dict* differential_heatmap(READER* reader, struct_cache* cache1, struct_cache* cache2, char mode, guint64 time_interval, int plot_type, int num_of_threads){
    
    if (mode == 'v')
        gen_breakpoints_virtualtime(reader, time_interval);
    else if (mode == 'r')
        gen_breakpoints_realtime(reader, time_interval);
    else{
        printf("unsupported mode: %c\n", mode);
        exit(1);
    }

    draw_dict *draw_dict1, *draw_dict2;
    // check cache is LRU or not
    if (cache1->type == e_LRU){
        draw_dict1 = heatmap_LRU(reader, cache1, mode, time_interval, plot_type, num_of_threads);
    }
    else{
        draw_dict1 = heatmap_nonLRU(reader, cache1, mode, time_interval, plot_type, num_of_threads);
    }

    if (cache2->type == e_LRU){
        draw_dict2 = heatmap_LRU(reader, cache2, mode, time_interval, plot_type, num_of_threads);
    }
    else{
        draw_dict2 = heatmap_nonLRU(reader, cache2, mode, time_interval, plot_type, num_of_threads);
    }

    
    guint64 i, j;
    for (i=0; i<draw_dict1->xlength; i++)
        for (j=0; j<draw_dict1->ylength; j++){
//            if (draw_dict1->matrix[i][j] > 1 || draw_dict1->matrix[i][j] < 0)
//                printf("ERROR -1 %ld, %ld: %f\n", i, j, draw_dict1->matrix[i][j]);
//            if (draw_dict2->matrix[i][j] > 1 || draw_dict2->matrix[i][j] < 0)
//                printf("ERROR -2 %ld, %ld: %f\n", i, j, draw_dict2->matrix[i][j]);
            draw_dict2->matrix[i][j] = draw_dict2->matrix[i][j] - draw_dict1->matrix[i][j];
        }
    free_draw_dict(draw_dict1);
    return draw_dict2;
}


void free_draw_dict(draw_dict* dd){
    int i;
    for (i=0; i<dd->xlength; i++){
        free(dd->matrix[i]);
    }
    free(dd->matrix);
    free(dd);
}


#include "reader.h"
#include "FIFO.h"
#include "Optimal.h"

int main(int argc, char* argv[]){
# define CACHESIZE 2000
# define BIN_SIZE 200


    printf("test_begin!\n");

    READER* reader = setup_reader(argv[1], 'v');

//    struct_cache* cache = fifo_init(CACHESIZE, 'v', NULL);

    struct optimal_init_params init_params = {.reader=reader, .next_access=NULL};
    struct_cache* optimal = optimal_init(CACHESIZE, 'v', (void*)&init_params);
    
    struct_cache* cache;
    cache = (struct_cache*)calloc(1, sizeof(struct_cache));
    cache->type = e_LRU;
    cache->size = CACHESIZE;
    cache->data_type = reader->type;

    
//    struct_cache* cache = (struct_cache*) malloc(sizeof(struct_cache));
//    cache->size = CACHESIZE;


//    printf("after initialization, begin profiling\n");
//    return_res** res = profiler(reader, cache, 4, BIN_SIZE);
    draw_dict* dd = differential_heatmap(reader, cache, optimal, 'r', 1000000, hit_rate_start_time_end_time, 8);

    guint64 i, j;
    for (i=0; i<dd->xlength; i++){
        for (j=0; j<dd->ylength; j++)
//            printf("%llu, %llu: %f\n", i, j, dd->matrix[i][j]);
            ;
    }

    printf("computation finished\n");
    free_draw_dict(dd);
//    cache->destroy(cache);
    free(cache);
    optimal->destroy(optimal);
    close_reader(reader);
    
    printf("test_finished!\n");
    sleep(10);
    
    return 0;
}



