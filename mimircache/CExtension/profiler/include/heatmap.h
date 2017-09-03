//
//  heatmap.h
//  heatmap
//
//  Created by Juncheng on 5/24/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef heatmap_h
#define heatmap_h

#include <stdio.h>
#include <stdlib.h>
#include <glib.h>
#include <unistd.h> 
#include "reader.h"
#include "glib_related.h" 
#include "cache.h" 
#include "LRUProfiler.h"
#include "Optimal.h"
#include "FIFO.h"
#include "utils.h"
#include "const.h"
#include <math.h>




#ifdef __cplusplus
extern "C"
{
#endif


typedef struct {
    /* 
     the data in matrix is stored in column based for performance consideration
     because in each computation thread, the result is stored on one column 
     think it as the x, y coordinates 
     */
    
    guint64 xlength;
    guint64 ylength;
    double** matrix;
    
}draw_dict;


typedef struct _multithreading_params_heatmap{
    reader_t* reader;
    struct cache* cache;
    int order;
    GArray* break_points;
    draw_dict* dd;
    guint64* progress;
    GMutex mtx;
    double log_base;
}mt_params_hm_t;


typedef enum _heatmap_type{
    hit_ratio_start_time_end_time,
    hit_ratio_start_time_cache_size,
    avg_rd_start_time_end_time,
    cold_miss_count_start_time_end_time,
    rd_distribution,
    future_rd_distribution,
    dist_distribution,          // this one is not using reuse distance, instead using distance to last access
    rt_distribution,            // real time distribution, use integer for time 
    rd_distribution_CDF
}heatmap_type_e;




double get_log_base(guint64 max, guint64 expect_result); 


void free_draw_dict(draw_dict* dd);




GSList* get_last_access_dist_seq(reader_t* reader,
                                 void (*funcPtr)(reader_t*, cache_line*));

draw_dict* heatmap(reader_t* reader, struct_cache* cache, char mode,
                   gint64 time_interval, gint64 num_of_pixels,
                   heatmap_type_e plot_type, int num_of_threads);
draw_dict* differential_heatmap(reader_t* reader, struct_cache* cache1,
                                struct_cache* cache2, char mode,
                                gint64 time_interval, gint64 num_of_pixels,
                                heatmap_type_e plot_type, int num_of_threads);
draw_dict* heatmap_rd_distribution(reader_t* reader, char mode,
                                   int num_of_threads, int CDF);


GArray* gen_breakpoints_virtualtime(reader_t* reader, gint64 time_interval,
                                    gint64 num_of_pixels);
GArray* gen_breakpoints_realtime(reader_t* reader, gint64 time_interval,
                                 gint64 num_of_pixels);



/* heatmap_thread */
void heatmap_LRU_hit_ratio_start_time_end_time_thread(gpointer data,
                                                     gpointer user_data);
void heatmap_nonLRU_hit_ratio_start_time_end_time_thread(gpointer data,
                                                        gpointer user_data);
void heatmap_rd_distribution_thread(gpointer data, gpointer user_data);
void heatmap_rd_distribution_CDF_thread(gpointer data, gpointer user_data);


#ifdef __cplusplus
}
#endif

#endif /* heatmap_h */
