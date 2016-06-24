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


struct multithreading_params_heatmap{
    READER* reader;
    struct cache* cache;
    int order;
    GArray* break_points;
    draw_dict* dd;
    guint64* progress;
    GMutex mtx;
    double log_base;
};


void free_draw_dict(draw_dict* dd);




GSList* get_last_access_dist_seq(READER* reader, void (*funcPtr)(READER*, cache_line*));

draw_dict* heatmap(READER* reader, struct_cache* cache, char mode, guint64 time_interval, int plot_type, int num_of_threads);
draw_dict* differential_heatmap(READER* reader, struct_cache* cache1, struct_cache* cache2, char mode, guint64 time_interval, int plot_type, int num_of_threads);
draw_dict* heatmap_rd_distribution(READER* reader, char mode, long time_interval, int num_of_threads, int CDF);


GArray* gen_breakpoints_virtualtime(READER* reader, guint64 time_interval);
GArray* gen_breakpoints_realtime(READER* reader, guint64 time_interval);



/* heatmap_thread */
void heatmap_LRU_hit_rate_start_time_end_time_thread(gpointer data, gpointer user_data);
void heatmap_nonLRU_hit_rate_start_time_end_time_thread(gpointer data, gpointer user_data);
void heatmap_rd_distribution_thread(gpointer data, gpointer user_data);
void heatmap_rd_distribution_CDF_thread(gpointer data, gpointer user_data);



#endif /* heatmap_h */
