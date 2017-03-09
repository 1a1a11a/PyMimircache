//
//  LRUAnalyzer.h
//  LRUAnalyzer
//
//  Created by Juncheng on 5/24/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef LRUAnalyzer_h
#define LRUAnalyzer_h

#include <stdio.h>
#include <stdlib.h>
#include <glib.h>
#include "splay.h"
#include "reader.h"
#include "glib_related.h" 
#include "const.h" 
#include "pqueue.h"




guint64* get_hit_count_seq   (reader_t* reader, gint64 size, gint64 begin, gint64 end);
double* get_hit_rate_seq     (reader_t* reader, gint64 size, gint64 begin, gint64 end);
double* get_miss_rate_seq    (reader_t* reader, gint64 size, gint64 begin, gint64 end);
gint64* get_reuse_dist_seq   (reader_t* reader, gint64 begin, gint64 end);
//guint64* get_rd_distribution (reader_t* reader, gint64 begin, gint64 end);
gint64* get_future_reuse_dist(reader_t* reader, gint64 begin, gint64 end);

GQueue * cal_best_LRU_cache_size(reader_t* reader, unsigned int num, int force_spacing, int cut_off_divider);

#endif /* LRUAnalyzer_h */
