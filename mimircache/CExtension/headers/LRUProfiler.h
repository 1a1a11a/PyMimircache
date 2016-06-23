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


typedef struct count_array{
    // TODO use this instead of pure C array, in case trace file are is too large
    
    
}count_array;




long long* get_hit_count_seq(READER* reader, long size, long long begin, long long end);
double* get_hit_rate_seq(READER* reader, long size, long long begin, long long end);
double* get_miss_rate_seq(READER* reader, long size, long long begin, long long end);
long long* get_reuse_dist_seq(READER* reader, long long begin, long long end);
long long* get_rd_distribution(READER* reader, long long begin, long long end);
long long* get_future_reuse_dist(READER* reader, long long begin, long long end);


#endif /* LRUAnalyzer_h */
