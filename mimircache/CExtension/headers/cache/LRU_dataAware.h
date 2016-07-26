//
//  LRU_dataAware.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef LRU_dataAware_h
#define LRU_dataAware_h


#include "cache.h" 
#include "LRU.h" 
#include "LFU.h" 



#define TIME_THRESHOLD 1000000
#define SCAN_CHECKER_THRESHOLD 8
#define AVG_REQUEST_RATE_MAGNIFIER 1



struct LRU_dataAware_params{
    struct_cache* LRU;
    struct_cache* LFU; 
    
    gboolean in_scan;
    GQueue *time_gqueue;
    guint avg_request_rate;
    guint64 begin_real_time;
    guint64 total_request_num; 
    guint scan_checker;              // 0, 1, 2.. scan_checker_THRESHOLD; when reach scan_checker_THRESHOLD, we think it is scan
};



extern inline void __LRU_dataAware_insert_element(struct_cache* LRU_dataAware, cache_line* cp);

extern inline gboolean LRU_dataAware_check_element(struct_cache* cache, cache_line* cp);

extern inline void __LRU_dataAware_update_element(struct_cache* LRU_dataAware, cache_line* cp);

extern inline void __LRU_dataAware_evict_element(struct_cache* LRU_dataAware, cache_line* cp);

extern inline gboolean LRU_dataAware_add_element(struct_cache* cache, cache_line* cp);


extern inline void LRU_dataAware_destroy(struct_cache* cache);
extern inline void LRU_dataAware_destroy_unique(struct_cache* cache);


struct_cache* LRU_dataAware_init(guint64 size, char data_type, void* params);



extern inline void __LRU_dataAware_remove_element(struct_cache* cache, void* data_to_remove);



#endif