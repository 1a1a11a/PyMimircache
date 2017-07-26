//
//  mimir.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef MIMIR_h
#define MIMIR_h


#include <math.h> 
#include "cache.h" 
#include "LRU.h"
#include "FIFO.h" 
#include "LFUFast.h"
#include "Optimal.h" 
#include "AMP.h" 


#ifdef __cplusplus
extern "C"
{
#endif


typedef gint64 TS_REPRESENTATION;


#define MINING_THRESHOLD  5120                      // when there are MINING_THRESHOLD entries longer than min_support, mining
#define PREFETCH_TABLE_SHARD_SIZE 2000
#define RECORDING_TABLE_MAXIMAL 0.02



#define GET_ROW_IN_RECORDING_TABLE(param, row_num)          \
                        ((param)->record_mining_struct->recording_table + (row_num) * (param)->record_mining_struct->recording_table_row_len)

#define GET_CURRENT_ROW_IN_RECORDING_TABLE(param)           \
                                        ((param)->record_mining_struct->recording_table + \
                                        (param)->record_mining_struct->recording_table_pointer * \
                                        (param)->record_mining_struct->recording_table_row_len)

#define GET_ROW_IN_MINING_TABLE(param, row_num)             \
                            ( ((TS_REPRESENTATION*)(param)->record_mining_struct->mining_table->data) + \
                            (param)->record_mining_struct->mining_table_row_len * (row_num))



/************ macros for getting and setting time stamp in 64-bit int   ************/
#define NUM_OF_TS(param)        ((gint)((param)>>60))

#define _GET_NTH_TS(param, n)    (((param)>>(15*(4-(n)))) & ((1UL<<15)-1) )
#define GET_NTH_TS(row, n)       ( (gint)_GET_NTH_TS( (*( (gint64*)(row) + 1 + (gint)(floor((n)/4)) )), ((n)%4) )  )




// clear the nth position, then set it
#define SET_NTH_TS(list, n, ts) ( ((list) & ( ~(((1UL<<15)-1)<<(15*(4-(n)))))) | ( ((ts) & ((1UL<<15)-1))<<(15*(4-(n))) ))
#define ADD_TS(list, ts)        ( ((SET_NTH_TS((list), (NUM_OF_TS((list))+1), (ts))) & ((1UL<<60)-1)) | ( (NUM_OF_TS((list))+1UL)<<60))






/** illustration on use gint64 to store 4 timestamps 
    the first four bits(0-3) are used as indicator,
    current indicator: use only last three bits as length of timestamps, 0~4. 
                        the first bit is not used because of uncertain behavior of
                        right shift, can be arithmetic shift or logical shift.
    the rest bits are divided by 4 and each used for one timestamp,
    4~18 bits are used as first timestamp,
    19~33, 34~48, 49~63 are used as second, third, fourth ts.
 **/


typedef enum _recording_loc{
    miss=0,     // this is the default, change order will have effect 
    evict,
    miss_evict,
    each_req,
}recording_loc_e;




struct MIMIR_init_params{
    char *cache_type;
    gint item_set_size;
    gint max_support;
    gint min_support;
    gint confidence;
    gint prefetch_list_size;
    
    recording_loc_e recording_loc;
    
//    gdouble training_period;
//    gchar training_period_type;
    
    gint block_size;
    gdouble max_metadata_size;
    gint cycle_time;                    /* when a prefetched element is going to be evicted,
                                            but haven't been accessed, how many chances we 
                                            give to it before it is really evicted 
                                            1: no cycle, no more chance, otherwise give cycle_time-1 chances 
                                         **/
    
    int mining_threshold;
    
    gint sequential_type;               /** 0: no sequential_prefetching,
                                         *  1: simple sequential_prefetching,
                                         *  2: AMP
                                         **/
    gint sequential_K;
    gint AMP_pthreshold;
    gint output_statistics;
};


struct MIMIR_params{
    struct_cache* cache;
    
    gint item_set_size;
    gint max_support;                       // allow to reach max support 
    gint min_support;
    gint confidence;                        // use the difference in length of sequence as confidence 
    gint cycle_time;
    gint mining_table_size;
    int mining_threshold; 
    recording_loc_e recording_loc;


    gint prefetch_list_size;
    gint block_size;                        // this should be equal to block_unit_size in cache->core 
    gint64 max_metadata_size;               // in bytes
    gint64 current_metadata_size;           // in bytes
    
    struct recording_mining_struct *record_mining_struct;
    
    
//    gchar training_period_type;

    
    GHashTable *prefetch_hashtable;             // request -> index in prefetch_table_array 
    gint32 current_prefetch_table_pointer;
    gboolean prefetch_table_fully_allocatd;    /* a flag indicates whether prefetch table is full or not
                                                    if it is full, we are replacing using a FIFO policy, 
                                                    and we don't allocate prefetch table shard when it 
                                                    reaches end of current shard
                                                */
    gint64 **prefetch_table_array;             // two dimension array

    guint64 ts;
    

    
    gint sequential_type; 
    gint sequential_K;
    
    
    // for statistics
    gint output_statistics;                 // a flag for turning on statistics analysis 
    GHashTable *prefetched_hashtable_mimir;
    guint64 hit_on_prefetch_mimir;
    guint64 num_of_prefetch_mimir;
    
    GHashTable *prefetched_hashtable_sequential;
    guint64 hit_on_prefetch_sequential;
    guint64 num_of_prefetch_sequential;

    guint64 num_of_check;
};



struct recording_mining_struct{
    GHashTable *hashtable;
    gint64* recording_table;                // N*(min_support/4+1) array, N is number of entries, extra one is for label
    gint8 recording_table_row_len;          // (min_support/4+1)
    gint8 mining_table_row_len;             // (max_support/4+1)
    gint64 num_of_rows_in_recording_table;
    gint64 recording_table_pointer;
    GArray* mining_table;                   // mining_threshold array, maybe can adaptively increase
    gint num_of_entry_available_for_mining;
};





extern gboolean MIMIR_check_element(struct_cache* MIMIR, cache_line* cp);
extern gboolean MIMIR_add_element(struct_cache* MIMIR, cache_line* cp);

extern void     __MIMIR_update_element(struct_cache* MIMIR, cache_line* cp);
extern void     __MIMIR_insert_element(struct_cache* MIMIR, cache_line* cp);
extern void     __MIMIR_evict_element(struct_cache* MIMIR, cache_line* cp);
// extern void*    __MIMIR__evict_with_return(struct_cache* MIMIR, cache_line* cp);



extern void     MIMIR_destroy(struct_cache* MIMIR);
extern void     MIMIR_destroy_unique(struct_cache* MIMIR);

extern void     __MIMIR_mining(struct_cache* MIMIR);
extern void     __MIMIR_aging(struct_cache* MIMIR);
extern void     mimir_add_to_prefetch_table(struct_cache* MIMIR, gpointer gp1, gpointer gp2);


struct_cache*   MIMIR_init(guint64 size, char data_type, int block_size, void* params); 


extern void     prefetch_node_destroyer(gpointer data);
extern void     prefetch_array_node_destroyer(gpointer data);


void prefetch_hashmap_count_length (gpointer key, gpointer value, gpointer user_data);


extern void MIMIR_remove_element(struct_cache* cache, void* data_to_remove);
extern gint64 MIMIR_get_size(struct_cache* cache);


#ifdef __cplusplus
}
#endif


#endif
