//
//  MS1.h
//  MS1cache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef MS1_h
#define MS1_h


#include <math.h> 
#include "cache.h" 
#include "LRU.h"
#include "FIFO.h" 
#include "Optimal.h" 
#include "AMP.h" 
#include "mimir.h"


#define CACHE_LINE_LABEL_SIZE2 16
#define INIT_PREFETCH_SCORE 8




//struct prefetch_hashtable_value{
//    GPtrArray* pArray;
//    
//};

struct MS1_prefetch_hashtable_valuelist_element{
    gint score;                 // how confident this element should be prefetched
    gpointer item_p;
//    char item[cache_line_label_size2];       /* THIS WILL CAUSE LARGE OVERHEAD ##################### */
//    char type;                              /* type of content can be either guint64(l) or char*(c) */
};

struct MS1_training_data_node{
    gint32* array;              // use array makes more sense as the length would be very short,
                                // use list will take more space for the pointers
    gint8 length;
    gpointer item_p; 
};

struct MS1_init_params{
    char *cache_type;
    gint item_set_size;
    gint max_support;
    gint min_support;
    gint confidence;
    gint prefetch_list_size;
    gdouble training_period;
    gchar training_period_type;
    gint64 prefetch_table_size;
    gint sequential_type;               /** 0: no sequential_prefetching,
                                         *  1: simple sequential_prefetching,
                                         *  2: AMP
                                         **/
    gint sequential_K; 
    gint output_statistics;
};



struct MS1_params{
    struct_cache* cache;
    
    gint ave_length;
    gint item_set_size;
    gint max_support;                       // allow to reach max support 
    gint min_support;
    gint confidence;                        // use the difference in length of sequence as confidence 
    
    gint prefetch_list_size;
    
    
    GHashTable *hashtable_for_training;      // from request to linkedlist node
    GHashTable *hashset_frequentItem;      // hashset for remembering frequent items
    GSList *training_data;       // a list of list of timestamps
    
    gdouble training_period;
    gchar training_period_type;
    gdouble last_train_time; 
    
    GHashTable *prefetch_hashtable;
    gint64 prefetch_table_size;

    guint64 ts;
    

    
    gint sequential_type; 
    gint sequential_K;
    
    
    // for statistics
    gint output_statistics;                 // a flag for turning on statistics analysis 
    GHashTable *prefetched_hashtable_MS1;
    guint64 hit_on_prefetch_MS1;
    guint64 num_of_prefetch_MS1;
    
    GHashTable *prefetched_hashtable_sequential;
    guint64 hit_on_prefetch_sequential;
    guint64 num_of_prefetch_sequential;

    guint64 num_of_check;
};



extern  void __MS1_insert_element(struct_cache* MS1, cache_line* cp);

extern  gboolean MS1_check_element(struct_cache* MS1, cache_line* cp);

extern  void __MS1_update_element(struct_cache* MS1, cache_line* cp);

extern  void __MS1_evict_element(struct_cache* MS1, cache_line* cp);

extern  gboolean MS1_add_element(struct_cache* MS1, cache_line* cp);


extern  void MS1_destroy(struct_cache* MS1);
extern  void MS1_destroy_unique(struct_cache* MS1);

extern void __MS1_mining(struct_cache* MS1);
extern void __MS1_aging(struct_cache* MS1);
extern void MS1_add_to_prefetch_table(struct_cache* MS1, gpointer gp1, gpointer gp2);


struct_cache* MS1_init(guint64 size, char data_type, void* params);


extern  void MS1_remove_element(struct_cache* cache, void* data_to_remove);
extern gpointer __MS1_evict_element_with_return(struct_cache* MS1, cache_line* cp);
extern guint64 MS1_get_size(struct_cache* cache);
extern void MS1_training_node_destroyer(gpointer data); 


#endif
