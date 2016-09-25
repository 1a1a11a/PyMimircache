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
#include "Optimal.h" 


#define CACHE_LINE_LABEL_SIZE2 16
#define INIT_PREFETCH_SCORE 8




//struct prefetch_hashtable_value{
//    GPtrArray* pArray;
//    
//};

struct prefetch_hashtable_valuelist_element{
    gint score;                 // how confident this element should be prefetched
    gpointer item_p;
//    char item[cache_line_label_size2];       /* THIS WILL CAUSE LARGE OVERHEAD ##################### */
//    char type;                              /* type of content can be either guint64(l) or char*(c) */
};

struct training_data_node{
    gint32* array;              // use array makes more sense as the length would be very short,
                                // use list will take more space for the pointers
    gint8 length;
    gpointer item_p; 
};

struct MIMIR_init_params{
    char *cache_type;
    gint item_set_size;
    gint max_support;
    gint min_support;
    gint confidence;
    gint prefetch_list_size;
    gdouble training_period;
    gchar training_period_type;
    gint output_statistics;
};



struct MIMIR_params{
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
    
    guint64 ts;
    
    gpointer last;
    char last_request_storage[CACHE_LINE_LABEL_SIZE2];
    
    
    // for statistics
    gint output_statistics;                 // a flag for turning on statistics analysis 
    GHashTable *prefetched_hashtable;
    guint64 hit_on_prefetch;
    guint64 num_of_prefetch;
    guint64 num_of_check;
};



extern  void __MIMIR_insert_element(struct_cache* MIMIR, cache_line* cp);

extern  gboolean MIMIR_check_element(struct_cache* MIMIR, cache_line* cp);

extern  void __MIMIR_update_element(struct_cache* MIMIR, cache_line* cp);

extern  void __MIMIR_evict_element(struct_cache* MIMIR, cache_line* cp);

extern  gboolean MIMIR_add_element(struct_cache* MIMIR, cache_line* cp);


extern  void MIMIR_destroy(struct_cache* MIMIR);
extern  void MIMIR_destroy_unique(struct_cache* MIMIR);

extern void __MIMIR_mining(struct_cache* MIMIR);
extern void __MIMIR_aging(struct_cache* MIMIR);
extern void add_to_prefetch_table(struct_cache* MIMIR, gpointer gp1, gpointer gp2);


struct_cache* MIMIR_init(guint64 size, char data_type, void* params);


void training_node_destroyer(gpointer data);
void prefetch_node_destroyer(gpointer data);
void prefetch_array_node_destroyer(gpointer data);


void
prefetch_hashmap_count_length (gpointer key, gpointer value, gpointer user_data); 


extern  void MIMIR_remove_element(struct_cache* cache, void* data_to_remove);
extern gpointer __MIMIR_evict_element_with_return(struct_cache* MIMIR, cache_line* cp);
extern guint64 MIMIR_get_size(struct_cache* cache);



#endif
