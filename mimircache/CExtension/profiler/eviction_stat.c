//
//  eviction_stat.c
//  mimircache
//
//  Created by Juncheng on 5/24/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "eviction_stat.h" 


#include "reader.h"
#include "FIFO.h"
#include "Optimal.h"
#include "LRU_K.h"
#include "LRU.h"

#ifdef __cplusplus
extern "C"
{
#endif


static gint64* get_eviction_reuse_dist(reader_t* reader, struct_cache* optimal);
static gint64* get_eviction_freq(reader_t* reader, struct_cache* optimal, gboolean accumulative);
//static gdouble* get_eviction_relative_freq(READER* reader, struct_cache* optimal);
static inline sTree* process_one_element_eviction_reuse_dist(cache_line* cp, sTree* splay_tree, GHashTable* hash_table, guint64 ts, gint64* reuse_dist, gpointer evicted);



static void traverse_trace(reader_t* reader, struct_cache* cache){
    /** this function traverse the trace file, add each request to cache, just like a cache simulation 
     *
     **/
    
    // create cache line struct and initialization
    cache_line* cp = new_cacheline();
    
    cp->type = cache->core->data_type;
    cp->block_unit_size = (size_t) reader->base->block_unit_size;
    
    gboolean (*add_element)(struct cache*, cache_line* cp);
    add_element = cache->core->add_element;
    
    read_one_element(reader, cp);
    while (cp->valid){
        add_element(cache, cp);
        read_one_element(reader, cp);
    }
    
    // clean up
    g_free(cp);
    reset_reader(reader);
    
}




gint64* eviction_stat(reader_t* reader_in, struct_cache* cache, evict_stat_type stat_type){
    /** this function first traverse the trace file to generate a list of evicted requests,
     ** then again traverse the trace file to obtain the statistics of evicted requests 
     **/ 
    
    // get cache eviction list
    cache->core->cache_debug_level = 1;
    cache->core->eviction_array_len = reader_in->base->total_num;
    if (reader_in->base->total_num == -1)
        get_num_of_cache_lines(reader_in);
    
    if (reader_in->base->data_type == 'l')
        cache->core->eviction_array = g_new0(guint64, reader_in->base->total_num);
    else
        cache->core->eviction_array = g_new0(gchar*, reader_in->base->total_num);
    
    traverse_trace(reader_in, cache);
    // done get eviction list
    
    
    if (stat_type == evict_reuse_dist){
        return get_eviction_reuse_dist(reader_in, cache);
    }
    else if (stat_type == evict_freq){
        return get_eviction_freq(reader_in, cache, FALSE);
    }
    else if (stat_type == evict_freq_accumulatve){
        return get_eviction_freq(reader_in, cache, TRUE);
    }
    else if (stat_type == evict_data_classification){
        return NULL;
    }
    else{
        ERROR("unsupported stat type\n");
        exit(1);
    }
}


gdouble* eviction_stat_over_time(reader_t* reader_in, char mode, guint64 time_interval, guint64 cache_size, char* stat_type){

    if (mode == 'r')
        gen_breakpoints_realtime(reader_in, time_interval, -1);
    else
        gen_breakpoints_virtualtime(reader_in, time_interval, -1);
    


    return NULL;
}


gint64* get_eviction_freq(reader_t* reader, struct_cache* optimal, gboolean accumulative){
    /** if insert then evict, its freq should be 1,
        in other words, the smallest freq should be 1, 
        if there is no eviction at ts, then it is -1.
     **/
    
    guint64 ts = 0;
    
    gpointer eviction_array = optimal->core->eviction_array;
    
    gint64 * freq_array = g_new0(gint64, reader->base->total_num);
    
    
    // create cache line struct and initializa
    cache_line* cp = new_cacheline();
    cp->type = reader->base->data_type;
    
    // create hashtable
    GHashTable * hash_table;
    if (reader->base->data_type == 'l'){
        //        cp->type = 'l';
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else if (reader->base->data_type == 'c' ){
        //        cp->type = 'c';
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else{
        ERROR("does not recognize reader data type %c\n", reader->base->data_type);
        abort();
    }
    
    gpointer gp;
    read_one_element(reader, cp);

    while (cp->valid){
        gp = g_hash_table_lookup(hash_table, cp->item_p);
        if (gp == NULL){
            // first time access
            gint64 *value = g_new(gint64, 1);
            *value = 1;
            if (cp->type == 'c'){
                g_hash_table_insert(hash_table, g_strdup((gchar*)(cp->item_p)), (gpointer)(value));
            }
            else if (cp->type == 'l'){
                gint64* key = g_new(gint64, 1);
                *key = *(guint64*)(cp->item_p);
                g_hash_table_insert(hash_table, (gpointer)(key), (gpointer)(value));
            }
            else{
                printf("unknown cache line content type: %c\n", cp->type);
                exit(1);
            }
        }
        else{
            *(gint64*)gp = *(gint64*)gp + 1;
        }
        read_one_element(reader, cp);

        
        // get freq of evicted item
        if (cp->type == 'c'){
            if (((gchar**)eviction_array)[ts] == NULL)
                freq_array[ts++] = -1;
            else{
                gp = g_hash_table_lookup(hash_table, ((gchar**)eviction_array)[ts]);
                freq_array[ts++] = *(gint64*)gp;
                
                // below line should be enabled only when we want to clear the freq count after eviction
                if (!accumulative)
                    *(gint64*)gp = 0;
            }
        }
        else if (cp->type == 'l'){
            if ( *((guint64*)eviction_array+ts) == 0)
                freq_array[ts++] = -1;
            else{
                gp = g_hash_table_lookup(hash_table, ((guint64*)eviction_array + ts));
                freq_array[ts++] = *(gint64*)gp;

                // below line should be enabled only when we want to clear the freq count after eviction
                if (!accumulative)
                    *(gint64*)gp = 0;
            }
        }
    }
    
    destroy_cacheline(cp);
    g_hash_table_destroy(hash_table);
    reset_reader(reader);
    return freq_array;
}






static gint64* get_eviction_reuse_dist(reader_t* reader, struct_cache* optimal){
    /*
     * TODO: might be better to split return result, in case the hit rate array is too large
     * Is there a better way to do this? this will cause huge amount memory
     * It is the user's responsibility to release the memory of hit count array returned by this function
     */
    
    guint64 ts = 0;
    gint64 reuse_dist;
    
    gpointer eviction_array = optimal->core->eviction_array;
    
    gint64 * reuse_dist_array = g_new0(gint64, reader->base->total_num);
    
    
    // create cache line struct and initializa
    cache_line* cp = new_cacheline();
    cp->type = reader->base->data_type;
    
    // create hashtable
    GHashTable * hash_table;
    if (reader->base->data_type == 'l'){
        //        cp->type = 'l';
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else if (reader->base->data_type == 'c' ){
        //        cp->type = 'c';
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else{
        ERROR("does not recognize reader data type %c\n", reader->base->data_type);
        abort();
    }
    
    // create splay tree
    sTree* splay_tree = NULL;
    

    read_one_element(reader, cp);
    
    if (cp->type == 'l'){
        while (cp->valid){
            splay_tree = process_one_element_eviction_reuse_dist(cp, splay_tree, hash_table, ts, &reuse_dist,
                                                      (gpointer) ((guint64*)eviction_array + ts) );
            reuse_dist_array[ts] = reuse_dist;
            read_one_element(reader, cp);
            ts++;
        }
    }
    else{
        while (cp->valid){
            splay_tree = process_one_element_eviction_reuse_dist(cp, splay_tree, hash_table, ts, &reuse_dist,
                                                      (gpointer) (((gchar**)eviction_array)[ts]) );
            reuse_dist_array[ts] = reuse_dist;
            read_one_element(reader, cp);
            ts++;
        }
    }
    
    
    // clean up
    destroy_cacheline(cp);
    g_hash_table_destroy(hash_table);
    free_sTree(splay_tree);
    reset_reader(reader);
    return reuse_dist_array;
}



static inline sTree* process_one_element_eviction_reuse_dist(cache_line* cp, sTree* splay_tree, GHashTable* hash_table, guint64 ts, gint64* reuse_dist, gpointer evicted){
    gpointer gp;
    
    gp = g_hash_table_lookup(hash_table, cp->item_p);
    
    sTree* newtree = splay_tree;
    if (gp == NULL){
        // first time access
        newtree = insert(ts, splay_tree);
        gint64 *value = g_new(gint64, 1);
        *value = ts;
        if (cp->type == 'c')
            g_hash_table_insert(hash_table, g_strdup((gchar*)(cp->item_p)), (gpointer)value);
        
        else if (cp->type == 'l'){
            gint64* key = g_new(gint64, 1);
            *key = *(guint64*)(cp->item_p);
            g_hash_table_insert(hash_table, (gpointer)(key), (gpointer)value);
        }
        else{
            printf("unknown cache line content type: %c\n", cp->type);
            exit(1);
        }
    }
    else{
        // not first time access
        // save old ts, update ts in hashtable
        guint64 old_ts = *(guint64*)gp;
        *(guint64*)gp = ts;
        
        // update splay tree
        newtree = splay_delete(old_ts, newtree);
        newtree = insert(ts, newtree);
        
    }
    
    // get evicted reuse distance
    if (evicted){
        if (cp->type == 'l')
            if (*(guint64*)evicted == 0){
                *reuse_dist = -1;
                return newtree;
            }
        gp = g_hash_table_lookup(hash_table, evicted);
        guint64 old_ts = *(guint64*)gp;
        newtree = splay(old_ts, newtree);
        *reuse_dist = node_value(newtree->right);
    }
    else
        *reuse_dist = -1;

    
    return newtree;
}



#ifdef __cplusplus
}
#endif
