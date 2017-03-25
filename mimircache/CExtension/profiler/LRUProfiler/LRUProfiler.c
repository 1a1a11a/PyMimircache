//
//  LRUAnalyzer.c
//  LRUAnalyzer
//
//  Created by Juncheng on 5/25/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "LRUProfiler.h"



/** priority queue structs and def
 */

typedef struct node_pq_LRUProfiler
{
    pqueue_pri_t pri;
    guint data;
    size_t pos;
} node_pq_LRUProfiler;


static inline int
cmp_pri(pqueue_pri_t next, pqueue_pri_t curr){
    return (next.pri1 < curr.pri1);
}


static inline pqueue_pri_t
get_pri(void *a){
    return ((node_pq_LRUProfiler *) a)->pri;
}


static inline void
set_pri(void *a, pqueue_pri_t pri){
    ((node_pq_LRUProfiler *) a)->pri = pri;
}

static inline size_t
get_pos(void *a){
    return ((node_pq_LRUProfiler *) a)->pos;
}


static inline void
set_pos(void *a, size_t pos){
    ((node_pq_LRUProfiler *) a)->pos = pos;
}



static inline sTree*
process_one_element(cache_line* cp,
                    sTree* splay_tree,
                    GHashTable* hash_table,
                    guint64 ts,
                    gint64* reuse_dist);


guint64* get_hit_count_seq(reader_t* reader,
                           gint64 size,
                           gint64 begin,
                           gint64 end){
    /* get the hit count, if size==-1, then do all the counting, otherwise,
     * treat the ones with reuse distance larger than size as out of range,
     * and put it in the second to the last bucket of hit_count_array
     * in other words, 0~size(included) are for counting rd=0~size-1, size+1 is
     * out of range, size+2 is cold miss, so total is size+3 buckets
     */
    
    /*
     size: the size of cache, if passed -1, then choose maximum size
     begin: the begin position of trace(starting at 0), in other words, skip first N elements
     end: the end position(excluded) of trace, after end, all trace requests are not processed
     */

    guint64 ts=0;
    gint64 reuse_dist;
    guint64 * hit_count_array;
    
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    
    if (size == -1)
        size = reader->base->total_num;
    
    if (begin == -1)
        begin = 0;
    
    if (end == -1)
        end = reader->base->total_num+1;
    
    
    if (end-begin<size)
        size = end-begin;
    
    /* for a cache_size=size, we need size+1 bucket for size 0~size(included),
     * the last element(size+1) is used for storing count of reuse distance > size
     * if size==reader->base->total_num, then the last two is not used
     */
    hit_count_array = g_new0(guint64, size+3);
    
    
    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();
    
    // create hashtable
    GHashTable * hash_table;
    if (reader->base->type == 'v'){
        cp->type = 'l';
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else{
        cp->type = 'c';
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    
    // create splay tree
    sTree* splay_tree = NULL;
    
    
    if (begin != -1 && begin != 0)
        skip_N_elements(reader, begin);

    read_one_element(reader, cp);
    while (cp->valid){
        splay_tree = process_one_element(cp, splay_tree, hash_table, ts, &reuse_dist);
        if (reuse_dist == -1)
            hit_count_array[size+2] += 1;
        else if (reuse_dist>=size)
            hit_count_array[size+1] += 1;
        else
            hit_count_array[reuse_dist+1] += 1;
        if (ts >= (guint64)end-begin)
            break;
        read_one_element(reader, cp);
        ts++;
    }

    // clean up
    destroy_cacheline(cp);
    g_hash_table_destroy(hash_table);
    free_sTree(splay_tree);
    reset_reader(reader);
    return hit_count_array;
    
    
}

guint64* get_hit_count_seq_shards(reader_t* reader,
                                  gint64 size,
                                  double sample_ratio,
                                  gint64 correction){
    /* same as get_hit_count_seq, but using shards generated data,
     * get the hit count, if size==-1, then do all the counting, otherwise,
     * treat the ones with reuse distance larger than size as out of range,
     * and put it in the second to the last bucket of hit_count_array
     * in other words, 0~size(included) are for counting rd=0~size-1, size+1 is
     * out of range, size+2 is cold miss, so total is size+3 buckets
     */
    
    
    guint64 ts=0;
    gint64 reuse_dist;
    guint64 * hit_count_array;
    
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    
    if (size == -1)
        size = (gint64) (reader->base->total_num / sample_ratio);
    
    
    /* for a cache_size=size, we need size+1 bucket for size 0~size(included),
     * the last element(size+1) is used for storing count of reuse distance > size
     * if size==reader->base->total_num, then the last two is not used
     */
    hit_count_array = g_new0(guint64, size+3);
    
    
    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();
    
    // create hashtable
    GHashTable * hash_table;
    if (reader->base->type == 'v'){
        cp->type = 'l';
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else{
        cp->type = 'c';
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    
    // create splay tree
    sTree* splay_tree = NULL;
    
    read_one_element(reader, cp);
    while (cp->valid){
        splay_tree = process_one_element(cp, splay_tree, hash_table, ts, &reuse_dist);
        if (reuse_dist == -1)
            hit_count_array[size+2] += 1;
        else
            reuse_dist = (gint64)(reuse_dist / sample_ratio);
        
        if (reuse_dist>=size)
            hit_count_array[size+1] += 1;
        else
            hit_count_array[reuse_dist+1] += 1;
        read_one_element(reader, cp);
        ts++;
    }
    if (correction != 0){
        if (correction<0 && -correction > hit_count_array[0])
            hit_count_array[0] = 0;
        else
            hit_count_array[0] += correction;
    }
    
    
    // clean up
    destroy_cacheline(cp);
    g_hash_table_destroy(hash_table);
    free_sTree(splay_tree);
    reset_reader(reader);
    return hit_count_array;
    
    
}


double* get_hit_rate_seq_shards(reader_t* reader,
                                gint64 size,
                                double sample_ratio,
                                gint64 correction){
    int i=0;
    if (reader->base->total_num == -1)
        reader->base->total_num = get_num_of_cache_lines(reader);
    
    if (size == -1)
        size = (gint64) (reader->base->total_num / sample_ratio);
    
    if (reader->udata->hit_rate && size==reader->base->total_num)
        return reader->udata->hit_rate;
    
    
    guint64* hit_count_array = get_hit_count_seq_shards(reader, size, sample_ratio, correction);
    double total_num = (double)(reader->base->total_num);
    
    
    
    double* hit_rate_array = g_new(double, size+3);
    hit_rate_array[0] = hit_count_array[0]/total_num;
    for (i=1; i<size+1; i++){
        hit_rate_array[i] = hit_count_array[i]/total_num + hit_rate_array[i-1];
    }
    // larger than given cache size
    hit_rate_array[size+1] = hit_count_array[size+1]/total_num;
    // cold miss
    hit_rate_array[size+2] = hit_count_array[size+2]/total_num;
    
    g_free(hit_count_array);
    if (size==reader->base->total_num)
        reader->udata->hit_rate = hit_rate_array;
    
    return hit_rate_array;
}


double* get_hit_rate_seq(reader_t* reader, gint64 size, gint64 begin, gint64 end){
    int i=0;
    if (reader->base->total_num == -1)
        reader->base->total_num = get_num_of_cache_lines(reader);
    
    if (size == -1)
        size = reader->base->total_num;
    if (end == -1)
        end = reader->base->total_num;
    if (begin == -1)
        begin = 0;
    if (end-begin<size)
        size = end-begin;

    if (reader->udata->hit_rate && size==reader->base->total_num && end-begin==reader->base->total_num)
        return reader->udata->hit_rate;

    
    guint64* hit_count_array = get_hit_count_seq(reader, size, begin, end);
    double total_num = (double)(end - begin);

    
    
    double* hit_rate_array = g_new(double, size+3);
    hit_rate_array[0] = hit_count_array[0]/total_num;
    for (i=1; i<size+1; i++){
        hit_rate_array[i] = hit_count_array[i]/total_num + hit_rate_array[i-1];
    }
    // larger than given cache size
    hit_rate_array[size+1] = hit_count_array[size+1]/total_num;
    // cold miss
    hit_rate_array[size+2] = hit_count_array[size+2]/total_num;
    
    g_free(hit_count_array);
    if (size==reader->base->total_num && end-begin==reader->base->total_num)
        reader->udata->hit_rate = hit_rate_array;
    
    return hit_rate_array;
}


double* get_miss_rate_seq(reader_t* reader, gint64 size, gint64 begin, gint64 end){
    int i=0;
    if (reader->base->total_num == -1)
        reader->base->total_num = get_num_of_cache_lines(reader);
    if (size == -1)
        size = reader->base->total_num;
    if (end == -1)
        end = reader->base->total_num;
    if (begin == -1)
        begin = 0;
    
    if (end-begin<size)
        size = end-begin;

    double* hit_rate_array = get_hit_rate_seq(reader, size, begin, end);

    double* miss_rate_array = g_new(double, size+3);
    for (i=0; i<size+1; i++)
        miss_rate_array[i] = 1 - hit_rate_array[i];
    miss_rate_array[size+1] = hit_rate_array[size+1];
    miss_rate_array[size+2] = hit_rate_array[size+2];

    if (size!=reader->base->total_num || end-begin!=reader->base->total_num)
        g_free(hit_rate_array);

    return miss_rate_array;
}


gint64* get_reuse_dist_seq(reader_t* reader, gint64 begin, gint64 end){
    /*
     * TODO: might be better to split return result, in case the hit rate array is too large
     * Is there a better way to do this? this will cause huge amount memory
     * It is the user's responsibility to release the memory of hit count array returned by this function
     */
    
    guint64 ts = 0, max_rd = 0;
    gint64 reuse_dist;

    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    
    if (begin != 0 && begin != -1){
        WARNING("range reuse distance computation is no longer supported(begin=%ld), "
                "the returned result is full reust distance\n", begin);
    }
    if (end !=0 && end != -1 && end != reader->base->total_num){
        WARNING("range reuse distance computation is no longer supported(end=%ld), "
                "the returned result is full reust distance\n", end);
    }
    
    
    if (begin < 0)
        begin = 0;
    if (end <= 0)
        end = reader->base->total_num;
    
    
    // check whether the reuse dist computation has been finished
    if (reader->sdata->reuse_dist &&
            reader->sdata->reuse_dist_type == NORMAL_REUSE_DISTANCE){
        return reader->sdata->reuse_dist;
    }

    gint64 * reuse_dist_array = g_new(gint64, reader->base->total_num);
    
    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();
    
    // create hashtable
    GHashTable * hash_table;
    if (reader->base->type == 'v'){
        cp->type = 'l';
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else{
        cp->type = 'c';
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    
    // create splay tree
    sTree* splay_tree = NULL;
    
    read_one_element(reader, cp);
    while (cp->valid){
        splay_tree = process_one_element(cp, splay_tree, hash_table,
                                         ts, &reuse_dist);
        reuse_dist_array[ts] = reuse_dist;
        if (reuse_dist > (gint64)max_rd){
            max_rd = reuse_dist;
        }
        if (ts >= (guint64)end-begin)
            break;
        read_one_element(reader, cp);
        ts++;
    }
    
    
    reader->sdata->reuse_dist = reuse_dist_array;
    reader->sdata->max_reuse_dist = max_rd;
    reader->sdata->reuse_dist_type = NORMAL_REUSE_DISTANCE;

    
    // clean up
    destroy_cacheline(cp);
    g_hash_table_destroy(hash_table);
    free_sTree(splay_tree);
    reset_reader(reader);
    return reuse_dist_array;
}


gint64* get_future_reuse_dist(reader_t* reader, gint64 begin, gint64 end){
    /*  this function finds har far in the future, a given item will be requested again,
     *  if it won't be requested again, then -1. 
     *  ATTENTION: the reuse distance of the last element is at last, 
     *  meaning the sequence is NOT reversed.
     
     *  It is the user's responsibility to release the memory of hit count array
     *  returned by this function.
     */
    
    guint64 ts = 0, max_rd = 0;
    gint64 reuse_dist;
    
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);

    if (begin != 0 && begin != -1){
        WARNING("range reuse distance computation is no longer supported, "
                "the returned result is full reust distance\n");
    }
    if (end !=0 && end != -1 && end != reader->base->total_num){
        WARNING("range reuse distance computation is no longer supported, "
                "the returned result is full reust distance\n");
    }
    
    if (begin < 0)
        begin = 0;
    if (end < 0)
        end = reader->base->total_num;
    
    // check whether the reuse dist computation has been finished or not
    if (reader->sdata->reuse_dist &&
        reader->sdata->reuse_dist_type == FUTURE_REUSE_DISTANCE){
        return reader->sdata->reuse_dist;
    }
    
    gint64 * reuse_dist_array = g_new(gint64, reader->base->total_num);
    
    // create cache lize struct and initializa
    cache_line* cp = new_cacheline();
    
    // create hashtable
    GHashTable * hash_table;
    if (reader->base->type == 'v'){
        cp->type = 'l';
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else{
        cp->type = 'c';
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    
    // create splay tree
    sTree* splay_tree = NULL;
    
    reader_set_read_pos(reader, 1.0);
    go_back_one_line(reader);
    read_one_element(reader, cp);
    set_no_eof(reader);
    while (cp->valid){
        if (ts==reader->base->total_num)
            break;
        
        splay_tree = process_one_element(cp, splay_tree, hash_table,
                                         ts, &reuse_dist);
        if (end-begin-1-(long)ts < 0){
            ERROR("array index %ld out of range\n", end-begin-1-ts);
            exit(1);
        }
        reuse_dist_array[end-begin-1-ts] = reuse_dist;
        if (reuse_dist > (gint64) max_rd)
            max_rd = reuse_dist;
        if (ts >= (guint64) end-begin)
            break;
        read_one_element_above(reader, cp);
        ts++;
    }
    
    // save to reader
    if (reader->sdata->reuse_dist != NULL){
        g_free(reader->sdata->reuse_dist);
        reader->sdata->reuse_dist = NULL;
    }
    reader->sdata->reuse_dist = reuse_dist_array;
    reader->sdata->max_reuse_dist = max_rd;
    reader->sdata->reuse_dist_type = FUTURE_REUSE_DISTANCE;

    // clean up
    destroy_cacheline(cp);
    g_hash_table_destroy(hash_table);
    free_sTree(splay_tree);
    reset_reader(reader);
    return reuse_dist_array;
}




static inline sTree* process_one_element(cache_line* cp,
                                         sTree* splay_tree,
                                         GHashTable* hash_table,
                                         guint64 ts,
                                         gint64* reuse_dist){
    gpointer gp;
    
    gp = g_hash_table_lookup(hash_table, cp->item_p);
    
    sTree* newtree;
    if (gp == NULL){
        // first time access
        newtree = insert(ts, splay_tree);
        gint64 *value = g_new(gint64, 1);
        if (value == NULL){
            ERROR("not enough memory\n");
            exit(1);
        }
        *value = ts;
        if (cp->type == 'c')
            g_hash_table_insert(hash_table, g_strdup((gchar*)(cp->item_p)), (gpointer)value);
        
        else if (cp->type == 'l'){
            gint64* key = g_new(gint64, 1);
            *key = *(guint64*)(cp->item_p);
            g_hash_table_insert(hash_table, (gpointer)(key), (gpointer)value);
        }
        else{
            ERROR("unknown cache line content type: %c\n", cp->type);
            exit(1);
        }
        *reuse_dist = -1;
    }
    else{
        // not first time access
        guint64 old_ts = *(guint64*)gp;
        newtree = splay(old_ts, splay_tree);
        *reuse_dist = node_value(newtree->right);
        *(guint64*)gp = ts;
        
        newtree = delete(old_ts, newtree);
        newtree = insert(ts, newtree);
        
    }
    return newtree;
}

