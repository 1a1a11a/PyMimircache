//
//  LRUAnalyzer.c
//  LRUAnalyzer
//
//  Created by Juncheng on 5/25/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "LRUProfiler.h"

#ifdef __cplusplus
extern "C"
{
#endif





static inline sTree*
process_one_element(cache_line* cp,
                    sTree* splay_tree,
                    GHashTable* hash_table,
                    guint64 ts,
                    gint64* reuse_dist);






/**
 * get hit count for 0~given cache size, 
 * non-parallel version
 *
 * @param reader: reader for reading data
 * @param size: the max profiling size, if -1, then the maxinum possible size 
 * @param begin: the begin pos of the trace 
 * @param end: teh end pos of the trace
 */

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
    // create cache line struct and initializa
    cache_line* cp = new_cacheline();
    cp->type = reader->base->data_type;
    
    // create hashtable
    GHashTable * hash_table;
    if (reader->base->data_type == 'l'){
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else if (reader->base->data_type == 'c' ){
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
            /* why + 1 here ? */ 
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


guint64* get_hitcount_withsize_seq(reader_t* reader, gint64 size, int block_unit_size){
    /* allow variable size of request */
    
    guint64 ts=0;
    gint64 reuse_dist;
    guint64 * hit_count_array;
    int n, i;
    
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    
    if (size == -1)
        size = reader->base->total_num;
    
    // check for size option
    if (reader->base->type == 'p' || reader->base->data_type == 'c'){
        ERROR("plain reader or dataType c does not support profiling with size\n");
        abort();
    }
    
    /* for a cache_size=size, we need size+1 bucket for size 0~size(included),
     * the last element(size+1) is used for storing count of reuse distance > size
     * if size==reader->base->total_num, then the last two is not used
     */
    hit_count_array = g_new0(guint64, size+3);
    
    
    // create cache line struct and initialization
    cache_line* cp = new_cacheline();
    cp->type = reader->base->data_type;

    cp->block_unit_size = reader->base->block_unit_size;
    cp->disk_sector_size = reader->base->disk_sector_size;
    if (cp->block_unit_size == 0 || cp->disk_sector_size == 0){
        ERROR("block unit size %d disk sector size %d cannot be 0\n",
              (int) cp->block_unit_size, (int) cp->disk_sector_size);
        abort();
    }
    
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
    while (cp->valid){
        *(gint64*)(cp->item_p) = (gint64) (*(gint64*)(cp->item_p) * cp->disk_sector_size / block_unit_size);

        splay_tree = process_one_element(cp, splay_tree, hash_table, ts, &reuse_dist);
        if (reuse_dist == -1)
            hit_count_array[size+2] += 1;
        else if (reuse_dist>=size)
            hit_count_array[size+1] += 1;
        else
            hit_count_array[reuse_dist+1] += 1;

        // new 170428
        if (cp->size == 0){
            if (cp->type == 'c'){
                WARNING("ts %lu, request lbn %s size 0\n", ts, (char*)(cp->item_p));
            }
            else{
                WARNING("ts %lu, request lbn %ld size 0\n", ts, *(gint64*)(cp->item_p));
            }
        }
        
        n = (int)ceil((double) cp->size/block_unit_size);
        for (i=0; i<n-1; i++){
            ts++;
            (*(guint64*)(cp->item_p)) ++;
            splay_tree = process_one_element(cp, splay_tree, hash_table, ts, &reuse_dist);
        }
        // end
        (*(guint64*)(cp->item_p)) -= (n-1);
        
        ts++;
        read_one_element(reader, cp);
    }
    
    // clean up
    destroy_cacheline(cp);
    g_hash_table_destroy(hash_table);
    free_sTree(splay_tree);
    reset_reader(reader);
    return hit_count_array;
    
    
}


double* get_hitrate_withsize_seq(reader_t* reader,
                                gint64 size,
                                int block_size){
    int i=0;
    if (reader->base->total_num == -1)
        reader->base->total_num = get_num_of_cache_lines(reader);
    
    if (size == -1)
        size = reader->base->total_num;
    
    if (reader->udata->hit_rate && size==reader->base->total_num)
        return reader->udata->hit_rate;
    
    
    guint64* hit_count_array = get_hitcount_withsize_seq(reader, size, block_size);
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
        if (correction<0 && -correction > (long) hit_count_array[0])
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
        WARNING("range reuse distance computation is no longer supported (begin=%ld), "
                "the returned result is full reust distance\n", begin);
    }
    if (end !=0 && end != -1 && end != reader->base->total_num){
        WARNING("range reuse distance computation is no longer supported (end=%ld), "
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
    while (cp->valid){
        if (/* DISABLES CODE */ (0)){
            if (cp->type == 'l')
                printf("read in %ld\n", *(gint64*)(cp->item_p));
            else
                printf("read in %s\n", (char*) (cp->item_p)); 
        }
        
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
    
    gint64 ts = 0, max_rd = 0;
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
    
    // create cache line struct and initializa
    cache_line* cp = new_cacheline();
    cp->type = reader->base->data_type;
    
    // create hashtable
    GHashTable * hash_table;
    if (reader->base->data_type == 'l'){
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else if (reader->base->data_type == 'c' ){
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
    
    reader_set_read_pos(reader, 1.0);
    go_back_one_line(reader);
    read_one_element(reader, cp);
    set_no_eof(reader);
    while (cp->valid){
        if (ts == reader->base->total_num)
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
        if (ts >= end-begin)
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


/*-----------------------------------------------------------------------------
 *
 * get_dist_to_last_access --
 *      compute distance (not reuse distance) to last access,
 *      then save to reader->sdata->reuse_dist
 *
 *
 * Input:
 *      reader:         the reader for data
 *
 * Return:
 *      a pointer to the distance array
 *
 *-----------------------------------------------------------------------------
 */
gint64* get_dist_to_last_access(reader_t* reader){
    
    guint64 ts = 0, max_dist = 0;
    gint64 dist;
    gint64* value;
    
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    
    
    // check whether dist computation has been finished
    if (reader->sdata->reuse_dist &&
        reader->sdata->reuse_dist_type == NORMAL_DISTANCE){
        return reader->sdata->reuse_dist;
    }
    
    gint64 * dist_array = g_new(gint64, reader->base->total_num);
    
    // create cache line struct and initializa
    cache_line* cp = new_cacheline();
    cp->type = reader->base->data_type;
    
    // create hashtable
    GHashTable * hash_table;
    if (reader->base->data_type == 'l'){
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else if (reader->base->data_type == 'c' ){
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else{
        ERROR("does not recognize reader data type %c\n", reader->base->data_type);
        abort();
    }
    
    read_one_element(reader, cp);
    while (cp->valid){
        if (/* DISABLES CODE */ (0)){
            if (cp->type == 'l')
                printf("read in %ld\n", *(gint64*)(cp->item_p));
            else
                printf("read in %s\n", (char*) (cp->item_p));
        }
        
        value = g_hash_table_lookup(hash_table, cp->item_p);
        if (value != NULL){
            dist = (gint64) ts - *(gint64*) (value);
        }
        else
            dist = -1;

        dist_array[ts] = dist;
        if (dist > (gint64)max_dist){
            max_dist = dist;
        }
        
        // insert into hashtable
        value = g_new(gint64, 1);
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

        
        read_one_element(reader, cp);
        ts++;
    }
    
    
    reader->sdata->reuse_dist = dist_array;
    reader->sdata->max_reuse_dist = max_dist;
    reader->sdata->reuse_dist_type = NORMAL_DISTANCE;
    
    
    // clean up
    destroy_cacheline(cp);
    g_hash_table_destroy(hash_table);
    reset_reader(reader);
    return dist_array;
}



/*-----------------------------------------------------------------------------
 *
 * get_reuse_time --
 *      compute reuse time (wall clock time) since last request,
 *      then save reader->sdata->reuse_dist
 *
 *      Notice that this is not reuse distance, 
 *      even though it is stored in reuse distance array
 *      And also the time is converted to integer, so floating point timestmap 
 *      loses its accuracy
 *
 * Input:
 *      reader:         the reader for data
 *
 * Return:
 *      a pointer to the reuse time array
 *
 *-----------------------------------------------------------------------------
 */
gint64* get_reuse_time(reader_t* reader){
    /*
     * TODO: might be better to split return result, in case the hit rate array is too large
     * Is there a better way to do this? this will cause huge amount memory
     * It is the user's responsibility to release the memory of hit count array returned by this function
     */
    
    guint64 ts = 0, max_rt = 0;
    gint64 rt;
    gint64* value;
    
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    
    
    // check whether dist computation has been finished
    if (reader->sdata->reuse_dist &&
        reader->sdata->reuse_dist_type == REUSE_TIME){
        return reader->sdata->reuse_dist;
    }
    
    gint64 * dist_array = g_new(gint64, reader->base->total_num);
    
    // create cache line struct and initializa
    cache_line* cp = new_cacheline();
    cp->type = reader->base->data_type;
    
    // create hashtable
    GHashTable * hash_table;
    if (reader->base->data_type == 'l'){
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else if (reader->base->data_type == 'c' ){
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                           (GDestroyNotify)simple_g_key_value_destroyer, \
                                           (GDestroyNotify)simple_g_key_value_destroyer);
    }
    else{
        ERROR("does not recognize reader data type %c\n", reader->base->data_type);
        abort();
    }
    
    read_one_element(reader, cp);
    if (cp->real_time == -1){
        ERROR("given reader does not have real time column\n");
        abort();
    }
    
    
    while (cp->valid){
        if (/* DISABLES CODE */ (0)){
            if (cp->type == 'l')
                printf("read in %ld\n", *(gint64*)(cp->item_p));
            else
                printf("read in %s\n", (char*) (cp->item_p));
        }
        
        value = g_hash_table_lookup(hash_table, cp->item_p);
        if (value != NULL){
            rt = (gint64) cp->real_time - *(gint64*) (value);
        }
        else
            rt = -1;
        
        dist_array[ts] = rt;
        if (rt > (gint64) max_rt){
            max_rt = rt;
        }

        // insert into hashtable
        value = g_new(gint64, 1);
        if (value == NULL){
            ERROR("not enough memory\n");
            exit(1);
        }
        *value = cp->real_time;
        if (cp->type == 'c')
            g_hash_table_insert(hash_table, g_strdup((gchar*)(cp->item_p)), (gpointer)value);
        
        else if (cp->type == 'l'){
            gint64* key = g_new(gint64, 1);
            *key = *(guint64*)(cp->item_p);
            g_hash_table_insert(hash_table, (gpointer)(key), (gpointer)value);
        }
        
        
        read_one_element(reader, cp);
        ts++;
    }
    
    
    reader->sdata->reuse_dist = dist_array;
    reader->sdata->max_reuse_dist = max_rt; 
    reader->sdata->reuse_dist_type = NORMAL_DISTANCE;
    
    
    // clean up
    destroy_cacheline(cp);
    g_hash_table_destroy(hash_table);
    reset_reader(reader);
    return dist_array;
}




/*-----------------------------------------------------------------------------
 *
 * cal_save_reuse_dist --
 *      compute reuse distance or future reuse distance, 
 *      then save to file to facilitate future computation
 *
 *      In detail, this function calculates reuse distance,
 *          then saves the array to the specified location,
 *          for each entry in the array, saves using gint64
 *
 * Input:
 *      reader:         the reader for data
 *      save_file_loc   the location to save frd file
 *
 * Return:
 *      None
 *
 *-----------------------------------------------------------------------------
 */
void cal_save_reuse_dist(reader_t * const reader,
                         const char * const save_file_loc,
                         const int reuse_type){
    
    gint64 *rd = NULL;
    
    if (reuse_type == NORMAL_REUSE_DISTANCE)
        rd = get_reuse_dist_seq(reader, 0, -1);
    else if (reuse_type == FUTURE_REUSE_DISTANCE)
        rd = get_future_reuse_dist(reader, 0, -1);
    else{
        ERROR("cannot recognize reuse_type %d\n", reuse_type);
        abort(); 
    }
        

    FILE* file = fopen(save_file_loc, "wb");
    fwrite(rd, sizeof(gint64), reader->base->total_num, file);
    fclose(file);
}


/*-----------------------------------------------------------------------------
 *
 * load_future_reuse_dist --
 *      this function is used for loading either reuse distance or
 *      future reuse distance from the file, which is pre-computed
 *
 *
 * Input:
 *      reader:         the reader for saving data
 *      load_file_loc   the location to file for loading rd or frd
 *
 * Return:
 *      None
 *
 *-----------------------------------------------------------------------------
 */
void load_reuse_dist(reader_t * const reader,
                     const char *const load_file_loc,
                     const int reuse_type){

    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);

    gint64 * reuse_dist_array = g_new(gint64, reader->base->total_num);
    FILE* file = fopen(load_file_loc, "rb");
    fread(reuse_dist_array, sizeof(gint64), reader->base->total_num, file);
    fclose(file);

    int i;
    gint64 max_rd = -1;
    for (i=0; i<reader->base->total_num; i++)
        if (reuse_dist_array[i] > max_rd)
            max_rd = reuse_dist_array[i];
    
    // save to reader
    if (reader->sdata->reuse_dist != NULL){
        g_free(reader->sdata->reuse_dist);
        reader->sdata->reuse_dist = NULL;
    }
    reader->sdata->reuse_dist = reuse_dist_array;
    reader->sdata->max_reuse_dist = max_rd;
    reader->sdata->reuse_dist_type = reuse_type;
}



/*-----------------------------------------------------------------------------
 *
 * process_one_element --
 *      this function is used for computing reuse distance for each request
 *      it maintains a hashmap and a splay tree, 
 *      time complexity is O(log(N)), N is the number of unique elements
 *
 *
 * Input:
 *      cp           the cache line struct contains input data (request label)
 *      splay_tree   the splay tree struct 
 *      hash_table   hashtable for remember last request timestamp (virtual)
 *      ts           current timestamp 
 *      reuse_dist   the calculated reuse distance
 *
 * Return:
 *      splay tree struct pointer, because splay tree is modified every time, 
 *      so it is essential to update the splay tree
 *
 *-----------------------------------------------------------------------------
 */
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
        
        newtree = splay_delete(old_ts, newtree);
        newtree = insert(ts, newtree);
        
    }
    return newtree;
}


#ifdef __cplusplus
}
#endif
