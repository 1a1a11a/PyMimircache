//
//  LRUAnalyzer.c
//  LRUAnalyzer
//
//  Created by Juncheng on 5/25/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "LRUProfiler.h"


static inline sTree* process_one_element(cache_line* cp, sTree* splay_tree, GHashTable* hash_table, long long ts, long long* reuse_dist);

long long* get_hit_count_seq(READER* reader, long size, long long begin, long long end){
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
     // printf("size of int :%lu\n", sizeof(int));
     // printf("size of long :%lu\n", sizeof(long));
     // printf("size of long long :%lu\n", sizeof(long long));
    int i;

    long long ts=0, reuse_dist;
    long long * hit_count_array;
    
    if (reader->total_num == -1)
        get_num_of_cache_lines(reader);
    
    if (size == -1)
        size = reader->total_num;

    if (begin == -1)
        begin = 0;
    
    if (end == -1)
        end = reader->total_num+1;
    
    
    if (end-begin<size)
        size = end-begin;

    /* for a cache_size=size, we need size+1 bucket for size 0~size(included),
     * the last element(size+1) is used for storing count of reuse distance > size
     * if size==reader->total_num, then the last two is not used
     */
    hit_count_array = malloc(sizeof(long long)* (size+3));

    
    for (i=0; i<size+3; i++)
        hit_count_array[i] = 0;
    
    
    // create cache lize struct and initialization
    cache_line* cp = (cache_line*)malloc(sizeof(cache_line));
    cp->op = -1;
    cp->size = -1;
    cp->valid = TRUE;
    
    // create hashtable
    GHashTable * hash_table; 
    if (reader->type == 'v'){
        cp->type = 'l'; 
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                            (GDestroyNotify)simple_key_value_destroyed, \
                                            (GDestroyNotify)simple_key_value_destroyed);
    }
    else{
        cp->type = 'c';
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                            (GDestroyNotify)simple_key_value_destroyed, \
                                            (GDestroyNotify)simple_key_value_destroyed);
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
        if (reader->ts >= end)
            break;
        read_one_element(reader, cp);
        ts++;
    }
        
    // clean up
    free(cp);
    g_hash_table_destroy(hash_table);
    free_sTree(splay_tree);
    reset_reader(reader);
    return hit_count_array;

    
}


double* get_hit_rate_seq(READER* reader, long size, long long begin, long long end){
    int i=0;
    if (reader->total_num == -1)
        reader->total_num = get_num_of_cache_lines(reader);

    if (size == -1)
        size = reader->total_num;
    if (end == -1)
        end = reader->total_num;
    if (begin == -1)
        begin = 0;

    if (reader->hit_rate && (size==-1 || size==reader->total_num) && end-begin==reader->total_num)
        return reader->hit_rate;
    
    long long* hit_count_array = get_hit_count_seq(reader, size, begin, end);
    double total_num = (double)(end - begin);
    

    double* hit_rate_array = malloc(sizeof(double)*(size+3));
    hit_rate_array[0] = hit_count_array[0]/total_num;
    for (i=1; i<size+1; i++){
        hit_rate_array[i] = hit_count_array[i]/total_num + hit_rate_array[i-1];
    }
    // larger than given cache size
    hit_rate_array[size+1] = hit_count_array[size+1]/total_num;
    // cold miss
    hit_rate_array[size+2] = hit_count_array[size+2]/total_num;
    
    free(hit_count_array);
    if (size==reader->total_num && end-begin==reader->total_num)
        reader->hit_rate = hit_rate_array; 

    return hit_rate_array;
}


double* get_miss_rate_seq(READER* reader, long size, long long begin, long long end){
    int i=0;
    if (reader->total_num == -1)
        reader->total_num = get_num_of_cache_lines(reader);
    double* miss_rate_array = get_hit_rate_seq(reader, size, begin, end);
    if (size == -1)
        size = reader->total_num;

    for (i=0; i<size+1; i++)
        miss_rate_array[i] = 1 - miss_rate_array[i];
    
    return miss_rate_array;
}


long long* get_reuse_dist_seq(READER* reader, long long begin, long long end){
    /*
     * TODO: might be better to split return result, in case the hit rate array is too large
     * Is there a better way to do this? this will cause huge amount memory
     * It is the user's responsibility to release the memory of hit count array returned by this function
     */
    
    long long ts, reuse_dist, max_rd=0;
    ts = 0;
    if (reader->total_num == -1)
        get_num_of_cache_lines(reader);

    if (begin < 0)
        begin = 0;
    if (end <= 0)
        end = reader->total_num; 
    
    long long * reuse_dist_array = malloc(sizeof(long long)*(end-begin));

    
    // check whether the reuse dist computation has been finished or not
    if (reader->reuse_dist){
        long long i;
        for (i=begin; i<end; i++)
            reuse_dist_array[i-begin] = reader->reuse_dist[i];
        return reuse_dist_array;
    }
    
    // create cache lize struct and initializa
    cache_line* cp = (cache_line*)malloc(sizeof(cache_line));
    cp->op = -1;
    cp->size = -1;
    
    cp->valid = TRUE;
    
    // create hashtable
    GHashTable * hash_table; 
    if (reader->type == 'v'){
        cp->type = 'l'; 
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                            (GDestroyNotify)simple_key_value_destroyed, \
                                            (GDestroyNotify)simple_key_value_destroyed);
    }
    else{
        cp->type = 'c';
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                            (GDestroyNotify)simple_key_value_destroyed, \
                                            (GDestroyNotify)simple_key_value_destroyed);
    }

    // create splay tree
    sTree* splay_tree = NULL;

    if (begin != -1 && begin != 0)
        skip_N_elements(reader, begin);

    read_one_element(reader, cp);
    while (cp->valid){
        splay_tree = process_one_element(cp, splay_tree, hash_table, ts, &reuse_dist);
        reuse_dist_array[ts] = reuse_dist;
        if (reuse_dist > max_rd)
            max_rd = reuse_dist;
        if (reader->ts >= end)
            break;
        read_one_element(reader, cp);
        ts++;
    }
    
    if (reader->reuse_dist)
        free(reader->reuse_dist);
    
    if (end-begin == reader->total_num){
        reader->reuse_dist = reuse_dist_array;
        reader->max_reuse_dist = (guint64) max_rd;
    }
    
    // clean up
    free(cp);
    g_hash_table_destroy(hash_table);
    free_sTree(splay_tree);
    
    reset_reader(reader);
    return reuse_dist_array;
}


long long* get_future_reuse_dist(READER* reader, long long begin, long long end){
    /* the reuse distance of the last element is at last 
     
     */
    
    /*
     * TODO: might be better to split return result, in case the hit rate array is too large
     * Is there a better way to do this? this will cause huge amount memory
     * It is the user's responsibility to release the memory of hit count array returned by this function
     */
    
    long long ts, reuse_dist, max_rd=0;
    ts = 0;
    if (reader->total_num == -1)
        get_num_of_cache_lines(reader);
    
    if (begin < 0)
        begin = 0;
    if (end < 0)
        end = reader->total_num;
    
    long long * reuse_dist_array = malloc(sizeof(long long)*(end-begin));
    
    // create cache lize struct and initializa
    cache_line* cp = (cache_line*)malloc(sizeof(cache_line));
    cp->op = -1;
    cp->size = -1;
    cp->valid = TRUE;
    
    // create hashtable
    GHashTable * hash_table;
    if (reader->type == 'v'){
        cp->type = 'l';
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                           (GDestroyNotify)simple_key_value_destroyed, \
                                           (GDestroyNotify)simple_key_value_destroyed);
    }
    else{
        cp->type = 'c';
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                           (GDestroyNotify)simple_key_value_destroyed, \
                                           (GDestroyNotify)simple_key_value_destroyed);
    }
    
    // create splay tree
    sTree* splay_tree = NULL;
    
    if (begin != -1 && begin != 0)
        skip_N_elements(reader, begin);
    
    reader_set_read_pos(reader, 1.0);
    go_back_one_line(reader);
    read_one_element(reader, cp);
    while (cp->valid){
        splay_tree = process_one_element(cp, splay_tree, hash_table, ts, &reuse_dist);
        reuse_dist_array[end-begin-1-ts] = reuse_dist;
        if (reuse_dist > max_rd)
            max_rd = reuse_dist; 
        if (reader->ts >= end)
            break;
        read_one_element_above(reader, cp);
        ts++;
    }
    
    if (reader->reuse_dist)
        free(reader->reuse_dist);
    reader->reuse_dist = reuse_dist_array;
    reader->max_reuse_dist = (guint64) max_rd;

    // clean up
    free(cp);
    g_hash_table_destroy(hash_table);
    free_sTree(splay_tree);
    reset_reader(reader);
    return reuse_dist_array;
}




long long* get_rd_distribution(READER* reader, long long begin, long long end){
    /*
     * TODO: might be better to split return result, in case the hit rate array is too large
     * Is there a better way to do this? this will cause huge amount memory
     * It is the user's responsibility to release the memory of hit count array returned by this function
     */
    
    long long ts, reuse_dist;
    ts = 0;
    if (reader->total_num == -1)
        get_num_of_cache_lines(reader);
    
    if (begin < 0)
        begin = 0;
    if (end < 0)
        end = reader->total_num; 

    long long * reuse_dist_distribution_array = malloc(sizeof(long long)*(end-begin+1));
    
    int i; 
    for (i=0; i<end-begin+1; i++)
        reuse_dist_distribution_array[i] = 0;

    
    // create cache lize struct and initialization
    cache_line* cp = (cache_line*)malloc(sizeof(cache_line));
    cp->op = -1;
    cp->size = -1;
    cp->valid = TRUE;
    
    // create hashtable
    GHashTable * hash_table; 
    if (reader->type == 'v'){
        cp->type = 'l'; 
        hash_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, \
                                            (GDestroyNotify)simple_key_value_destroyed, \
                                            (GDestroyNotify)simple_key_value_destroyed);
    }
    else{
        cp->type = 'c';
        hash_table = g_hash_table_new_full(g_str_hash, g_str_equal, \
                                            (GDestroyNotify)simple_key_value_destroyed, \
                                            (GDestroyNotify)simple_key_value_destroyed);
    }


    // create splay tree
    sTree* splay_tree = NULL;
    

    if (begin != -1 && begin != 0)
        skip_N_elements(reader, begin);

    read_one_element(reader, cp);
    while (cp->valid){
        splay_tree = process_one_element(cp, splay_tree, hash_table, ts, &reuse_dist);
        if (reuse_dist != -1)  
            reuse_dist_distribution_array[reuse_dist] += 1;
        else
            reuse_dist_distribution_array[end-begin] += 1;
        if (reader->ts >= end)
            break;
        read_one_element(reader, cp);
        ts++;
    }
    
    
    // clean up
    free(cp);
    g_hash_table_destroy(hash_table);
    free_sTree(splay_tree);
    reset_reader(reader);
    return reuse_dist_distribution_array;
}


guint64 cal_best_LRU_cache_size(READER* reader, double threshhold){

#define point_spacing 5 
    
    
    guint64 cache_size = 0;
    if (!reader->hit_rate)
        get_hit_rate_seq(reader, -1, 0, -1);
    
    return cache_size;
}





static inline sTree* process_one_element(cache_line* cp, sTree* splay_tree, GHashTable* hash_table, long long ts, long long* reuse_dist){
    gpointer gp;
    if (cp->type == 'c') 
        gp = g_hash_table_lookup(hash_table, (gconstpointer)(cp->str_content));
    else if (cp->type == 'i')
        gp = g_hash_table_lookup(hash_table, (gconstpointer)(&(cp->int_content)));
    else if (cp->type == 'l')
        gp = g_hash_table_lookup(hash_table, (gconstpointer)(&(cp->long_content)));
    else{
        gp = NULL; 
        printf("unknown cache line type: %c\n", cp->type);
        exit(1);
    }

    sTree* newtree;
    if (gp == NULL){
        // first time access
        newtree = insert(ts, splay_tree);
        long long *value = (long long*)malloc(sizeof(long long));
        if (value == NULL){
            printf("not enough memory\n");
            exit(1);
        }
        *value = ts;
        if (cp->type == 'c') 
            g_hash_table_insert(hash_table, g_strdup((gchar*)(cp->str_content)), (gpointer)value);
        
        else if (cp->type == 'l'){
            gint64* key = g_new(gint64, 1);
            // long *key = (long* )malloc(sizeof(uint64_t));
            if (key == NULL){
                printf("not enough memory\n");
                exit(1);
            }            
            *key = cp->long_content;
            g_hash_table_insert(hash_table, (gpointer)(key), (gpointer)value);            
        }
        else if (cp->type == 'i'){
            int *key = (int* )malloc(sizeof(int));
            if (key == NULL){
                printf("not enough memory\n");
                exit(1);
            }  
            *key = cp->int_content;
            g_hash_table_insert(hash_table, (gpointer)(key), (gpointer)value);            
        }
        else{
            printf("unknown cache line content type: %c\n", cp->type);
            exit(1);
        }

        *reuse_dist = -1;
    }
    else{
        // not first time access
        long long old_ts = *(long long*)gp;
        newtree = splay(old_ts, splay_tree);
        *reuse_dist = node_value(newtree->right);
        *(long long*)gp = ts;
        
        newtree = delete(old_ts, newtree);
        newtree = insert(ts, newtree);
        
    }
    return newtree;
}



//#include "reader.h"
//
//int main(int argc, char* argv[]){
//# define CACHESIZE 1
//# define BIN_SIZE 1
//
//
//    printf("test_begin!\n");
//
//    READER* reader = setup_reader(argv[1], 'v');
//
//
//    printf("after initialization, begin profiling\n");
//    double* hr = get_hit_rate_seq(reader, -1, 0, -1);
//    printf("hit rate p: %p\n", hr);
//    
//    hr = get_hit_rate_seq(reader, -1, 10, 20);
//    printf("hit rate p: %p\n", hr);
//    long long *hc = get_hit_count_seq(reader, -1, 10, 20);
//    
//    int i;
//        for (i=0; i<20-10+3; i++){
//            printf("%d: %f\n", i, hr[i]);
//        }
//    for (i=0; i<20-10+3; i++){
//        printf("%d: %lld\n", i, hc[i]);
//    }
//
//    printf("test_finished!\n");
//    return 0;
//}
