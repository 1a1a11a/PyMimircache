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
cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
    return (next < curr);
}


static inline pqueue_pri_t
get_pri(void *a)
{
    return ((node_pq_LRUProfiler *) a)->pri;
}


static inline void
set_pri(void *a, pqueue_pri_t pri)
{
    ((node_pq_LRUProfiler *) a)->pri = pri;
}

static inline size_t
get_pos(void *a)
{
    return ((node_pq_LRUProfiler *) a)->pos;
}


static inline void
set_pos(void *a, size_t pos)
{
    ((node_pq_LRUProfiler *) a)->pos = pos;
}






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
    
    if (end-begin<size)
        size = end-begin;
    
    if (reader->hit_rate && size==reader->total_num && end-begin==reader->total_num)
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

static inline double cal_slope(double* hit_rate, guint64 pos, int num_of_points_for_average, int spacing){
    /**
     * hit_rate: double array for hit rate,
     * pos: the location of the curve for calculating slope, in other words, x-coordinate,
     * num_of_points_for_average: take num_of_points_for_average-1 points after pos and calculate ave slope,
     * spacing: the other group of points for calculating slope
     **/
    
    double slope;
    double ave_hr1 = 0, ave_hr2 = 0;
    int i;
    for (i=0; i<num_of_points_for_average; i++){
        ave_hr1 += hit_rate[pos + i];
        ave_hr2 += hit_rate[pos + spacing + i];
    }
    ave_hr1 /= num_of_points_for_average;
    ave_hr2 /= num_of_points_for_average;
    slope = (ave_hr2-ave_hr1) / spacing;
    return slope;
}


static inline double fake_second_derivative(double* hit_rate, guint64 pos){
    /** this function doesn't check boundary, check before use **/
    
    double fake_slope1, fake_slope2;
    fake_slope1 = hit_rate[pos] - hit_rate[pos-1];
    fake_slope2 = hit_rate[pos+1] - hit_rate[pos];
    return (fake_slope2 - fake_slope1);
}


//static inline void find_range(double* hit_rate, double step, long granularity, guint64 *begin, guint64 *end){
//    guint64 i;
//    for (i=*end; i>*begin+granularity; i-=granularity){
//        if (hit_rate[i] - hit_rate[i-granularity] > step){
//            *end = i;
//            *begin = i - granularity;
//            return;
//        }
//    }
//}

static inline gboolean verify_plateau(double* hit_rate, double max_hr, double ave_slope, guint *begin, guint* end, int num_of_points_for_average, int slope_difference, guint max_length){
    
#define cutoff 3
    
    guint64 i = 0;
    double init_slope = (hit_rate[*begin+num_of_points_for_average] - hit_rate[*begin])/num_of_points_for_average;
    double fake_slope = (hit_rate[*begin-i] - hit_rate[*begin-i-num_of_points_for_average])/num_of_points_for_average;
    gboolean second_chance = TRUE;
    // printf("enter verify_plateau, *pos = %u\n", *begin);
    while ((fake_slope/init_slope >= slope_difference || second_chance) && i<max_length && (int)(*begin)-(int)i>0){
        if (fake_slope/init_slope < slope_difference){
            // printf("second_chance, i = %lu, slope diff = %lf\n", i, fake_slope/init_slope);
            second_chance = FALSE;
        }
        else
            second_chance = TRUE;
        i++;
        // fake_slope = (hit_rate[*begin-i] - hit_rate[*begin-i-num_of_points_for_average])/num_of_points_for_average;
        fake_slope = (hit_rate[*begin] - hit_rate[*begin-i])/i;
    }
    if ( (hit_rate[*begin] - hit_rate[*begin - i] > max_hr/50) && ((hit_rate[*begin] - hit_rate[*begin - i])/i > ave_slope) ){
        // check whether decrease begin will increase slope
        guint64 j=0;
        *end = *begin - i;
        
        
        // slope method
        double max_slope = (hit_rate[*begin-j] - hit_rate[*begin - j - 1]);
        double *slopes = (double*) malloc(sizeof(double) * i);
        for (j=0; j<i; j++){
            slopes[j] = (hit_rate[*begin-j] - hit_rate[*begin - j - 1]);
            if (slopes[j] > max_slope)
                max_slope = slopes[j];
        }
        for (j=0; j<i-1; j++){
            if (slopes[j] > max_slope/cutoff && slopes[j+1] > max_slope/cutoff)
                break;
        }
        free(slopes);
        *begin = *begin - (j-1);
        
        
        // second derivative method
        // double* first_derivatives = (double*) malloc(sizeof(double)*i);
        // double second_derivative, max_second_derivative = 0;
        // guint64 max_2_d_pos = 0;
        // for (j=0; j<i-1; j++)
        //     first_derivatives[j] = hit_rate[*begin-j+1] - hit_rate[*begin-j];
        
        // for (j=0; j<i-2; j++){
        //     second_derivative = first_derivatives[j+1] - first_derivatives[j];
        //     if (second_derivative > max_second_derivative){
        //         max_second_derivative = second_derivative;
        //         max_2_d_pos = j;
        //     }
        // }
        // *begin = *begin - max_2_d_pos;
        // free(first_derivatives);
        
        // printf("success i = %u, shift j = %lu, begin=%u, end=%u\n", i, j, *begin, *end);
        return TRUE;
    }
    else{
        // if (hit_rate[*begin] - hit_rate[*begin - i] <= max_hr/50)
        //     printf("failed i = %u, max_hr problem: %lf < %lf\n", i, hit_rate[*begin] - hit_rate[*begin - i], max_hr/50);
        // else
        //     printf("failed i = %u, ave_slope problem\n", i);
        return FALSE;
    }
}



GQueue * cal_best_LRU_cache_size(READER* reader, unsigned int num, int force_spacing, int cut_off_divider){
    /**
     * find the best cache sizes for LRU, no max than num, any two cache_sizes will be separated by force_spacing,
     * and the number of returned cache_sizes will be cutoff if there are no more cache sizes having slope > max_slope/cut_off_divider
     **/
    
    
#define largest_granularity 100000
#define num_of_points_for_average 5
#define step 0.1
#define slope_difference 1.6
#define multiplier 100000000
#define BASE 200
    
    GQueue * gq = g_queue_new();
    pqueue_t* pq = pqueue_init(num, cmp_pri, get_pri, set_pri, get_pos, set_pos);
    GSList* slist = NULL;       // used to track the allocated node
    
    
    
    if (!reader->hit_rate)
        get_hit_rate_seq(reader, -1, 0, -1);
    double* hit_rate = reader->hit_rate;
    double max_hr = reader->hit_rate[reader->total_num-3];
    double ave_slope = 0;
    
    guint64 i = reader->total_num - 1;
    
    while (i >= largest_granularity){
        if (hit_rate[i] - hit_rate[i - largest_granularity] > step)
            // this means the hit rate begins to drop, and the largest plateau has passed
            break;
        i -= largest_granularity;
    }
    ave_slope = max_hr / i;
    guint pos = (guint)i;       // if it is still > 2^32, then the cache size is useless
    if (reader->total_num - pos < num_of_points_for_average-1)
        pos = (guint)reader->total_num - num_of_points_for_average-1;
    guint end = pos+force_spacing;
    double slope1, slope2;
    while (pos >= BASE){
        slope1 = (hit_rate[pos+num_of_points_for_average] - hit_rate[pos])/num_of_points_for_average;
        slope2 = (hit_rate[pos] - hit_rate[pos-num_of_points_for_average])/num_of_points_for_average;
        if (((int)end - (int)pos) > force_spacing && slope2 / slope1 >= slope_difference){
            if (verify_plateau(hit_rate, max_hr, ave_slope, &pos, &end, num_of_points_for_average, slope_difference, 1000)){
                node_pq_LRUProfiler* node = (node_pq_LRUProfiler*) malloc(sizeof(node_pq_LRUProfiler));
                node->data = pos;
                node->pri = (unsigned long long)( (hit_rate[pos] - hit_rate[end])/(pos-end) * multiplier );
                pqueue_insert(pq, (void*)node);
                slist = g_slist_prepend(slist, (gpointer)node);
                pos = end+1;    // in case after pos-1, it will becomoe negative
            }
        }
        pos -= 1;
    }
    if (pqueue_peek(pq) == NULL)
    {   // can't find best size
        pqueue_free(pq);
        g_slist_free_full(slist, simple_key_value_destroyed);
        g_queue_free(gq);
        reader->best_LRU_cache_size = NULL;
        return NULL;
    }
    unsigned long long max_pri = ((node_pq_LRUProfiler* )pqueue_peek(pq))->pri;
    for (i=0; i<num; i++){
        node_pq_LRUProfiler* node = (node_pq_LRUProfiler*) pqueue_pop(pq);
        if (node==NULL || node->pri < max_pri/cut_off_divider)
            break;
        g_queue_push_tail(gq, GUINT_TO_POINTER(node->data));
    }
    
    pqueue_free(pq);
    g_slist_free_full(slist, simple_key_value_destroyed);
    reader->best_LRU_cache_size = gq;
    return gq;
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


static void print_GQ(gpointer data, gpointer user_data){
    printf("%u\n", GPOINTER_TO_UINT(data));
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
//    for (i=0; i<20-10+3; i++){
//        printf("%d: %f\n", i, hr[i]);
//    }
//    for (i=0; i<20-10+3; i++){
//        printf("%d: %lld\n", i, hc[i]);
//    }
//    
//    printf("begin get best cache size test\n");
//    
//    cal_best_LRU_cache_size(reader, 20, 200, 20);
//    if (reader->best_LRU_cache_size == NULL){
//        printf("no best cache size found\n");
//        exit(-1);
//    }
//    
//    printf("num: %u\n", reader->best_LRU_cache_size->length);
//    g_queue_foreach(reader->best_LRU_cache_size, print_GQ, NULL);
//    
//    
//    
//    printf("test_finished!\n");
//    return 0;
//}