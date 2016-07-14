#include "heatmap.h"

static inline gint process_one_element_last_access(cache_line* cp, GHashTable* hash_table, guint64 ts);


GSList* get_last_access_dist_seq(READER* reader, void (*funcPtr)(READER*, cache_line*)){
    /*
    !!!!!! the list returned from this function is in reversed order!!!!!!
    */

    /* this function currently using int, may cause some problem when the 
    trace file is tooooooo large 
    */

    GSList* list= NULL; 

    if (reader->total_num == -1)
        get_num_of_cache_lines(reader);

    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();

    // create hashtable
    GHashTable * hash_table; 
    if (reader->type == 'v'){
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
    
    guint64 ts = 0;
    gint dist;

    if (funcPtr == read_one_element){
        read_one_element(reader, cp);
    }
    else if (funcPtr == read_one_element_above){
        reader_set_read_pos(reader, 1.0);
        go_back_one_line(reader);
        read_one_element(reader, cp);
        if (reader->type == 'c')
            reader->reader_end = FALSE;
    }
    else{
        fprintf(stderr, "unknown function pointer received in heatmap: get_last_access_dist_seq\n");
        exit(1);
    }
    while (cp->valid){
        dist = process_one_element_last_access(cp, hash_table, ts);
        list = g_slist_prepend(list, GINT_TO_POINTER(dist));
        funcPtr(reader, cp);
        ts++;
    }
    if (reader->type == 'c' && reader->has_header){
        list = g_slist_remove(list, list->data);
    }


    // clean up
    g_free(cp);
    g_hash_table_destroy(hash_table);
    reset_reader(reader);
    return list;
}




static inline gint process_one_element_last_access(cache_line* cp, GHashTable* hash_table, guint64 ts){
    gpointer gp;
    gp = g_hash_table_lookup(hash_table, cp->item);
    gint ret;
    if (gp == NULL){
        // first time access
        ret = -1;
        guint64* value = g_new(guint64, 1);
        *value = ts;
        if (cp->type == 'c') 
            g_hash_table_insert(hash_table, g_strdup((gchar*)(cp->item_p)), (gpointer)value);
        
        else if (cp->type == 'l'){
            guint64* key = g_new(guint64, 1);
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
        guint64 old_ts = *(guint64*)gp;
        ret = (gint) (ts - old_ts);
        *(guint64*)gp = cp->ts;
    }
    return ret;
}


GArray* gen_breakpoints_virtualtime(READER* reader, guint64 time_interval){
    /* 
     return a GArray of break points, including the last break points
     */
    
    if (reader->total_num == -1)
        get_num_of_cache_lines(reader);
    
    if (reader->break_points){
        if (reader->break_points->mode == 'v' && reader->break_points->time_interval == time_interval)
            return reader->break_points->array;
        else{
            g_array_free(reader->break_points->array, TRUE);
            free(reader->break_points);
        }
    }
    
    guint i;
    guint array_size = (guint) ceil((reader->total_num/time_interval)+1) ;
    array_size ++ ;
    
    GArray* break_points = g_array_sized_new(FALSE, FALSE, sizeof(guint64), array_size);
    for (i=0; i<array_size-1; i++){
        guint64 value = i * time_interval;
        g_array_append_val(break_points, value);
    }
    g_array_append_val(break_points, reader->total_num);
    
    
    if (break_points->len > 10000)
        printf("%snumber of pixels in one dimension are more than 10000, exact size: %d, it may take a very long time, if you didn't intend to do it, please try with a larger time stamp\n", KRED, break_points->len);
    else if (break_points->len < 20)
        printf("%snumber of pixels in one dimension are less than 20, exact size: %d, each pixel will be very large, if you didn't intend to do it, please try with a smaller time stamp\n", KRED, break_points->len);
    
    
    struct break_point* bp = g_new(struct break_point, 1);
    bp->mode = 'v';
    bp->time_interval = time_interval;
    bp->array = break_points;
    reader->break_points = bp;

    reset_reader(reader);
    return break_points;
}


GArray* gen_breakpoints_realtime(READER* reader, guint64 time_interval){
    /*
     currently this only works for vscsi reader !!!
     return a GArray of break points, including the last break points
     */
    if (reader->type != 'v' && reader->type != 'c'){
        printf("gen_breakpoints_realtime currently only support vscsi reader and some csv reader, you provide %c reader, program exit\n", reader->type);
        exit(1);
    }
    if (reader->type == 'c')
        if (reader->real_time_column == -1){
            printf("gen_breakpoints_realtime needs you to provide real_time_column parameter, program exit\n");
            exit(1);
        }
            
    if (reader->break_points){
        if (reader->break_points->mode == 'r' && reader->break_points->time_interval == time_interval){
            return reader->break_points->array;
        }
        else{
            g_array_free(reader->break_points->array, TRUE);
            free(reader->break_points);
        }
    }
    
    if (reader->total_num == -1)
        get_num_of_cache_lines(reader);

    guint64 previous_time = 0;
    GArray* break_points = g_array_new(FALSE, FALSE, sizeof(guint64));

    // create cache lize struct and initialization
    cache_line* cp = new_cacheline();
    
    guint64 num = 0;

    read_one_element(reader, cp);
    previous_time = cp->real_time;
    g_array_append_val(break_points, num);

    while (cp->valid){
        if (cp->real_time - previous_time > (guint64)time_interval){
            g_array_append_val(break_points, num);
            previous_time = cp->real_time;
        }
        read_one_element(reader, cp);
        num++;
    }
    if ((long long)g_array_index(break_points, guint64, break_points->len-1) != reader->total_num)
        g_array_append_val(break_points, reader->total_num);

    if (break_points->len > 10000)
        printf("%snumber of pixels in one dimension are more than 10000, exact size: %d, it may take a very long time, if you didn't intend to do it, please try with a larger time stamp\n", KRED, break_points->len);
    else if (break_points->len < 20)
        printf("%snumber of pixels in one dimension are less than 20, exact size: %d, each pixel will be very large, if you didn't intend to do it, please try with a smaller time stamp\n", KRED, break_points->len);

    struct break_point* bp = g_new(struct break_point, 1);
    bp->mode = 'r';
    bp->time_interval = time_interval;
    bp->array = break_points;
    reader->break_points = bp;
    
    // clean up
    g_free(cp);
    reset_reader(reader);
    return break_points;
}


//
//#include "reader.h"
//#include "FIFO.h"
//#include "Optimal.h"
//
//int main(int argc, char* argv[]){
//# define CACHESIZE 2000
//# define BIN_SIZE 200
//    int i;
//
//    printf("test_begin!\n");
//
//    READER* reader = setup_reader(argv[1], 'v');
//
////    struct_cache* cache = fifo_init(CACHESIZE, 'v', NULL);
//
////    struct optimal_init_params init_params = {.reader=reader, .next_access=NULL};
////
////    struct_cache* cache = optimal_init(CACHESIZE, 'v', (void*)&init_params);
//
//
//
//    printf("after initialization, begin profiling\n");
//    
//    GArray* break_points = gen_breakpoints_realtime(reader, 1000000000);
//    for (i=0; i<break_points->len; i++)
//        printf("%lu\n", g_array_index(break_points, guint64, i));
////    g_array_free(break_points, TRUE);
//    
//    break_points = gen_breakpoints_realtime(reader, 1000000000);
//    for (i=0; i<break_points->len; i++)
//        printf("%lu\n", g_array_index(break_points, guint64, i));
//    
//    close_reader(reader);
//    printf("test_finished!\n");
//    return 0;
//}



