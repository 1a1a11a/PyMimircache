//
//  mimir.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//


#include "cache.h" 
#include "mimir.h"
#include <time.h>
#include <stdlib.h>


/* Mithril prefetching 
 * current version only records at cache miss, 
 * alternatively, we can also prefetch at eviction, or both 
 */

//#define TRACK_BLOCK 192618l

#ifdef __cplusplus
extern "C"
{
#endif



static inline void __MIMIR_record_entry_min_support_1(struct_cache* MIMIR, cache_line* cp);


void
check_rm_hashtable (gpointer key, gpointer value, gpointer user_data){
    struct_cache* MIMIR = (struct_cache*) user_data;
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    
    gint index = GPOINTER_TO_INT(value);
    if (index < 0){
        /* in mining table */
        gint64* row_in_mining_table = GET_ROW_IN_MINING_TABLE(MIMIR_params, -index-1);
        if (MIMIR->core->data_type == 'l'){
            if (*(gint64*)(key) != row_in_mining_table[0]){
                printf("ERROR hashtable mining key value not match %ld %ld\n", *(gint64*)(key), row_in_mining_table[0]);
                exit(1);
            }
        }
        else{
            if (strcmp(key, (char*)(row_in_mining_table[0]))!=0){
                printf("ERROR hashtable mining key value not match %s %s\n", (char*)key, (char*)(row_in_mining_table[0]));
                exit(1);
            }
        }
    }
        
    else{
        /* in recording table */
        gint64* row_in_recording_table = GET_ROW_IN_RECORDING_TABLE(MIMIR_params, index);
        if (MIMIR->core->data_type == 'l'){
            if (*(gint64*)(key) != row_in_recording_table[0]){
                printf("ERROR hashtable recording key value not match %ld %ld\n", *(gint64*)(key), row_in_recording_table[0]);
                exit(1);
            }
        }
        else{
            if (strcmp(key, (char*)(row_in_recording_table[0]))!=0){
                printf("ERROR hashtable recording key value not match %s %s\n", (char*)key, (char*)(row_in_recording_table[0]));
                exit(1);
            }
        }

        
    }

}

void check_prefetched_hashtable (gpointer key, gpointer value, gpointer user_data){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*) user_data;
    GHashTable *h = ((struct LRU_params*)(MIMIR_params->cache->cache_params))->hashtable;
    if (!g_hash_table_contains(h, key)){
        printf("ERROR in prefetched, not in cache %ld, %d\n", *(gint64*)key, GPOINTER_TO_INT(value));
        exit(1);
    }
}


//void training_hashtable_count_length(gpointer key, gpointer value, gpointer user_data){
//    GList* list_node = (GList*) value;
//    struct training_data_node *data_node = (struct training_data_node*) (list_node->data);
//    guint64 *counter = (guint64*) user_data;
//    (*counter) += data_node->length;
//}


void print_mining_table(struct_cache* MIMIR){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);

    struct recording_mining_struct* r_m_struct = MIMIR_params->record_mining_struct;
    guint32 i;
    for (i=0; i<r_m_struct->mining_table->len; i++)
        printf("%d: %s %ld\n", i, (char*)*(GET_ROW_IN_MINING_TABLE(MIMIR_params, i)), *(GET_ROW_IN_MINING_TABLE(MIMIR_params, i)));
}

void check_mining_table(struct_cache* MIMIR){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    struct recording_mining_struct* r_m_struct = MIMIR_params->record_mining_struct;
    guint32 i;
    for (i=0; i < r_m_struct->mining_table->len; i++)
        if (((char*)*(GET_ROW_IN_MINING_TABLE(MIMIR_params, i)))[0] <= '0' ||
            ((char*)*(GET_ROW_IN_MINING_TABLE(MIMIR_params, i)))[0] > '9'){
            printf("check mining table error %d/%u %s %c %d\n",
                   i, r_m_struct->mining_table->len,
                   ((char*)*(GET_ROW_IN_MINING_TABLE(MIMIR_params, i))),
                   ((char*)*(GET_ROW_IN_MINING_TABLE(MIMIR_params, i)))[0],
                   ((char*)*(GET_ROW_IN_MINING_TABLE(MIMIR_params, i)))[0]);
            exit(1);
        }
}




static inline gint __MIMIR_get_total_num_of_ts(gint64* row, gint row_length){
    int i, t;
    int count = 0;
    for (i=1; i<row_length; i++){
        t = NUM_OF_TS(row[i]);
        if (t == 0)
            return count;
        count += t;
    }
    return count;
}

static inline gboolean
__MIMIR_remove_from_recording_hashtable(char data_type,
                                        struct recording_mining_struct* r_m_struct,
                                        gint64* row){
    if (data_type == 'l'){
        return g_hash_table_remove(r_m_struct->hashtable, row);
    }
    else if (data_type == 'c'){
        return g_hash_table_remove(r_m_struct->hashtable, (char*)*row);
    }
    else
        fprintf(stderr, "does not recognize data type in __MIMIR_record_entry\n");
    return FALSE;
}





static inline gboolean __MIMIR_check_sequential(struct_cache* MIMIR, cache_line *cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    if (MIMIR_params->sequential_K == 0)                  // new 1010
        return FALSE;
    
    int i;
    if (MIMIR->core->data_type != 'l')
        printf("ERROR sequential prefetching but data is not long int\n");
    gint64 last = *(gint64*)(cp->item_p);
    gboolean sequential = TRUE;
    gpointer old_item_p = cp->item_p;
    cp->item_p = &last;
    gint sequential_K = MIMIR_params->sequential_K;
    if (sequential_K == -1)
        /* when use AMP, this is -1 */
        sequential_K = 1;
    for (i=0; i<sequential_K; i++){
        last --;
        if ( !MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp) ){
            sequential = FALSE;
            break;
        }
    }
    cp->item_p = old_item_p;
    return sequential;
}

void __MIMIR_insert_element(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    MIMIR_params->cache->core->__insert_element(MIMIR_params->cache, cp);
}



static inline void __MIMIR_record_entry(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    struct recording_mining_struct *r_m_struct = MIMIR_params->record_mining_struct;

    /** check whether this is a frequent item, if so, don't insert
     *  if the item is not in recording table, but in cache, means it is frequent, discard it   **/
//    if (MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp) &&
//        !g_hash_table_contains(r_m_struct->hashtable, cp->item_p))
//        return;
//
//    /* only record when it is not in the cache, even though it is in the recording/mining table */ 
//    if (MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp))
//        return;

    
    if ( MIMIR_params->recording_loc != each_req )
        if (MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp))
            return;
    
    
    /* check it is sequtial or not 0925     */
    if (MIMIR_params->sequential_type && __MIMIR_check_sequential(MIMIR, cp))
        return;

    if (MIMIR_params->min_support == 1){
        __MIMIR_record_entry_min_support_1(MIMIR, cp);
    }

    else{
        gint64* row_in_recording_table;
        // check the item in hashtable for training
        gint index = GPOINTER_TO_INT( g_hash_table_lookup(r_m_struct->hashtable, cp->item_p) );
#ifdef TRACK_BLOCK
        if (cp->ts > 1891280 && cp->ts < 1891290){
            printf("ts %lu, lookup rm_table %d\n", cp->ts, index);
            if (index > 0){
                // in recording table
                gint64 *row = GET_ROW_IN_RECORDING_TABLE(MIMIR_params, index);
                printf("ts %lu, number of ts %d\n", cp->ts, NUM_OF_TS(row[1]));
            }
        }
#endif
        
        if (index == 0){
            // the node is not in the recording/mining data, should be added
            row_in_recording_table = GET_CURRENT_ROW_IN_RECORDING_TABLE(MIMIR_params);
            if (cp->type == 'l'){
#ifdef SANITY_CHECK
                if (row_in_recording_table[0] != 0){
                    ERROR("recording table should already be cleaned, but not\n");
                    abort();
                }
#endif 
                
                row_in_recording_table[0] = *(gint64*)(cp->item_p);
                g_hash_table_insert(r_m_struct->hashtable, row_in_recording_table,
                                    GINT_TO_POINTER(r_m_struct->recording_table_pointer));
            }
            else if (cp->type == 'c'){
                gchar* str = g_strdup(cp->item_p);
                row_in_recording_table[0] = (gint64)str;
                g_hash_table_insert(r_m_struct->hashtable, str,
                                    GINT_TO_POINTER(r_m_struct->recording_table_pointer));
            }
            else{
                fprintf(stderr, "does not recognize data type in __MIMIR_record_entry\n");
            }
            
            row_in_recording_table[1] = ADD_TS(row_in_recording_table[1], MIMIR_params->ts);
            
            // move pointer to next position
            r_m_struct->recording_table_pointer ++;
            if (r_m_struct->recording_table_pointer >= r_m_struct->num_of_rows_in_recording_table){
                /* recording table is full */
                r_m_struct->recording_table_pointer = 1;
            }
            
            row_in_recording_table = GET_ROW_IN_RECORDING_TABLE(MIMIR_params, r_m_struct->recording_table_pointer);
            
            
            if (row_in_recording_table[0] != 0){
                /* clear current row,
                 this is because the recording table is full and we need to begin from beginning
                 and current position has old resident, we need to remove them */
                if (cp->type == 'c'){
                    if (!g_hash_table_contains(r_m_struct->hashtable, (char*)(row_in_recording_table[0])))
                        ERROR("remove old entry from recording table, but it is not in recording hashtable \
                               %s, pointer %ld, ts %ld\n", (char*)(row_in_recording_table[0]),
                               r_m_struct->recording_table_pointer, MIMIR_params->ts);
                    g_hash_table_remove(r_m_struct->hashtable, (char*)(row_in_recording_table[0]));
                }
                else if (cp->type == 'l'){
                    if (!g_hash_table_contains(r_m_struct->hashtable, row_in_recording_table)){
                        ERROR("remove old entry from recording table, but it is not "
                               "in recording hashtable, block %ld, recording table pos %ld, ts %ld\n", \
                               *row_in_recording_table, r_m_struct->recording_table_pointer, MIMIR_params->ts);
                        long temp = r_m_struct->recording_table_pointer - 1;
                        printf("previous line block %ld\n", *(gint64*)(GET_ROW_IN_RECORDING_TABLE(MIMIR_params, temp)));
                        abort();
                    }
                    g_hash_table_remove(r_m_struct->hashtable, row_in_recording_table);
                }
                else{
                    ERROR("does not recognize data type\n");
                    abort();
                }
                int i;
                for (i=0; i< r_m_struct->recording_table_row_len; i++){
                    row_in_recording_table[i] = 0 ;
                }
            }
            
        }
        else{
            /** first check it is in recording table or mining table,
             *  if in mining table (index < 0),
             *  check how many entries are there, if equal max_support, remove it
             *  otherwise add to mining table;
             *  if in recording table (index > 0),
             *  check how many entries are there in the list,
             *  if equal to min_support-1, add and move to mining table,
             **/
            if (index < 0){
                /* in mining table */
                gint64* row_in_mining_table = GET_ROW_IN_MINING_TABLE(MIMIR_params, -index-1);
                
#ifdef SANITY_CHECK
                if (cp->type == 'l'){
                    if (*(gint64*)(cp->item_p) != row_in_mining_table[0]){
                        ERROR("hashtable mining found position not correct %ld %ld\n", *(gint64*)(cp->item_p), row_in_mining_table[0]);
                        abort();
                    }
                }
                else{
                    if (strcmp(cp->item_p, (char*)(row_in_mining_table[0]))!=0){
                        ERROR("hashtable mining found position not correct %s %s\n", (char*)(cp->item_p), (char*)(row_in_mining_table[0]));
                        abort(); 
                    }
                }
#endif
                
                int i, timestamps_length = 0;
                
                
                for (i=1; i < r_m_struct->mining_table_row_len; i++){
                    timestamps_length += NUM_OF_TS(row_in_mining_table[i]);
                    if (NUM_OF_TS(row_in_mining_table[i]) < 4){
                        row_in_mining_table[i] = ADD_TS(row_in_mining_table[i], MIMIR_params->ts);
                        break;
                    }
                }
                if (timestamps_length == MIMIR_params->max_support){
                    /* no timestamp added, drop this request, it is too frequent */
                    if (!__MIMIR_remove_from_recording_hashtable(cp->type, r_m_struct, row_in_mining_table))
                        printf("ERROR move from recording table to mining table, first step remove from recording table failed, \
                               %s, mining table length %u\n", (char*)*row_in_mining_table, r_m_struct->mining_table->len);
                    
                    //                guint old_len = r_m_struct->mining_table->len;
                    /* for dataType c, now the pointer to string has been freed, so mining table entry is incorrent,
                     but it needs to be deleted, so it is OK
                     */
                    
                    g_array_remove_index_fast(r_m_struct->mining_table, -index-1);
                    
                    
                    // if array is moved, need to update hashtable
                    if (-index - 1 != (long) r_m_struct->mining_table->len ){          // r_m_struct->mining_table->len == old_len-1 &&
                        if (cp->type == 'l'){
                            g_hash_table_replace(r_m_struct->hashtable, row_in_mining_table, GINT_TO_POINTER(index));
                        }
                        else if (cp->type == 'c'){
                            gpointer gp = g_strdup((char*)*row_in_mining_table);
                            g_hash_table_insert(r_m_struct->hashtable, gp, GINT_TO_POINTER(index));
                        }
                    }
                    r_m_struct->num_of_entry_available_for_mining --;
                }
            }
            else {
                /* in recording table */
                row_in_recording_table = GET_ROW_IN_RECORDING_TABLE(MIMIR_params, index);
                gint64* current_row_in_recording_table = GET_ROW_IN_RECORDING_TABLE(MIMIR_params,
                                                                                    r_m_struct->recording_table_pointer-1);
                int i, timestamps_length = 0;
                
#ifdef SANITY_CHECK
                if (cp->type == 'l'){
                    if (*(gint64*)(cp->item_p) != row_in_recording_table[0]){
                        ERROR("Hashtable recording found position not correct %ld %ld\n",
                              *(gint64*)(cp->item_p), row_in_recording_table[0]);
                        abort();
                    }
                }
                else{
                    if (strcmp(cp->item_p, (char*)(row_in_recording_table[0]))!=0){
                        ERROR("Hashtable recording found position not correct %s %s\n",
                              (char*)(cp->item_p), (char*)(row_in_recording_table[0]));
                        abort();
                    }
                }
#endif
                
                for (i=1; i < r_m_struct->recording_table_row_len; i++){
                    timestamps_length += NUM_OF_TS(row_in_recording_table[i]);
                    if (NUM_OF_TS(row_in_recording_table[i]) < 4){
                        row_in_recording_table[i] = ADD_TS(row_in_recording_table[i], MIMIR_params->ts);
                        break;
                    }
                }
                
                if (timestamps_length == MIMIR_params->min_support-1){
                    /* time to move to mining table */
                    gint64 array_ele[r_m_struct->mining_table_row_len];
                    memcpy(array_ele, row_in_recording_table, sizeof(TS_REPRESENTATION) *
                           r_m_struct->recording_table_row_len);
                    // this is important as we don't clear the content of array after mining
                    memset(array_ele + r_m_struct->recording_table_row_len, 0,
                           sizeof(TS_REPRESENTATION) * (r_m_struct->mining_table_row_len
                                                        - r_m_struct->recording_table_row_len));
#ifdef SANITY_CHECK
                    if ((long) r_m_struct->mining_table->len >= MIMIR_params->mining_table_size){
                        /* if this happens, array will re-malloc, which will make
                         * the hashtable key not reliable when data_type is l */
                        ERROR("mining table length reaches limit, but does not mining, "
                               "entry %d, size %u, threshold %d\n",
                               r_m_struct->num_of_entry_available_for_mining,
                               r_m_struct->mining_table->len, MIMIR_params->mining_table_size);
                        abort();
                    }
#endif
                    g_array_append_val(r_m_struct->mining_table, array_ele);
                    r_m_struct->num_of_entry_available_for_mining ++;
                    
                    
                    if (index != r_m_struct->recording_table_pointer-1 &&
                        r_m_struct->recording_table_pointer >= 2){
                        // moved line is not the last entry in recording table
#ifdef SANITY_CHECK
                        if (row_in_recording_table == current_row_in_recording_table)
                            ERROR("FOUND SRC DEST same, ts %ld %p %p %ld %ld %d %ld\n",
                                   MIMIR_params->ts, row_in_recording_table,
                                   current_row_in_recording_table,
                                   *row_in_recording_table, *current_row_in_recording_table,
                                   index, r_m_struct->recording_table_pointer-1);
#endif
                        memcpy(row_in_recording_table, current_row_in_recording_table,
                               sizeof(TS_REPRESENTATION)*r_m_struct->recording_table_row_len);
                    }
                    if (r_m_struct->recording_table_pointer >= 2)
                        for (i=0; i<r_m_struct->recording_table_row_len; i++)
                            current_row_in_recording_table[i] = 0;
                    else{
                        // if current pointer points to 1, then don't move it, clear the row (that moves to mining table)
                        for (i=0; i<r_m_struct->recording_table_row_len; i++)
                            row_in_recording_table[i] = 0;
                    }
                    
                    
                    gint64* inserted_row_in_mining_table = GET_ROW_IN_MINING_TABLE(MIMIR_params, r_m_struct->mining_table->len-1);
                    if (cp->type == 'l'){
                        /* because we don't want to have zero as index, so we add one before taking negative,
                         in other words, the range of mining table index is -1 ~ -max_index-1, mapping to 0~max_index
                         */
#ifdef SANITY_CHECK 
                        if (*inserted_row_in_mining_table != *(gint64*)(cp->item_p)){
                            ERROR("current block %ld, moving mining row block %ld\n",
                                  *(gint64*)(cp->item_p), *inserted_row_in_mining_table);
                            abort();
                        }
#endif
                        g_hash_table_replace(r_m_struct->hashtable, inserted_row_in_mining_table,
                                             GINT_TO_POINTER(-((gint)r_m_struct->mining_table->len-1+1)));
//                        if (index != r_m_struct->recording_table_pointer && r_m_struct->recording_table_pointer >= 2)
                        // 0503 MODIFY
                        if (index != r_m_struct->recording_table_pointer - 1 && r_m_struct->recording_table_pointer >= 2)
                            // last entry in the recording table is moved up index position
                            g_hash_table_replace(r_m_struct->hashtable, row_in_recording_table, GINT_TO_POINTER(index));
                    }
                    else if (cp->type == 'c'){
                        gpointer gp1 = g_strdup((char*)(*inserted_row_in_mining_table));
                        /* use insert because we don't want to free the original key, instead free just passed key */
                        g_hash_table_insert(r_m_struct->hashtable, gp1,
                                            GINT_TO_POINTER(-((gint)r_m_struct->mining_table->len-1+1) ));
                        if (index != r_m_struct->recording_table_pointer - 1 && r_m_struct->recording_table_pointer >= 2){
                            gpointer gp2 = g_strdup((char*)(*row_in_recording_table));
                            g_hash_table_insert(r_m_struct->hashtable, gp2, GINT_TO_POINTER(index));
                        }
                    }
                    else{
                        ERROR("does not recognize data type\n");
                        abort();
                    }
                    
                    // one entry has been moved to mining table, shrinking recording table size by 1
                    if (r_m_struct->recording_table_pointer >= 2)
                        r_m_struct->recording_table_pointer --;
                }
            }
        }
    }
    if (r_m_struct->num_of_entry_available_for_mining >= MIMIR_params->mining_table_size ||
        (MIMIR_params->min_support == 1 && r_m_struct->num_of_entry_available_for_mining > MIMIR_params->mining_threshold/8)){
        __MIMIR_mining(MIMIR);
        r_m_struct->num_of_entry_available_for_mining = 0;
    }
}


static inline void __MIMIR_record_entry_min_support_1(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    struct recording_mining_struct *r_m_struct = MIMIR_params->record_mining_struct;
    
#ifdef TRACK_BLOCK 
    if (*(gint64*) (cp->item_p) == TRACK_BLOCK){
        int old_pos = GPOINTER_TO_INT( g_hash_table_lookup(r_m_struct->hashtable, cp->item_p));
        printf("insert %ld, old pos %d", TRACK_BLOCK, old_pos);
        if (old_pos == 0)
            printf("\n");
        else
            printf(", block at old_pos %ld\n", *(gint64*) GET_ROW_IN_MINING_TABLE(MIMIR_params, old_pos-1));
        
    }
    else {
        gint64 b = TRACK_BLOCK;
        int old_pos = GPOINTER_TO_INT( g_hash_table_lookup(r_m_struct->hashtable, &b));
        if (old_pos != 0){
            ERROR("ts %lu, checking %ld, %ld is found at pos %d\n", cp->ts, TRACK_BLOCK,
                  *(gint64*) GET_ROW_IN_MINING_TABLE(MIMIR_params, old_pos-1), old_pos);
            abort();
        }
    }
#endif
    
    int i;
    // check the item in hashtable for training
    gint index = GPOINTER_TO_INT( g_hash_table_lookup(r_m_struct->hashtable, cp->item_p) );
    if (index == 0){
        // the node is not in the recording/mining data, should be added
        gint64 array_ele[r_m_struct->mining_table_row_len];
        gpointer hash_key;
        if (cp->type == 'c'){
            array_ele[0] = (gint64) g_strdup(cp->item_p);
            hash_key = (void*)(array_ele[0]);
        }
        else{
            array_ele[0] = *(gint64*) (cp->item_p);
            hash_key = GET_ROW_IN_MINING_TABLE(MIMIR_params,
                                               r_m_struct->mining_table->len);
        }
        for (i=1; i<r_m_struct->mining_table_row_len; i++)
            array_ele[i] = 0;
        array_ele[1] = ADD_TS(array_ele[1], MIMIR_params->ts);

        g_array_append_val(r_m_struct->mining_table, array_ele);
        r_m_struct->num_of_entry_available_for_mining ++;

        g_hash_table_insert(r_m_struct->hashtable, hash_key,
                            GINT_TO_POINTER(r_m_struct->mining_table->len));       // all index is real row number + 1
#ifdef TRACK_BLOCK
        gint64 b = TRACK_BLOCK;
        int old_pos = GPOINTER_TO_INT( g_hash_table_lookup(r_m_struct->hashtable, &b));
        if (old_pos != 0){
            ERROR("HASH COLLISON: ts %lu, checking %ld, %ld is found at pos %d, just added %ld, ", cp->ts, TRACK_BLOCK,
                  *(gint64*) GET_ROW_IN_MINING_TABLE(MIMIR_params, old_pos-1), old_pos,
                  *(gint64*)hash_key);
            old_pos = GPOINTER_TO_INT( g_hash_table_lookup(r_m_struct->hashtable, hash_key));
            fprintf(stderr, "%ld hashed to pos %d\n", *(gint64*)hash_key, old_pos);
//            abort();
        }
#endif

#ifdef SANITY_CHECK
        gint64* row_in_mining_table = GET_ROW_IN_MINING_TABLE(MIMIR_params, r_m_struct->mining_table->len-1);
        if (cp->type == 'l'){
            if (*(gint64*)(cp->item_p) != row_in_mining_table[0]){
                ERROR("after inserting, hashtable mining not consistent %ld %ld\n",
                      *(gint64*)(cp->item_p), row_in_mining_table[0]);
                abort();
            }
        }
        else{
            if (strcmp(cp->item_p, (char*)(row_in_mining_table[0]))!=0){
                ERROR("after inserting, hashtable mining not consistent %s %s\n",
                      (char*)(cp->item_p), (char*)(row_in_mining_table[0]));
                abort();
            }
        }
#endif
    }
    
    else{
        /* in mining table */
        gint64* row_in_mining_table = GET_ROW_IN_MINING_TABLE(MIMIR_params, index-1);
        
#ifdef SANITY_CHECK
        if (cp->type == 'l'){
            if (*(gint64*)(cp->item_p) != row_in_mining_table[0]){
                ERROR("ts %lu, hashtable mining found position not correct %ld %ld\n",
                      cp->ts, *(gint64*)(cp->item_p), row_in_mining_table[0]);
//                abort();
            }
        }
        else{
            if (strcmp(cp->item_p, (char*)(row_in_mining_table[0]))!=0){
                ERROR("ts %lu, hashtable mining found position not correct %s %s\n",
                      cp->ts, (char*)(cp->item_p), (char*)(row_in_mining_table[0]));
//                abort();
            }
        }
#endif
        
        int timestamps_length = 0;
        
        for (i=1; i < r_m_struct->mining_table_row_len; i++){
            timestamps_length += NUM_OF_TS(row_in_mining_table[i]);
            if (NUM_OF_TS(row_in_mining_table[i]) < 4){
                row_in_mining_table[i] = ADD_TS(row_in_mining_table[i], MIMIR_params->ts);
                break;
            }
        }
        if (timestamps_length == MIMIR_params->max_support){
            /* no timestamp added, drop this request, it is too frequent */
            if (!__MIMIR_remove_from_recording_hashtable(cp->type, r_m_struct, row_in_mining_table))
                ERROR("move from recording table to mining table, first step remove from recording table failed, \
                       %s, mining table length %u\n", (char*)*row_in_mining_table, r_m_struct->mining_table->len);
            
            g_array_remove_index_fast(r_m_struct->mining_table, index-1);
            
            // if array is moved, need to update hashtable
            if (index - 1 != (long) r_m_struct->mining_table->len ){          // r_m_struct->mining_table->len == old_len-1 &&
                if (cp->type == 'l'){
                    g_hash_table_replace(r_m_struct->hashtable, row_in_mining_table, GINT_TO_POINTER(index));
                }
                else if (cp->type == 'c'){
                    gpointer gp = g_strdup((char*)*row_in_mining_table);
                    g_hash_table_insert(r_m_struct->hashtable, gp, GINT_TO_POINTER(index));
                }
            }
            r_m_struct->num_of_entry_available_for_mining --;
        }
    }
}



static inline void __MIMIR_prefetch(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    gint prefetch_table_index = GPOINTER_TO_INT(g_hash_table_lookup(MIMIR_params->prefetch_hashtable, cp->item_p));
    gint dim1 = (gint)floor(prefetch_table_index/(double)PREFETCH_TABLE_SHARD_SIZE);
    gint dim2 = prefetch_table_index % PREFETCH_TABLE_SHARD_SIZE * (MIMIR_params->prefetch_list_size+1);
    
    if (prefetch_table_index){
        gpointer old_cp_gp = cp->item_p;
        int i;
        for (i=1; i<MIMIR_params->prefetch_list_size+1; i++){
            // begin from 1 because index 0 is the label of originated request
            if (MIMIR_params->prefetch_table_array[dim1][dim2+i] == 0)
                break;
            if (cp->type == 'l')
                cp->item_p = &(MIMIR_params->prefetch_table_array[dim1][dim2+i]);
            else
                cp->item_p = (char*) (MIMIR_params->prefetch_table_array[dim1][dim2+i]);

            if (MIMIR_params->output_statistics)
                MIMIR_params->num_of_check += 1;
            
            // can't use MIMIR_check_element here
            if (MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp)){
                continue;
            }
            MIMIR_params->cache->core->__insert_element(MIMIR_params->cache, cp);
            while ( (long) MIMIR_params->cache->core->get_size(MIMIR_params->cache) > (long) MIMIR_params->cache->core->size)
                // use this because we need to record stat when evicting 
                __MIMIR_evict_element(MIMIR, cp);
            
            if (MIMIR_params->output_statistics){
                MIMIR_params->num_of_prefetch_mimir += 1;
                
                gpointer item_p;
                if (cp->type == 'l'){
                    item_p = (gpointer)g_new(guint64, 1);
                    *(guint64*)item_p = *(guint64*)(cp->item_p);
                }
                else
                    item_p = (gpointer)g_strdup((gchar*)(cp->item_p));

                g_hash_table_insert(MIMIR_params->prefetched_hashtable_mimir, item_p, GINT_TO_POINTER(1));
            }
        }
        cp->item_p = old_cp_gp;
    }

    
    
    
    
    
    // prefetch sequential      0925
    if (MIMIR_params->sequential_type == 1 && __MIMIR_check_sequential(MIMIR, cp)){
        gpointer old_gp = cp->item_p;
        gint64 next = *(gint64*) (cp->item_p) + 1;
        cp->item_p = &next;
        
        if (MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp)){
            MIMIR_params->cache->core->__update_element(MIMIR_params->cache, cp);
            cp->item_p = old_gp;
            return;
        }

        // use this because we need to record stat when evicting
        MIMIR_params->cache->core->__insert_element(MIMIR_params->cache, cp);
        while ( (long) MIMIR_params->cache->core->get_size(MIMIR_params->cache)
                        > MIMIR_params->cache->core->size)
            __MIMIR_evict_element(MIMIR, cp);
        
        
        if (MIMIR_params->output_statistics){
            MIMIR_params->num_of_prefetch_sequential += 1;
                
            gpointer item_p;
            item_p = (gpointer)g_new(gint64, 1);
            *(gint64*)item_p = next;
            
            g_hash_table_add(MIMIR_params->prefetched_hashtable_sequential, item_p);
        }
        cp->item_p = old_gp;
    }
}



gboolean MIMIR_check_element(struct_cache* MIMIR, cache_line* cp){
     
    // maybe we can move prefetch to insert, meaning prefetch only when cache miss
    // check element, record entry and prefetch element
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    if (MIMIR_params->output_statistics){
        if (g_hash_table_contains(MIMIR_params->prefetched_hashtable_mimir, cp->item_p)){
            MIMIR_params->hit_on_prefetch_mimir += 1;
            g_hash_table_remove(MIMIR_params->prefetched_hashtable_mimir, cp->item_p);
        }
        if (g_hash_table_contains(MIMIR_params->prefetched_hashtable_sequential, cp->item_p)){
            MIMIR_params->hit_on_prefetch_sequential += 1;
            g_hash_table_remove(MIMIR_params->prefetched_hashtable_sequential, cp->item_p);
        }

    }
 
    gboolean result = MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp);

    if (MIMIR_params->recording_loc != evict)
        __MIMIR_record_entry(MIMIR, cp);

    
    return result;
}


void __MIMIR_update_element(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    MIMIR_params->cache->core->__update_element(MIMIR_params->cache, cp);
}


void __MIMIR_evict_element(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    // used for debug
    // g_hash_table_foreach(MIMIR_params->prefetched_hashtable_mimir, check_prefetched_hashtable, MIMIR_params);
    if (MIMIR_params->output_statistics){
        gpointer gp;
        gp = MIMIR_params->cache->core->__evict_with_return(MIMIR_params->cache, cp);

        gint type = GPOINTER_TO_INT(g_hash_table_lookup(MIMIR_params->prefetched_hashtable_mimir, gp));
        if (type !=0 && type < MIMIR_params->cycle_time){
            // give one more chance
            gpointer new_key;
            if (cp->type == 'l'){
                new_key = g_new(gint64, 1);
                *(gint64*) new_key = *(gint64*) gp;
            }
            else {
                new_key = g_strdup(gp);
            }

            g_hash_table_insert(MIMIR_params->prefetched_hashtable_mimir, new_key, GINT_TO_POINTER(type+1));
            gpointer old_cp = cp->item_p;
            cp->item_p = gp;
            __MIMIR_insert_element(MIMIR, cp);
            cp->item_p = old_cp;
            __MIMIR_evict_element(MIMIR, cp);
        }
        else{
            gpointer old_gp = cp->item_p;
            cp->item_p = gp;
            // comment to disable record at evicting
            if (MIMIR_params->recording_loc == evict ||
                MIMIR_params->recording_loc == miss_evict){
                __MIMIR_record_entry(MIMIR, cp);
            }
            cp->item_p = old_gp;
            g_hash_table_remove(MIMIR_params->prefetched_hashtable_mimir, gp);
            g_hash_table_remove(MIMIR_params->prefetched_hashtable_sequential, gp);
        }
        g_free(gp);
    }
    else
        MIMIR_params->cache->core->__evict_element(MIMIR_params->cache, cp);
}



gboolean MIMIR_add_element(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    MIMIR_params->ts ++;

    if (MIMIR_params->cache->core->type == e_AMP){
        gboolean result = MIMIR_check_element(MIMIR, cp);
        AMP_add_element_no_eviction(MIMIR_params->cache, cp);
        __MIMIR_prefetch(MIMIR, cp);
        while ( (long) MIMIR_params->cache->core->get_size(MIMIR_params->cache) > MIMIR->core->size)
            __MIMIR_evict_element(MIMIR, cp);

        return result;
    }
    else{
        if (MIMIR_check_element(MIMIR, cp)){
            __MIMIR_update_element(MIMIR, cp);
            __MIMIR_prefetch(MIMIR, cp);
            return TRUE;
        }
        else{
            __MIMIR_insert_element(MIMIR, cp);
            __MIMIR_prefetch(MIMIR, cp);
            while ( (long) MIMIR_params->cache->core->get_size(MIMIR_params->cache) > MIMIR->core->size)
                __MIMIR_evict_element(MIMIR, cp);
            return FALSE;
        }
    }
}






gboolean MIMIR_check_element_only(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    return MIMIR_params->cache->core->check_element(MIMIR_params->cache, cp);
}




void __MIMIR_evict_element_only(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);

    // used for debug
    // g_hash_table_foreach(MIMIR_params->prefetched_hashtable_mimir, check_prefetched_hashtable, MIMIR_params);
        gpointer gp;
        gp = MIMIR_params->cache->core->__evict_with_return(MIMIR_params->cache, cp);
        
        gint type = GPOINTER_TO_INT(g_hash_table_lookup(MIMIR_params->prefetched_hashtable_mimir, gp));
        if (type !=0 && type < MIMIR_params->cycle_time){
            // give one more chance
            gpointer new_key;
            if (cp->type == 'l'){
                new_key = g_new(gint64, 1);
                *(gint64*) new_key = *(gint64*) gp;
            }
            else {
                new_key = g_strdup(gp);
            }
            if (MIMIR_params->output_statistics)
                g_hash_table_insert(MIMIR_params->prefetched_hashtable_mimir, new_key, GINT_TO_POINTER(type+1));
            gpointer old_cp = cp->item_p;
            cp->item_p = gp;
            __MIMIR_insert_element(MIMIR, cp);          // insert is same
            cp->item_p = old_cp;
            __MIMIR_evict_element_only(MIMIR, cp);
        }
        else{
            gpointer old_gp = cp->item_p;
            cp->item_p = gp;
            cp->item_p = old_gp;
            if (MIMIR_params->output_statistics){
                g_hash_table_remove(MIMIR_params->prefetched_hashtable_mimir, gp);
                g_hash_table_remove(MIMIR_params->prefetched_hashtable_sequential, gp);
            }
        }
        g_free(gp);
}




gboolean MIMIR_add_element_only(struct_cache* MIMIR, cache_line* cp){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    
    if (MIMIR_params->cache->core->type == e_AMP){
        gboolean result = MIMIR_check_element_only(MIMIR, cp);
        AMP_add_element_only_no_eviction(MIMIR_params->cache, cp);
        while ( (long) MIMIR_params->cache->core->get_size(MIMIR_params->cache) > MIMIR->core->size)
            __MIMIR_evict_element_only(MIMIR, cp);
        return result;
    }
    else{
        if (MIMIR_check_element_only(MIMIR, cp)){
            // new
            __MIMIR_update_element(MIMIR, cp);  // update is same
            return TRUE;
        }
        else{
            __MIMIR_insert_element(MIMIR, cp);  // insert is same 
            while ( (long) MIMIR_params->cache->core->get_size(MIMIR_params->cache) > MIMIR->core->size)
                __MIMIR_evict_element_only(MIMIR, cp);
            return FALSE;
        }
    }
}





gboolean MIMIR_add_element_withsize(struct_cache* MIMIR, cache_line* cp){
    int i;
    gboolean ret_val;
    if (MIMIR->core->block_unit_size != 0 && cp->disk_sector_size != 0){

        *(gint64*)(cp->item_p) = (gint64) (*(gint64*)(cp->item_p) *
                                           cp->disk_sector_size /
                                           MIMIR->core->block_unit_size);
    }

    
#ifdef TRACK_BLOCK
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    struct recording_mining_struct *r_m_struct = MIMIR_params->record_mining_struct;
    static int last_found_pos = 0;
    gint64 b = TRACK_BLOCK;
    if (*(gint64*)(cp->item_p) == TRACK_BLOCK){
        printf("ts %lu, track block add\n", cp->ts);
    }
    int old_pos = GPOINTER_TO_INT( g_hash_table_lookup(r_m_struct->hashtable, &b));
    if (old_pos != 0){
        if (old_pos > 0){
            gint64* row = GET_ROW_IN_RECORDING_TABLE(MIMIR_params, old_pos);
            if (*row != TRACK_BLOCK){
                ERROR("the row (%d) from recording table does not match track block, it is %ld\n", old_pos, *row);
                abort();
            }
        }
        else if (old_pos < 0){
            gint64* row = GET_ROW_IN_MINING_TABLE(MIMIR_params, -old_pos-1);
            if (*row != TRACK_BLOCK){
                ERROR("the row (%d) from recording table does not match track block, it is %ld\n", old_pos, *row);
                abort();
            }
        }
        if (last_found_pos != old_pos){
            ERROR("ts %lu, found track block change in hashtable, pos %d\n", cp->ts, old_pos);
            last_found_pos = old_pos;

        }
    }
    else {
        if (last_found_pos != 0){
            printf("ts %lu, track block %ld disappeared, might because of mining\n", cp->ts, TRACK_BLOCK);
        }
    }
#endif

    
    ret_val = MIMIR_add_element(MIMIR, cp);
    
    
    if (MIMIR->core->block_unit_size != 0 && cp->disk_sector_size != 0){
        int n = (int)ceil((double) cp->size/MIMIR->core->block_unit_size);
    
        for (i=0; i<n-1; i++){
            (*(guint64*)(cp->item_p)) ++;
            MIMIR_add_element_only(MIMIR, cp);
        }
        *(gint64*)(cp->item_p) -= (n-1);
    }

    return ret_val;
}





void MIMIR_destroy(struct_cache* cache){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(cache->cache_params);

    g_hash_table_destroy(MIMIR_params->prefetch_hashtable);
    g_hash_table_destroy(MIMIR_params->record_mining_struct->hashtable);
    g_free(MIMIR_params->record_mining_struct->recording_table);
    g_array_free(MIMIR_params->record_mining_struct->mining_table, TRUE);
    g_free(MIMIR_params->record_mining_struct);

    int i = 0;
    gint max_num_of_shards_in_prefetch_table = (gint) (MIMIR_params->max_metadata_size /
                                                       (PREFETCH_TABLE_SHARD_SIZE * MIMIR_params->prefetch_list_size));

    
    while (i<max_num_of_shards_in_prefetch_table){
        if (MIMIR_params->prefetch_table_array[i]){
            if (cache->core->data_type == 'c'){
                int j=0;
                for (j=0; j<PREFETCH_TABLE_SHARD_SIZE*(1+MIMIR_params->prefetch_list_size); j++)
                    if ((char*)MIMIR_params->prefetch_table_array[i][j])
                        g_free((char*)MIMIR_params->prefetch_table_array[i][j]);
            }
            g_free(MIMIR_params->prefetch_table_array[i]);
        }
        else
            break;
        i++;
    }
    g_free(MIMIR_params->prefetch_table_array);
    
    if (MIMIR_params->output_statistics){
        g_hash_table_destroy(MIMIR_params->prefetched_hashtable_mimir);
        g_hash_table_destroy(MIMIR_params->prefetched_hashtable_sequential);
    }
    MIMIR_params->cache->core->destroy(MIMIR_params->cache);            // 0921
    cache_destroy(cache);
}

void MIMIR_destroy_unique(struct_cache* cache){
    /* the difference between destroy_unique and destroy
     is that the former one only free the resources that are
     unique to the cache, freeing these resources won't affect
     other caches copied from original cache
     in Optimal, next_access should not be freed in destroy_unique,
     because it is shared between different caches copied from the original one.
     */
    
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(cache->cache_params);
    g_hash_table_destroy(MIMIR_params->prefetch_hashtable);
    g_hash_table_destroy(MIMIR_params->record_mining_struct->hashtable);
    
    g_free(MIMIR_params->record_mining_struct->recording_table);
    g_array_free(MIMIR_params->record_mining_struct->mining_table, TRUE);
    g_free(MIMIR_params->record_mining_struct);

    int i = 0;
    gint max_num_of_shards_in_prefetch_table = (gint) (MIMIR_params->max_metadata_size /
                                                       (PREFETCH_TABLE_SHARD_SIZE * MIMIR_params->prefetch_list_size));

    while (i<max_num_of_shards_in_prefetch_table){
        if (MIMIR_params->prefetch_table_array[i]){
            if (cache->core->data_type == 'c'){
                int j=0;
                for (j=0; j<PREFETCH_TABLE_SHARD_SIZE*(1+MIMIR_params->prefetch_list_size); j++)
                    if ((char*)MIMIR_params->prefetch_table_array[i][j])
                        g_free((char*)MIMIR_params->prefetch_table_array[i][j]);
            }
            g_free(MIMIR_params->prefetch_table_array[i]);
        }
        else
            break;
        i++;
    }
    g_free(MIMIR_params->prefetch_table_array);

    if (MIMIR_params->output_statistics){
        g_hash_table_destroy(MIMIR_params->prefetched_hashtable_mimir);
        g_hash_table_destroy(MIMIR_params->prefetched_hashtable_sequential);
    }
    MIMIR_params->cache->core->destroy_unique(MIMIR_params->cache);         // 0921
    cache_destroy_unique(cache);
}


struct_cache* MIMIR_init(guint64 size, char data_type, int block_size, void* params){
#ifdef SANITY_CHECK
    printf("SANITY_CHECK enabled\n");
#endif
    struct_cache *cache = cache_init(size, data_type, block_size);
    cache->cache_params = g_new0(struct MIMIR_params, 1);
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(cache->cache_params);
    
    cache->core->type               = e_mimir;
    cache->core->cache_init         = MIMIR_init;
    cache->core->destroy            = MIMIR_destroy;
    cache->core->destroy_unique     = MIMIR_destroy_unique;
    cache->core->add_element        = MIMIR_add_element;
    cache->core->check_element      = MIMIR_check_element;
    cache->core->__insert_element   = __MIMIR_insert_element;
    cache->core->__evict_element    = __MIMIR_evict_element;
    cache->core->get_size           = MIMIR_get_size;
    cache->core->cache_init_params  = params;
    cache->core->add_element_only   = MIMIR_add_element_only;
    cache->core->add_element_withsize = MIMIR_add_element_withsize; 
    
    struct MIMIR_init_params *init_params = (struct MIMIR_init_params*) params;
    
    
    MIMIR_params->item_set_size     = init_params->item_set_size;
    MIMIR_params->max_support       = init_params->max_support;
    MIMIR_params->min_support       = init_params->min_support;
    MIMIR_params->confidence        = init_params->confidence;
    MIMIR_params->prefetch_list_size= init_params->prefetch_list_size;
    MIMIR_params->sequential_type   = init_params->sequential_type;
    MIMIR_params->sequential_K      = init_params->sequential_K;
    MIMIR_params->cycle_time        = init_params->cycle_time;
    MIMIR_params->block_size        = init_params->block_size;
    MIMIR_params->recording_loc     = init_params->recording_loc;
    MIMIR_params->mining_threshold  = init_params->mining_threshold;

    
//    MIMIR_params->mining_table_size  = (gint)(MINING_THRESHOLD / pow(2, MIMIR_params->min_support));
    MIMIR_params->mining_table_size  = (gint)(init_params->mining_threshold/MIMIR_params->min_support);
    MIMIR_params->record_mining_struct = g_new0(struct recording_mining_struct, 1);
    struct recording_mining_struct* r_m_struct = MIMIR_params->record_mining_struct;
    r_m_struct->recording_table_pointer = 1;
    r_m_struct->recording_table_row_len = (gint)ceil((double) MIMIR_params->min_support / (double)4) + 1;
    r_m_struct->mining_table_row_len    = (gint)ceil((double) MIMIR_params->max_support / (double)4) + 1;
    r_m_struct->mining_table = g_array_sized_new(FALSE, TRUE,
                                                 sizeof(gint64) * r_m_struct->mining_table_row_len,
                                                 MIMIR_params->mining_table_size);
    
    
    
    MIMIR_params->max_metadata_size = (gint64) (init_params->block_size * size * init_params->max_metadata_size);
    gint max_num_of_shards_in_prefetch_table = (gint) (MIMIR_params->max_metadata_size /
                                                       (PREFETCH_TABLE_SHARD_SIZE * init_params->prefetch_list_size));
    MIMIR_params->current_prefetch_table_pointer = 1;
    MIMIR_params->prefetch_table_fully_allocatd = FALSE;
    // always save to size+1 position, and enlarge table when size%shards_size == 0
    MIMIR_params->prefetch_table_array = g_new0(gint64*, max_num_of_shards_in_prefetch_table);
    MIMIR_params->prefetch_table_array[0] = g_new0(gint64, PREFETCH_TABLE_SHARD_SIZE*(MIMIR_params->prefetch_list_size+1));
    
    
    /* now adjust the cache size by deducting current meta data size
        8 is the size of storage for block, 4 is the size of storage for index to array */
    MIMIR_params->current_metadata_size = (init_params->max_support * 2 + 8 + 4) * MIMIR_params->mining_table_size  +
                                            max_num_of_shards_in_prefetch_table * 8 +
                                            PREFETCH_TABLE_SHARD_SIZE * (MIMIR_params->prefetch_list_size*8 + 8 + 4) ;

    if (MIMIR_params->max_support != 1){
        r_m_struct->num_of_rows_in_recording_table = (gint)(size * MIMIR_params->block_size * RECORDING_TABLE_MAXIMAL /
                                                            ( (gint)ceil((double) MIMIR_params->min_support/(double)2) * 2 + 8 + 4));
        r_m_struct->recording_table = g_new0(gint64, r_m_struct->num_of_rows_in_recording_table
                                                        * r_m_struct->recording_table_row_len);        // this should begins with 1
        MIMIR_params->current_metadata_size += ( ((gint)ceil((double) init_params->min_support/(double)4+1) * 8 + 4) *
                                                r_m_struct->num_of_rows_in_recording_table);
    }

    
    size = size - (gint)(MIMIR_params->current_metadata_size/init_params->block_size);

    MIMIR_params->ts = 0;

    MIMIR_params->output_statistics = init_params->output_statistics;
    MIMIR_params->output_statistics = 1;
    
    
    MIMIR_params->hit_on_prefetch_mimir = 0;
    MIMIR_params->hit_on_prefetch_sequential = 0;
    MIMIR_params->num_of_prefetch_mimir = 0;
    MIMIR_params->num_of_prefetch_sequential = 0;
    MIMIR_params->num_of_check = 0;
    
    
    if (strcmp(init_params->cache_type, "LRU") == 0)
        MIMIR_params->cache = LRU_init(size, data_type, block_size, NULL);
    else if (strcmp(init_params->cache_type, "FIFO") == 0)
        MIMIR_params->cache = fifo_init(size, data_type, block_size, NULL);
    else if (strcmp(init_params->cache_type, "LFU") == 0)       // use LFU_fast
        MIMIR_params->cache = LFU_fast_init(size, data_type, block_size, NULL);
    else if (strcmp(init_params->cache_type, "AMP") == 0){
        struct AMP_init_params *AMP_init_params = g_new0(struct AMP_init_params, 1);
        AMP_init_params->APT            = 4;
        AMP_init_params->p_threshold    = init_params->AMP_pthreshold;
        AMP_init_params->read_size      = 1;
        AMP_init_params->K              = init_params->sequential_K;
        MIMIR_params->cache = AMP_init(size, data_type, block_size, AMP_init_params);
    }
    else if (strcmp(init_params->cache_type, "Optimal") == 0){
        struct optimal_init_params *optimal_init_params = g_new(struct optimal_init_params, 1);
        optimal_init_params->reader = NULL;
        MIMIR_params->cache = NULL;
        ;
    }
    else{
        fprintf(stderr, "can't recognize cache type: %s\n", init_params->cache_type);
        MIMIR_params->cache = LRU_init(size, data_type, block_size, NULL);
    }

    

    if (data_type == 'l'){
        r_m_struct->hashtable = g_hash_table_new_full(g_int64_hash, g_int_equal, NULL, NULL);
        MIMIR_params->prefetch_hashtable = g_hash_table_new_full(g_int64_hash, g_int64_equal, NULL, NULL);

        if (MIMIR_params->output_statistics){
            MIMIR_params->prefetched_hashtable_mimir = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                                             simple_g_key_value_destroyer,
                                                                             NULL);
            MIMIR_params->prefetched_hashtable_sequential = g_hash_table_new_full(g_int64_hash, g_int64_equal,
                                                                             simple_g_key_value_destroyer,
                                                                             NULL);
        }
    }
    else if (data_type == 'c'){
        r_m_struct->hashtable = g_hash_table_new_full(g_str_hash, g_str_equal, simple_g_key_value_destroyer, NULL);
        MIMIR_params->prefetch_hashtable = g_hash_table_new_full(g_str_hash, g_str_equal, NULL, NULL);

        if (MIMIR_params->output_statistics){
            MIMIR_params->prefetched_hashtable_mimir = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                             simple_g_key_value_destroyer,
                                                                             NULL);
            MIMIR_params->prefetched_hashtable_sequential = g_hash_table_new_full(g_str_hash, g_str_equal,
                                                                                  simple_g_key_value_destroyer,
                                                                                  NULL);
        }
    }
    else{
        g_error("does not support given data type: %c\n", data_type);
    }
    
    return cache;
}



gint mining_table_entry_cmp (gconstpointer a, gconstpointer b){
    return (gint)GET_NTH_TS(a, 1) - (gint)GET_NTH_TS(b, 1);
}



void __MIMIR_mining(struct_cache* MIMIR){
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    struct recording_mining_struct *r_m_struct = MIMIR_params->record_mining_struct;
#ifdef PROFILING
    GTimer *timer = g_timer_new();
    gulong microsecond;
    g_timer_start(timer);
#endif
    
    __MIMIR_aging(MIMIR);
#ifdef PROFILING
    printf("ts: %lu, aging takes %lf seconds\n", MIMIR_params->ts, g_timer_elapsed(timer, &microsecond));
#endif
    

    int i, j, k;

    /* first sort mining table, then do the mining */
    /* first remove all elements from hashtable, otherwise after sort, it will mess up for data_type l 
        but we can't do this for dataType c, otherwise the string will be freed during remove in hashtable 
     */
    gint64* item = (gint64*)r_m_struct->mining_table->data;
    if (MIMIR->core->data_type == 'l'){
        for (i=0; i<(int) r_m_struct->mining_table->len; i++){
            g_hash_table_remove(r_m_struct->hashtable, item);
            item += r_m_struct->mining_table_row_len;
        }
    }
    
    g_array_sort(r_m_struct->mining_table, mining_table_entry_cmp);

    
    gboolean associated_flag, first_flag;
    gint64* item1, *item2;
    gint num_of_ts1, num_of_ts2, shorter_length;
    for (i=0; i < (long) r_m_struct->mining_table->len-1; i++){
        item1 = GET_ROW_IN_MINING_TABLE(MIMIR_params, i);
        num_of_ts1 = __MIMIR_get_total_num_of_ts(item1, r_m_struct->mining_table_row_len);
        first_flag = TRUE;
        
        for (j=i+1; j < (long) r_m_struct->mining_table->len; j++){
            item2 = GET_ROW_IN_MINING_TABLE(MIMIR_params, j);

            // check first timestamp
            if ( GET_NTH_TS(item2, 1) - GET_NTH_TS(item1, 1) > MIMIR_params->item_set_size)
                break;
            num_of_ts2 = __MIMIR_get_total_num_of_ts(item2, r_m_struct->mining_table_row_len);
            
            if (ABS( num_of_ts1 - num_of_ts2) > MIMIR_params->confidence){
                continue;
            }
            
            shorter_length = MIN(num_of_ts1, num_of_ts2);
            
            associated_flag = FALSE;
            if (first_flag){
                associated_flag = TRUE;
                first_flag = FALSE;
            }
            // is next line useless??
            if (shorter_length == 1 && ABS(GET_NTH_TS(item1, 1) - GET_NTH_TS(item2, 1)) == 1)
                associated_flag = TRUE;
            
            gint error = 0;
            for (k=1; k<shorter_length; k++){
                 if ( ABS(GET_NTH_TS(item1, k) - GET_NTH_TS(item2, k)) > MIMIR_params->item_set_size){
                    error ++;
                    if (error > MIMIR_params->confidence){
                        associated_flag = FALSE;
                        break;
                    }
                }

                if ( ABS(GET_NTH_TS(item1, k) - GET_NTH_TS(item2, k)) == 1 ){
                    associated_flag = TRUE;
                }
            }
            if (associated_flag){
                // finally, add to prefetch table
                if (MIMIR->core->data_type == 'l')
                    mimir_add_to_prefetch_table(MIMIR, item1, item2);
                else if (MIMIR->core->data_type == 'c')
                    mimir_add_to_prefetch_table(MIMIR, (char*)*item1, (char*)*item2);
            }
        }
        if (MIMIR->core->data_type == 'c')
            if (!g_hash_table_remove(r_m_struct->hashtable, (char*)*item1)){
                printf("ERROR remove mining table entry, but not in hash %s\n", (char*)*item1);
                exit(1);
            }
    }
    // delete last element
    if (MIMIR->core->data_type == 'c'){
        item1 = GET_ROW_IN_MINING_TABLE(MIMIR_params, i);
        if (!g_hash_table_remove(r_m_struct->hashtable, (char*)*item1)){
            printf("ERROR remove mining table entry, but not in hash %s\n", (char*)*item1);
            exit(1);
        }
    }
    
    
    // may be just following?
    r_m_struct->mining_table->len = 0;
    
    
    
#ifdef PROFILING
    printf("ts: %lu, clearing training data takes %lf seconds\n", MIMIR_params->ts, g_timer_elapsed(timer, &microsecond));
    g_timer_stop(timer);
    g_timer_destroy(timer);
#endif
}



void mimir_add_to_prefetch_table(struct_cache* MIMIR, gpointer gp1, gpointer gp2){
    /** currently prefetch table can only support up to 2^31 entries, and this function assumes the platform is 64 bit */

    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    
    gint prefetch_table_index = GPOINTER_TO_INT(g_hash_table_lookup(MIMIR_params->prefetch_hashtable, gp1));
    gint dim1 = (gint)floor(prefetch_table_index/(double)PREFETCH_TABLE_SHARD_SIZE);
    gint dim2 = prefetch_table_index % PREFETCH_TABLE_SHARD_SIZE * (MIMIR_params->prefetch_list_size+1);
    
    // insert into prefetch hashtable
    int i;
    if (prefetch_table_index){
        // already have an entry in prefetch table, just add to that entry
        gboolean insert = TRUE;
        if (MIMIR->core->data_type == 'l'){
            for (i=1; i<MIMIR_params->prefetch_list_size+1; i++){
                // if this element is already in the array, then don't need add again
                // ATTENTION: the following assumes a 64 bit platform
#ifdef SANITY_CHECK
                if (MIMIR_params->prefetch_table_array[dim1][dim2] != *(gint64*)(gp1)){
                    fprintf(stderr, "ERROR prefetch table pos wrong %ld %ld, dim %d %d\n",
                            *(gint64*)gp1, MIMIR_params->prefetch_table_array[dim1][dim2], dim1, dim2);
                    exit(1);
                }
#endif
                if ((MIMIR_params->prefetch_table_array[dim1][dim2+i]) == 0)
                    break;
                if ((MIMIR_params->prefetch_table_array[dim1][dim2+i]) == *(gint64*)(gp2))
                    /* update score here, not implemented yet */
                    insert = FALSE;
            }
        }
        else{
            for (i=1; i<MIMIR_params->prefetch_list_size+1; i++){
                // if this element is already in the cache, then don't need prefetch again
                if ( strcmp((char*)(MIMIR_params->prefetch_table_array[dim1][dim2]), gp1) != 0){
                    fprintf(stderr, "ERROR prefetch table pos wrong\n");
                    exit(1);
                }
                if ((MIMIR_params->prefetch_table_array[dim1][dim2+i]) == 0)
                    break;
                if ( strcmp((gchar*)(MIMIR_params->prefetch_table_array[dim1][dim2+i]), (gchar*)gp2) == 0 )
                    /* update score here, not implemented yet */
                    insert = FALSE;
            }
        }
        
        if (insert){
            if (i == MIMIR_params->prefetch_list_size+1){
                // list full, randomly pick one for replacement
//                i = rand()%MIMIR_params->prefetch_list_size + 1;
//                if (MIMIR->core->data_type == 'c'){
//                    g_free((gchar*)(MIMIR_params->prefetch_table_array[dim1][dim2+i]));
//                }
                
                // use FIFO
                int j;
                for (j=2; j<MIMIR_params->prefetch_list_size+1; j++)
                    MIMIR_params->prefetch_table_array[dim1][dim2+j-1] = MIMIR_params->prefetch_table_array[dim1][dim2+j];
                i = MIMIR_params->prefetch_list_size;
                
                
            }
            // new add at position i
            if (MIMIR->core->data_type == 'c'){
                char* key2 = g_strdup((gchar*)gp2);
                MIMIR_params->prefetch_table_array[dim1][dim2+i] = (gint64)key2;
            }
            else{
                MIMIR_params->prefetch_table_array[dim1][dim2+i] = *(gint64*)(gp2);
            }
        }
    }
    else{
        // does not have entry, need to add a new entry
        MIMIR_params->current_prefetch_table_pointer ++;
        dim1 = (gint)floor(MIMIR_params->current_prefetch_table_pointer/(double)PREFETCH_TABLE_SHARD_SIZE);
        dim2 = MIMIR_params->current_prefetch_table_pointer % PREFETCH_TABLE_SHARD_SIZE *
                                (MIMIR_params->prefetch_list_size + 1);
        
        /* check whether prefetch table is fully allocated, if True, we are going to replace the 
            entry at current_prefetch_table_pointer by set the entry it points to as 0, 
            delete from prefetch_hashtable and add new entry */
        if (MIMIR_params->prefetch_table_fully_allocatd){
            if (MIMIR->core->data_type == 'l')
                g_hash_table_remove(MIMIR_params->prefetch_hashtable,
                                    &(MIMIR_params->prefetch_table_array[dim1][dim2]));
            else
                g_hash_table_remove(MIMIR_params->prefetch_hashtable,
                                    (char*)(MIMIR_params->prefetch_table_array[dim1][dim2]));

            
            if (MIMIR->core->data_type == 'c'){
                int i;
                for (i=0; i<MIMIR_params->prefetch_list_size+1; i++){
                    g_free((gchar*)MIMIR_params->prefetch_table_array[dim1][dim2+i]);
                }
            }
                
            memset(&(MIMIR_params->prefetch_table_array[dim1][dim2]), 0,
                   sizeof(gint64) * (MIMIR_params->prefetch_list_size+1));
        }
        
        gpointer gp1_dup;
        if (MIMIR->core->data_type == 'l')
            gp1_dup = &(MIMIR_params->prefetch_table_array[dim1][dim2]);
        else
            gp1_dup = (gpointer)g_strdup((gchar*)(gp1));
        
        if (MIMIR->core->data_type == 'c'){
            char* key2 = g_strdup((gchar*)gp2);
            MIMIR_params->prefetch_table_array[dim1][dim2+1] = (gint64)key2;
            MIMIR_params->prefetch_table_array[dim1][dim2]   = (gint64)gp1_dup;
        }
        else{
            MIMIR_params->prefetch_table_array[dim1][dim2+1] = *(gint64*)(gp2);
            MIMIR_params->prefetch_table_array[dim1][dim2]   = *(gint64*)(gp1);
        }

#ifdef SANITY_CHECK
        //make sure gp1 is not in prefetch_hashtable
        if (g_hash_table_contains(MIMIR_params->prefetch_hashtable, gp1)){
            gpointer gp = g_hash_table_lookup(MIMIR_params->prefetch_hashtable, gp1);
            printf("ERROR datatype %c, contains %ld %s, value %d, %d\n",
                   MIMIR->core->data_type, *(gint64*)gp1,
                   (char*)gp1, GPOINTER_TO_INT(gp), prefetch_table_index);
        }
#endif
        
        g_hash_table_insert(MIMIR_params->prefetch_hashtable, gp1_dup,
                            GINT_TO_POINTER(MIMIR_params->current_prefetch_table_pointer));
        
        
        // check current shard is full or not
        if ( (MIMIR_params->current_prefetch_table_pointer +1) % PREFETCH_TABLE_SHARD_SIZE == 0){
            /* need to allocate a new shard for prefetch table */
            if (MIMIR_params->current_metadata_size +
                    PREFETCH_TABLE_SHARD_SIZE * (MIMIR_params->prefetch_list_size * 8 +8 + 4)
                            < MIMIR_params->max_metadata_size){
                MIMIR_params->prefetch_table_array[dim1+1] = g_new0(gint64, PREFETCH_TABLE_SHARD_SIZE *
                                                                    (MIMIR_params->prefetch_list_size + 1));
                gint required_meta_data_size = PREFETCH_TABLE_SHARD_SIZE * (MIMIR_params->prefetch_list_size * 8 + 8 + 4);
                MIMIR_params->current_metadata_size += required_meta_data_size;
                MIMIR_params->cache->core->size = MIMIR->core->size -
                                                    (gint)((MIMIR_params->current_metadata_size) /MIMIR_params->block_size);
            }
            else{
                MIMIR_params->prefetch_table_fully_allocatd = TRUE;
                MIMIR_params->current_prefetch_table_pointer = 1;
            }
        }
    }
}





void __MIMIR_aging(struct_cache* MIMIR){
    
    ;
}

gint64 MIMIR_get_size(struct_cache* cache){
    struct MIMIR_params* mimir_params = (struct MIMIR_params*)(cache->cache_params);
    return (gint64) mimir_params->cache->core->get_size(mimir_params->cache);
}



void prefetch_node_destroyer(gpointer data){
    g_ptr_array_free((GPtrArray*)data, TRUE);}

void prefetch_array_node_destroyer(gpointer data){
    g_free(data);
}


void __MIMIR_mining_test(struct_cache* MIMIR){
    // new function for different approaches selecting strong associations
    struct MIMIR_params* MIMIR_params = (struct MIMIR_params*)(MIMIR->cache_params);
    struct recording_mining_struct *r_m_struct = MIMIR_params->record_mining_struct;
#ifdef PROFILING
    GTimer *timer = g_timer_new();
    gulong microsecond;
    g_timer_start(timer);
#endif
    
    __MIMIR_aging(MIMIR);
#ifdef PROFILING
    printf("ts: %lu, aging takes %lf seconds\n", MIMIR_params->ts, g_timer_elapsed(timer, &microsecond));
#endif
    
    
    int i, j, k;
    double gmean_min = 0, gmean;
    gint64* min_ptr = NULL;
    
    /* first sort mining table, then do the mining */
    /* first remove all elements from hashtable, otherwise after sort, it will mess up for data_type l
     but we can't do this for dataType c, otherwise the string will be freed during remove in hashtable
     */
    gint64* item = (gint64*)r_m_struct->mining_table->data;
    if (MIMIR->core->data_type == 'l'){
        for (i=0; i<(int) r_m_struct->mining_table->len; i++){
            g_hash_table_remove(r_m_struct->hashtable, item);
            item += r_m_struct->mining_table_row_len;
        }
    }
    
    g_array_sort(r_m_struct->mining_table, mining_table_entry_cmp);
    
    
//    gboolean associated_flag, first_flag;
    gint64* item1, *item2;
    gint num_of_ts1, num_of_ts2, shorter_length;
    for (i=0; i < (long) r_m_struct->mining_table->len-1; i++){
        gmean_min = 0;
        item1 = GET_ROW_IN_MINING_TABLE(MIMIR_params, i);
        num_of_ts1 = __MIMIR_get_total_num_of_ts(item1, r_m_struct->mining_table_row_len);
        
        for (j=i+1; j < (long) r_m_struct->mining_table->len; j++){
            item2 = GET_ROW_IN_MINING_TABLE(MIMIR_params, j);
            
            // check first timestamp
            if ( GET_NTH_TS(item2, 1) - GET_NTH_TS(item1, 1) > MIMIR_params->item_set_size)
                break;
            num_of_ts2 = __MIMIR_get_total_num_of_ts(item2, r_m_struct->mining_table_row_len);
            
            if (ABS( num_of_ts1 - num_of_ts2) > MIMIR_params->confidence){
                continue;
            }
            
            shorter_length = MIN(num_of_ts1, num_of_ts2);
            
            gmean = 1;
            for (k=0; k<shorter_length; k++)
                gmean *= ABS(GET_NTH_TS(item1, k) - GET_NTH_TS(item2, k));
            gmean = pow(gmean, 1.0/shorter_length);
            if (gmean_min < 0.000000001 || gmean < gmean_min){
                gmean_min = gmean;
                min_ptr = item;
            }
        }
        
        if (gmean_min < MIMIR_params->item_set_size/2+1){
            // finally, add to prefetch table
            if (MIMIR->core->data_type == 'l')
                mimir_add_to_prefetch_table(MIMIR, item1, min_ptr);
            else if (MIMIR->core->data_type == 'c')
                mimir_add_to_prefetch_table(MIMIR, (char*)*item1, (char*)*min_ptr);
        }
        
        
        if (MIMIR->core->data_type == 'c')
            if (!g_hash_table_remove(r_m_struct->hashtable, (char*)*item1)){
                printf("ERROR remove mining table entry, but not in hash %s\n", (char*)*item1);
                exit(1);
            }
    }
    // delete last element
    if (MIMIR->core->data_type == 'c'){
        item1 = GET_ROW_IN_MINING_TABLE(MIMIR_params, i);
        if (!g_hash_table_remove(r_m_struct->hashtable, (char*)*item1)){
            printf("ERROR remove mining table entry, but not in hash %s\n", (char*)*item1);
            exit(1);
        }
    }
    
    
    // may be just following?
    r_m_struct->mining_table->len = 0;
    
    
    
#ifdef PROFILING
    printf("ts: %lu, clearing training data takes %lf seconds\n", MIMIR_params->ts, g_timer_elapsed(timer, &microsecond));
    g_timer_stop(timer);
    g_timer_destroy(timer);
#endif
}


#ifdef __cplusplus
}
#endif





