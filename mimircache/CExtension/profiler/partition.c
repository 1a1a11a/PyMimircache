

#include "generalProfiler.h" 
#include "partition.h"

#include "reader.h"
#include "FIFO.h"
#include "Optimal.h"
#include "LRU_K.h"
#include "LRU.h"
#include "python_wrapper.h"
#include "LRU_dataAware.h"
#include "AMP.h"




/* this module is not reentrant-safe */


/************************* Util functions **************************/

partition_t* init_partition_t(uint8_t n_partitions, uint64_t cache_size){
    partition_t *partitions         =   g_new0(partition_t, 1);
    partitions->cache_size          =   cache_size;
    partitions->n_partitions        =   n_partitions;
    partitions->partition_history   =   g_new(GArray*, n_partitions);
    partitions->current_partition   =   g_new0(uint64_t, n_partitions);
    int i;
    for (i=0; i<n_partitions; i++)
        partitions->partition_history[i] = g_array_new(FALSE, FALSE, sizeof(double));
    return partitions;
}

void free_partition_t(partition_t *partition){
    int i;
    for (i=0; i<partition->n_partitions; i++){
        g_array_free(partition->partition_history[i], TRUE);
    }
    g_free(partition->current_partition);
    g_free(partition->partition_history);
    g_free(partition);
}



/*********************** partition related function *************************/
partition_t* get_partition(READER* reader, struct cache* cache, uint8_t n_partitions){
    
    partition_t *partitions = init_partition_t(n_partitions, cache->core->size);
    
    // create cache line struct and initialization
    cache_line* cp      =       new_cacheline();
    cp->type            =       reader->data_type;
//    cp->content         =       g_new0(gchar, 1);
//    cp->size_of_content =       1; 
    
 
    gboolean  (*add_element)   (struct cache*, cache_line*)     =   cache->core->add_element;
    gboolean  (*check_element) (struct cache*, cache_line*)     =   cache->core->check_element;
    
    // newly added 0912, may not work for all cache
    void      (*insert_element)(struct cache*, cache_line*)     =   cache->core->__insert_element;
    void      (*update_element)(struct cache*, cache_line*)     =   cache->core->__update_element;
    void      (*evict_element) (struct cache*, cache_line*)     =   cache->core->__evict_element;
    gpointer  (*evict_with_return)(struct cache*, cache_line*)  =   cache->core->__evict_with_return;
    guint64   (*get_size)      (struct cache*)                  =   cache->core->get_size;

    
    char* key;
    int i;
    guint64 current_size, cache_size;
    double percent;
    
    struct optimal_params* optimal_params;
    if (cache->core->type == e_Optimal)
        optimal_params = (struct optimal_params*)(cache->cache_params);

    
    cache_size = cache->core->size;
    
    read_one_element(reader, cp);
    
    while (cp->valid){
        if (check_element(cache, cp))
            update_element(cache, cp);
        else{
            insert_element(cache, cp);
            partitions->current_partition[(cp->item)[0]-'A'] ++;
        }
        current_size = get_size(cache);
        if (current_size > cache->core->size){
            key = evict_with_return(cache, cp);
            partitions->current_partition[key[0]-'A'] --;
            g_free(key);
        }
        if (current_size >= cache->core->size){
            for (i=0; i<n_partitions; i++){
                percent = (double) (partitions->current_partition[i])/cache_size;
                g_array_append_val(partitions->partition_history[i], percent);
            }
        }
        if (get_size(cache) > cache_size)
            fprintf(stderr, "ERROR current size %lu, given size %ld\n", get_size(cache), cache_size);

        read_one_element(reader, cp);
        if (cache->core->type == e_Optimal)
            optimal_params->ts ++;
    }

    
    
    return partitions;
}





