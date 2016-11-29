//
//  partition.h
//  mimircache
//
//  Created by Juncheng on 11/19/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef partition_h
#define partition_h


#include <stdio.h>
#include <stdlib.h>
#include <glib.h>
#include <string.h>
#include <stdint.h>
#include "glib_related.h"
#include "cache.h"
#include "reader.h"
#include "const.h"
#include "FIFO.h"
#include "Optimal.h"
#include "LRU_K.h"
#include "LRU.h"
#include "python_wrapper.h"
#include "LRU_dataAware.h"
#include "AMP.h"


typedef struct{
    uint8_t     n_partitions;
    uint64_t    cache_size;
    GArray**    partition_history;
    uint64_t*   current_partition;
    uint64_t    jump_over_count;            // the first m requests are not in the partition because the cache was not full at
                                            // that time
}partition_t;




partition_t* init_partition_t(uint8_t n_partitions, uint64_t cache_size);
void free_partition_t(partition_t *partition);


partition_t* get_partition(READER* reader, struct cache* cache, uint8_t n_partitions);
return_res** profiler_partition(READER* reader_in, struct_cache* cache_in, int num_of_threads_in, int bin_size_in);



#endif /* partition_h */
