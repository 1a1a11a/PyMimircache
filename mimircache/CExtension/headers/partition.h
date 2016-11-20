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
#include "reader.h"
#include "glib_related.h"
#include "cache.h"
#include "const.h"


typedef struct{
    uint8_t     n_partitions;
    uint64_t    cache_size;
    GArray**    partition_history;
    uint64_t*   current_partition;
}partition_t;




partition_t* init_partition_t(uint8_t n_partitions, uint64_t cache_size);
void free_partition_t(partition_t *partition);


partition_t* get_partition(READER* reader, struct cache* cache, uint8_t n_partitions);
    


#endif /* partition_h */
