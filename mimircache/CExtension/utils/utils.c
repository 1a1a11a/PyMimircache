//
//  utils.c
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "utils.h"
#include "const.h"






int set_thread_affinity(pthread_t tid){
    static int last_core_id = -1;
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);

    last_core_id ++;
    last_core_id %= num_cores;
    DEBUG_MSG("assign thread affinity %d/%d\n", last_core_id, num_cores);
    
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(last_core_id, &cpuset);
    
    int rc = pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        WARNING("Error calling pthread_setaffinity_np: %d\n", rc);
    }
}
