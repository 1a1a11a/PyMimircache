//
//  utils.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef UTILS_h
#define UTILS_h

#include "const.h" 

#include <stdio.h>
#include <math.h> 
#include <glib.h> 
#include <pthread.h>
#include <unistd.h>

#include <sched.h>



int set_thread_affinity(pthread_t tid);





#endif /* UTILS_h */
