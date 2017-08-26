//
//  constAkamaiSimulator.cpp
//  akamaiSimulator
//
//  Created by Juncheng on 7/11/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//


#ifndef constAkamaiSimulator_HPP
#define constAkamaiSimulator_HPP




#define NUM_CACHE_LAYERS 2          // currently only support two layers, limited in cacheServerThread::add_original_req 

#define MAX_SERVER_REQ_QUEUE_SIZE 1024
//#define MAX_SERVER_REQ_QUEUE_SIZE 1024*1024
//#define MAX_SERVER_REQ_QUEUE_SIZE 1024*1024*1024


//#define THREAD_AFFINITY



#define TIME_SYNCHRONIZATION 
#define SYNCHRONIZATION_TIME_DIFF 1 
//#define SYNCHRONIZATION_TIME_DIFF 120


#define L2_COPY 1 


#undef DEBUG_DEADLOCK 



#endif /* constAkamaiSimulator_HPP */ 

