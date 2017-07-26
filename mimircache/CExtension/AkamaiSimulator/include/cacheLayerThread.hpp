//
//  cacheLayerThread.hpp
//  mimircache
//
//  Created by Juncheng Yang on 7/13/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#ifndef cacheLayerThread_hpp
#define cacheLayerThread_hpp


#ifdef __cplusplus
extern "C"
{
#endif
    
#include <stdio.h>
#include <glib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "cache.h"
#include "cacheHeader.h"
#include "reader.h"
#include "logging.h"
    
#ifdef __cplusplus
}
#endif


#include <cstdlib> 
#include <atomic>
#include <algorithm>
#include <functional>
#include <vector>
#include <queue>
#include <thread> 
#include <mutex>
#include <future>
#include <chrono>
#include <climits>
#include <iostream> 
#include <fstream>
#include <iostream>
#include <string> 

#include "constAkamaiSimulator.hpp"
#include "cacheLayer.hpp"





namespace akamaiSimulator {
    
    
    class cp_comparator{
    public:
        bool operator ()(cache_line_t* a, cache_line_t* b) {
            return (a->real_time > b->real_time);
        }
    };
    
    
    
    class cacheLayerThread{
        
        int layer_id;
        int num_of_servers;
        cacheLayer *cache_layer;
        
        unsigned long last_log_output_time;
        std::ofstream log_filestream; 
        std::string log_file_loc;
        
        // to control simultaneous add_request
        std::mutex mtx_add_req;
        
        
        double *cache_server_timestamps;
        
        // used to limit the number of request from a cache server
        // this is not used
        int *cache_server_req_counter;
        
        
        
        
        
        bool all_server_finished;
        std::atomic<double> minimal_timestamp;
        
        // the index of the server that has the minimal ts
        long minimal_ts_server_ind;
        

        std::priority_queue<cache_line_t*,
        std::vector<cache_line_t*>, cp_comparator> *req_pq;

        
        
    public:
        void init();
        
        /** initialize cache layer, layer id is used to obtain
         *  the size of the layer in cache servers */ 
        cacheLayerThread(cacheLayer *cache_layer);
        
        void run(unsigned int log_interval=0);
        
        void add_request(unsigned long cache_server_id, cache_line_t *cp);
        void cache_server_finish(unsigned long cache_server_id);
        
        
        cacheLayer* get_cache_layer();
        double get_minimal_timestamp();
        cacheLayerStat* get_layer_stat();
        
        
        void log_layer_stat(unsigned long layer_time);
        
        ~cacheLayerThread();
    };
    
    
    
}



#endif /* cacheLayerThread_hpp */
