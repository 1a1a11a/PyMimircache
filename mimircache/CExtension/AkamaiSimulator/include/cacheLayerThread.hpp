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
#include <deque>
#include <queue>
#include <thread> 
#include <mutex>
#include <condition_variable>
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
    
    
    class cacheServerReqQueue{
        /** the queue for incoming cache server requests 
         *  this is used for synchronization on all the servers in the same layer */
        
        std::mutex mtx;
        std::condition_variable condt;
        std::deque<cache_line_t*> dq;
        unsigned long max_queue_size;
        std::atomic_ulong min_ts;
        std::atomic_ulong min_ts_server_id;
        unsigned long server_id;

        
        bool _can_add_req(cache_line_t* cp);
    
    public:
        unsigned long current_ts; 
        
        cacheServerReqQueue(unsigned long max_queue_size,
                            unsigned long server_id);
        

        
        /* this can be blocking */
        void add_request(cache_line_t* cp, bool real_add);
        
        cache_line_t* get_request();
        cache_line_t* get_request(double max_ts);
        size_t get_size();
        unsigned long get_min_ts();
        unsigned long get_min_ts_server_id();
        
        void synchronize_time(double t, unsigned long min_ts_ind);
        void queue_size_change();
    };
    
    
    
    
    
    class cacheLayerThread{
        
        int layer_id;
        unsigned long num_of_servers;
        cacheLayer *cache_layer;
        
        unsigned long last_log_output_time;
        std::ofstream log_filestream; 
        std::string log_file_loc;
        
        // to control simultaneous add_request
        std::mutex mtx_add_req;
        std::mutex mtx_sync_time; 
        
        
//        volatile double *cache_server_timestamps;
        std::atomic<double> *cache_server_timestamps;
        
        
        
        
        
        
        bool all_server_finished;
//        double minimal_timestamp;
        std::atomic<double> minimal_timestamp;
        
        // the index of the server that has the minimal ts
//        long minimal_ts_server_ind;
        std::atomic_long minimal_ts_server_ind;
        

        
        cacheServerReqQueue **cache_server_req_queue;
        
        
//        std::priority_queue<cache_line_t*,
//                std::vector<cache_line_t*>, cp_comparator> *req_pq;

        
        
    public:
        void init();
        
        /** initialize cache layer, layer id is used to obtain
         *  the size of the layer in cache servers */ 
        cacheLayerThread(cacheLayer *cache_layer);
        
        void run(unsigned int log_interval, const std::string log_folder);
        
        void add_request(unsigned long cache_server_id,
                         cache_line_t *cp, bool real_add);

        /** update the server ts in the layer, this is needed because sometimes
         *  a server may not hit higher layer for a long time */
//        void update_layer_ts(unsigned long cache_server_id, cache_line_t *cp);
        
        void cache_server_finish(unsigned long cache_server_id);
        
        
        cacheLayer* get_cache_layer();
        double get_minimal_timestamp();
        cacheLayerStat* get_layer_stat();
        
        
        void log_layer_stat(unsigned long layer_time);
        
        ~cacheLayerThread();
    };
    
    
    
}



#endif /* cacheLayerThread_hpp */
