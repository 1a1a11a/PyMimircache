//
//  cacheServerThread.cpp
//  akamaiSimulator
//
//  Created by Juncheng on 7/11/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//


#ifndef cacheServerThread_HPP
#define cacheServerThread_HPP



#include <thread>
#include <future>
#include <string>
#include <iostream> 
#include <fstream> 
#include "cacheServer.hpp"
#include "cacheLayerThread.hpp"
#include "constAkamaiSimulator.hpp" 



#ifdef __cplusplus
extern "C"
{
#endif
    
#include "cacheHeader.h"
#include "reader.h"
#include "logging.h"
#include <sys/stat.h>
#include <sys/types.h>


#ifdef __cplusplus
}
#endif




namespace akamaiSimulator {
    
    
    class cacheServerThread {
        cacheServer *cache_server;
        unsigned long cache_server_id;
        
        unsigned long last_log_output_time;
        std::string log_file_loc;
        std::ofstream log_filestream;
        
        cacheLayerThread *cache_layer_threads[NUM_CACHE_LAYERS];
        
        reader_t * trace_reader;
        
        bool use_real_time;
        
        
        
    public:

        cacheServerThread(cacheServer *cache_server,
                          cacheLayerThread **cache_layer_threads,
                          reader_t *reader, bool use_real_time=true);
        
        
        
        void set_trace_reader(reader_t* reader);

        
        void run(unsigned int log_interval, const std::string log_folder);
        
        
        
        /** the difference of this function and add_request is that
         *  this function allows the complete cache request flow,
         *  in other words, if it is a miss in add_request, it won't go further,
         *  but in add_original_request, it is checked in first layer cache,
         *  if it is a miss, then it will goes to next layer of cache, this process
         *  is repeated until it is a cache hit */
        void add_original_request(cache_line_t* const cp);

        
        
        
        cacheServer* get_cache_server();
        unsigned long get_cache_server_id();
        void log_server_stat(unsigned long server_time);

        
        
        
        ~cacheServerThread();
    };
    
    
}


#endif /* cacheServerThread_HPP */ 

