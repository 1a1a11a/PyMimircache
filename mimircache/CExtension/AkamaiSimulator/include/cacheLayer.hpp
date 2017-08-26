//
//  cacheLayer.hpp
//  mimircache
//
//  Created by Juncheng Yang on 7/13/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#ifndef cacheLayer_hpp
#define cacheLayer_hpp


#ifdef __cplusplus
extern "C"
{
#endif
    
#include <stdio.h>
#include <glib.h>
#include <string.h>
    
#include "cache.h"
#include "cacheHeader.h"
#include "reader.h"
#include "ketama.h"
#include "logging.h"
    
#ifdef __cplusplus
}
#endif


#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>
#include "constAkamaiSimulator.hpp"
#include "cacheServer.hpp"
#include "consistentHashRing.hpp"


/** currently consistent hash ring does not support change after initialization */



namespace akamaiSimulator {
    
    
    
    class cacheLayerStat{
    public:
        unsigned long layer_id;
        unsigned long num_server;
        
        /** the number of requests from servers in last layer */
        unsigned long *num_req_from_server;
        
        /** the number of requests to severs in this layer, if weight is equal,
         *  then the requests routed to each server should be at the same level */
        unsigned long *num_req_to_server;
        
        unsigned long *per_server_hit_count;
        
        /** the hit count of this layer */
        unsigned long layer_hit_count;
        unsigned long layer_req_count;
        
        double *weight;
        
        
        cacheLayerStat(const unsigned long num_server,
                       const unsigned long layer_id);
        cacheLayerStat(const cacheLayerStat& stat);
        ~cacheLayerStat();
        
        // copy constructor
        cacheLayerStat& operator= (const cacheLayerStat& stat);
        
        void set_weight(double *weight);
        
    };
    
    
    
    class cacheLayer{
        int layer_id;
        double *weight;
        cacheLayer *next_layer;
        
        
        std::vector<cacheServer*> cache_servers;
        consistantHashRing ring;
        
    public:
        cacheLayerStat *layer_stat;
        akamaiStat *akamai_stat; 
        
        /** initialize cache layer, layer id is used to obtain
         *  the size of the layer in cache servers */ 
        cacheLayer(std::vector<cacheServer*> cache_servers,
                   akamaiStat* akamai_stat,
                   const int layer_id,
                   enum hashType hash_type=MD5);

        cacheLayer(cacheServer** cache_servers,
                   const unsigned long num_servers,
                   akamaiStat* akamai_stat, 
                   const int layer_id,
                   enum hashType hash_type=MD5);

        
        /** using the cache server size and boundaries to calculate the 
         *  cache size for each layer and adjust the weight accordingly */
        void cal_weight();

        /** this function is called periodically to adjust the weight of
         *  cache server, because during the period, cache server may have 
         *  adjusted cache size or boundary */
        void rebalance();
        
        void add_request(const unsigned long cache_server_id,
                             cache_line_t * const cp);
        cacheServer& get_server(const int index);
        cacheLayer* get_next_layer(); 
        size_t get_num_server();
        int get_layer_id();
        
        
        void add_server(cacheServer* cache_server);
        void set_next_layer(cacheLayer* next_layer); 
        int get_server_index(cache_line_t* const cp);
        cacheLayerStat* get_layer_stat();
        
        
        ~cacheLayer();
        
        
        
        
        static std::string build_stat_str(cacheLayerStat* stat){
            std::stringstream ss;
            ss.precision(8);
            ss << "CACHE LAYER " << stat->layer_id << " stat, layer req " <<
                stat->layer_req_count << ", hit " << stat->layer_hit_count <<
                ", overall hit rate " << static_cast<double>(stat->layer_hit_count) /
                    (stat->layer_req_count==0 ? 1 : stat->layer_req_count) <<
                "\nper server stat (id, req_from_server, req_to_server, weight, hit count, hit rate)\n";
            
            for (unsigned int i=0; i<stat->num_server; i++){
                if (i != 0 && i%1 == 0)
                    ss << "\n";
                ss << "(" << i << ", " << stat->num_req_from_server[i] << ", " <<
                    stat->num_req_to_server[i] << ", " << stat->weight[i] << ", " <<
                    stat->per_server_hit_count[i] << ", " << 
                    static_cast<double>(stat->per_server_hit_count[i])
                        /(stat->num_req_to_server[i]==0 ? 1 : stat->num_req_to_server[i]) << ")\t";
            }
            ss << "\n";
            return ss.str();
        }
        
        
        static std::string build_stat_str_short(cacheLayerStat* stat){
            std::stringstream ss;
            ss.precision(8);
            ss << "req\t" << stat->layer_req_count
                << "\thit\t" << stat->layer_hit_count << "\toverall hit ratio\t"
                << static_cast<double>(stat->layer_hit_count) /
            (stat->layer_req_count==0 ? 1 : stat->layer_req_count) << "\tserver hit ratio\t";
            
            for (unsigned int i=0; i<stat->num_server; i++){
                ss << static_cast<double>(stat->per_server_hit_count[i])
                /(stat->num_req_to_server[i]==0 ? 1 : stat->num_req_to_server[i]) << " ";
            }
            ss << "\n";
            return ss.str();
        }

        
        static void print_stat(cacheLayerStat* stat){
            std::cout << cacheLayer::build_stat_str(stat);
      }

    };
    
    
    
}



#endif /* cacheLayer_hpp */
