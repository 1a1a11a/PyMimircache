//
//  cacheServer.cpp
//  akamaiSimulator
//
//  Created by Juncheng on 7/11/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//


#ifndef cacheServer_HPP
#define cacheServer_HPP


#ifdef __cplusplus
extern "C"
{
#endif
    
#include <glib.h>
    
#include "cache.h"
#include "cacheHeader.h"
#include "ketama.h"
#include "logging.h"
    
#ifdef __cplusplus
}
#endif

#include <string>
#include <iostream>
#include <ostream> 
#include <sstream>
#include <fstream> 

#include "constAkamaiSimulator.hpp"
#include "consistentHashRing.hpp"
#include "akamaiStat.hpp"



#define SERVER_NAME_LEN_MAX 22      // this is limited by libketama


namespace akamaiSimulator {
    
    
    class cacheServerStat{
    public:
        unsigned long cache_server_id;
        unsigned long cache_size;
        unsigned long layer_size[NUM_CACHE_LAYERS];

        unsigned long num_req;
        unsigned long num_hit;
        unsigned long num_req_per_layer[NUM_CACHE_LAYERS];
        unsigned long num_hit_per_layer[NUM_CACHE_LAYERS];

        
        
        cacheServerStat(const unsigned long server_id,
                        const unsigned long cache_size,
                        const double* const boundary);
        
        cacheServerStat(const cacheServerStat& stat);
        ~cacheServerStat();
        
        // copy constructor
        cacheServerStat& operator= (const cacheServerStat& stat);
        
        void set_new_boundaries(double *boundaries);

    };
    
    class cacheServer {
        std::string cache_server_name;
        unsigned long cache_server_id;
        
        
        gint64 cache_size;
        double boundaries[NUM_CACHE_LAYERS];                      // ATTENTION change boundary on cacheServer won't affect weight in consistentHashRing
        cache_t *caches[NUM_CACHE_LAYERS];
        
        
        gboolean adjust_caches();
        
    public:
        cacheServerStat *server_stat;
        akamaiStat *akamai_stat;

        
        cacheServer(const unsigned long id,
                    const gint64 size,
                    const double* const boundaries,
                    const cache_type cache_alg,
                    const char data_type,
                    const int block_size=0,
                    void *params=NULL,
                    akamaiStat* const akamai_stat,
                    const std::string server_name="default server");
        
        cacheServer(const unsigned long id,
                    const cache_t ** const caches,
                    akamaiStat* const akamai_stat,
                    const std::string server_name="default server");
        
        
        gboolean set_L1cache(const cache_t * const cache);
        gboolean set_L2cache(const cache_t * const cache);
        gboolean set_Lncache(int n, const cache_t* const cache);
        gboolean set_caches(const cache_t ** const caches);
        
        gboolean set_boundary(const double* const boundaries);
        gboolean set_size(const gint64 size);
        
        
        gboolean add_request(const cache_line_t* const cp,
                             const unsigned long layer_id);
        
        
        
        
        
        double *get_boundary();
        gint64 get_cache_size();
        unsigned long get_server_id();
        unsigned long* get_layer_size();

        
        
        gboolean verify();
        
        ~cacheServer();

        
        static std::string build_stat_str(cacheServerStat* stat){
            std::stringstream ss;
            ss.precision(8);
            ss << "CACHE SERVER " << stat->cache_server_id << " stat, size " <<
                stat->cache_size << ", " << stat->num_req << " req, " <<
                stat->num_hit << " hit, " << "overall hit ratio " <<
                static_cast<double>(stat->num_hit)/(stat->num_req==0?1:stat->num_req) <<
                "\nper server stat " <<
                "(layerID, layer size, num of req, num of hit, hit ratio)\n";
            
            for (unsigned int i=0; i<NUM_CACHE_LAYERS; i++){
                if (i != 0 && i%1 == 0)
                    ss << "\n";
                ss << "(" << i+1 << ", " << stat->layer_size[i] << ", " <<
                    stat->num_req_per_layer[i] << ", " <<
                    stat->num_hit_per_layer[i] << ", " <<
                    static_cast<double>(stat->num_hit_per_layer[i])/
                    (stat->num_req_per_layer[i]==0 ? 1 : stat->num_req_per_layer[i]) << ")";
            }
            ss << "\n"; 
            return ss.str();
        }
        
        static std::string build_stat_str_short(cacheServerStat* stat){
            std::stringstream ss;
            ss.precision(8);
            ss << "overall\t" << stat->num_req << "\t" << stat->num_hit << "\t" <<
            static_cast<double>(stat->num_hit)/(stat->num_req==0?1:stat->num_req);
            
            for (unsigned int i=0; i<NUM_CACHE_LAYERS; i++){
                ss << "\tlayer" << i+1 << "\t" << stat->layer_size[i] << "\t" <<
                stat->num_req_per_layer[i] << "\t" <<
                stat->num_hit_per_layer[i] << "\t" <<
                static_cast<double>(stat->num_hit_per_layer[i])/
                (stat->num_req_per_layer[i]==0 ? 1 : stat->num_req_per_layer[i]);
            }
            ss << "\n";
            return ss.str();
        }

        
        static void print_stat(cacheServerStat* stat){
            std::cout << cacheServer::build_stat_str(stat);
        }
        

    };
    
    
}


#endif /* cacheServer_HPP */ 

