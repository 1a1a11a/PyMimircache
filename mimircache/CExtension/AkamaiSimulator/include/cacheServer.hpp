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
#include "splay.h" 
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
#include <stdexcept>
#include <mutex>
#include <atomic>

#include "constAkamaiSimulator.hpp"
#include "consistentHashRing.hpp"
#include "akamaiStat.hpp"



#define SERVER_NAME_LEN_MAX 22      // this is limited by libketama


namespace akamaiSimulator {
    
    class akamaiStat;
    
    
    class cacheServerStat{
    public:
        unsigned long cache_server_id;
        unsigned long cache_size;
        unsigned long *layer_size;      // points the array inside cacheServer

        
        std::atomic_ulong num_req;
        std::atomic_ulong num_hit;
        std::atomic_ulong num_req_per_layer[NUM_CACHE_LAYERS];
        std::atomic_ulong num_hit_per_layer[NUM_CACHE_LAYERS];
        
        
        
//        unsigned long num_req;
//        unsigned long num_hit;
//        unsigned long num_req_per_layer[NUM_CACHE_LAYERS];
//        unsigned long num_hit_per_layer[NUM_CACHE_LAYERS];

        
        
        cacheServerStat(const unsigned long server_id,
                        const unsigned long cache_size,
                        unsigned long* layer_size);
        
        cacheServerStat(const cacheServerStat& stat);
        ~cacheServerStat();
        
        // copy constructor
        cacheServerStat& operator= (const cacheServerStat& stat);
        
//        void set_new_boundaries(double *boundaries);

    };
    
    
    
    
    
    
    class cacheProfiler{
        
        /** max_cache_size is the max possible layer cache size
         *  this is used for array allocation during initialization 
         *  currently using server cache size is good enough */
        unsigned long long max_cache_size;
        unsigned char data_type;
        unsigned long L1_latency;
        unsigned long L2_latency;
        unsigned long Lo_latency;

        unsigned long *layer_size;
        
        /* ts is also the request count */
        unsigned long ts[NUM_CACHE_LAYERS];
        /* ts when last time boundary shift suggestion is given */
        unsigned long last_adjust_ts;
        
        GHashTable *hashtable[NUM_CACHE_LAYERS];
        sTree *splay_tree[NUM_CACHE_LAYERS];
        
        /** this uses a lot of mem, but we don't care here
         *  the last element of the array is cold miss count 
         *  the second to last is the count of req with 
         *  rd > max_cache_size */
        unsigned long long *rd_count_array[NUM_CACHE_LAYERS];
        unsigned long long rd_count_array_size[NUM_CACHE_LAYERS];
        
        unsigned long adjust_interval;
        unsigned long space_optimize_interval;
        
        std::mutex mtx; 
        
        
        
        /* the logic how to adjust boundary */
        int __how_to_adjust();
        
        /** if we don't optimize profiler space usage,
         *  the maximal reuse distance can be infinity 
         *  which is huge in memory usagee */ 
        void __optimize_profiler_space(unsigned long i);

        
        
    public:
        
        cacheProfiler(const unsigned long max_cache_size,
                      const unsigned char data_type,
                      unsigned long *layer_size,
                      const unsigned long adjust_interval=20000,
                      const unsigned long space_optimize_interval=1000000,
                      const unsigned long L1_latency=10,
                      const unsigned long L2_latency=20,
                      const unsigned long Lo_latency=50);

        void __init(const unsigned long max_cache_size,
                    const unsigned char data_type,
                    unsigned long *layer_size,
                    const unsigned long adjust_interval,
                    const unsigned long space_optimize_interval,
                    const unsigned long L1_latency,
                    const unsigned long L2_latency,
                    const unsigned long Lo_latency);

        void set_new_boundaries(unsigned long* layer_size);
        
        /** add request to profiler and get back boundary adjustment decision
         *  if return 1,  L1 increases, L2 decreases,
         *  if return -1, L1 decreases, L2 increases 
         *  if return 0,  no change */
        int add_request(const cache_line_t* const cp,
                        const unsigned long layer_id);
        
        
        
//        gboolean __check_hashtable_entry(gpointer key, gpointer value, gpointer data);

        
        /* clear all data, begin fresh */
        void clear();
        
        ~cacheProfiler();
        
        
        /*-----------------------------------------------------------------------------
         *
         * process_one_element --
         *      this function is used for computing reuse distance for each request
         *      it maintains a hashmap and a splay tree,
         *      time complexity is O(log(N)), N is the number of unique elements
         *
         *
         * Input:
         *      cp           the cache line struct contains input data (request label)
         *      splay_tree   the splay tree struct
         *      hash_table   hashtable for remember last request timestamp (virtual)
         *      ts           current timestamp
         *      reuse_dist   the calculated reuse distance
         *
         * Return:
         *      splay tree struct pointer, because splay tree is modified every time,
         *      so it is essential to update the splay tree
         *
         *-----------------------------------------------------------------------------
         */
        static inline sTree* process_one_element(const cache_line* const cp,
                                                 sTree* splay_tree,
                                                 GHashTable* hash_table,
                                                 unsigned long ts,
                                                 long long* reuse_dist){
            gpointer gp;
            
            gp = g_hash_table_lookup(hash_table, cp->item_p);
            
            sTree* newtree;
            if (gp == NULL){
                // first time access
                newtree = insert(ts, splay_tree);
                gint64 *value = g_new(gint64, 1);
                if (value == NULL){
                    error_msg("not enough memory\n");
                    abort();
                }
                *value = ts;
                if (cp->type == 'c')
                    g_hash_table_insert(hash_table, g_strdup((gchar*)(cp->item_p)), (gpointer)value);
                
                else if (cp->type == 'l'){
                    gint64* key = g_new(gint64, 1);
                    *key = *(guint64*)(cp->item_p);
                    g_hash_table_insert(hash_table, (gpointer)(key), (gpointer)value);
                }
                else{
                    error_msg("unknown cache line content type: %c\n", cp->type);
                    abort();
                }
                *reuse_dist = -1;
            }
            else{
                // not first time access
                guint64 old_ts = *(guint64*)gp;
                newtree = splay(old_ts, splay_tree);
                *reuse_dist = node_value(newtree->right);
                *(guint64*)gp = ts;
                
                newtree = splay_delete(old_ts, newtree);
                newtree = insert(ts, newtree);
                
            }
            return newtree;
        }
    };
    
    
    
    
    
    class cacheServer {
        std::string cache_server_name;
        unsigned long cache_server_id;
        
        /* related to dynamic boundary change */
        bool dynamic_boundary_flag;
        cacheProfiler *cache_profiler;
        
        
        unsigned long cache_size;
//        double boundaries[NUM_CACHE_LAYERS];                      // ATTENTION change boundary on cacheServer won't affect weight in consistentHashRing
        unsigned long layer_size[NUM_CACHE_LAYERS];
        cache_t *caches[NUM_CACHE_LAYERS];
        
        /** all cache size related lock,
         *  current usage: prevent dynamic boundary size chage 
         *  from affecting adding requests */ 
        std::mutex mtx_layer_size;
        
        bool adjust_caches();
        
    public:
        cacheServerStat *server_stat;
        akamaiStat *akamai_stat;

        
        cacheServer(const unsigned long id,
                    akamaiStat* const akamai_stat,
                    bool dynamic_boundary_flag,
                    const unsigned long size,
                    const double* const boundaries,
                    const cache_type cache_alg,
                    const unsigned char data_type,
                    const int block_size=0,
                    void *params=NULL,
                    const std::string server_name="default server");
        
        cacheServer(const unsigned long id,
                    const cache_t ** const caches,
                    bool dynamic_boundary_flag,
                    akamaiStat* const akamai_stat,
                    const std::string server_name="default server");
        
        
        bool set_L1cache(const cache_t * const cache);
        bool set_L2cache(const cache_t * const cache);
        bool set_Lncache(int n, const cache_t* const cache);
        bool set_caches(const cache_t ** const caches);
        
        bool set_size(const unsigned long size);
        
        
        bool add_request(const cache_line_t* const cp,
                         const unsigned long layer_id);
        
        
        
        
        
        unsigned long get_cache_size();
        unsigned long* get_layer_size();
        unsigned long get_server_id();

        
        
        bool verify();
        
        ~cacheServer();

        
        static std::string build_stat_str(cacheServerStat* stat){
            std::stringstream ss;
            ss.precision(8);
            ss << "CACHE SERVER " << stat->cache_server_id << " stat, size " <<
                stat->cache_size << ", " << stat->num_req.load() << " req, " <<
                stat->num_hit.load() << " hit, " << "overall hit ratio " <<
                static_cast<double>(stat->num_hit)/(stat->num_req.load()==0?1:stat->num_req.load()) <<
                "\nper server stat " <<
                "(layerID, layer size, num of req, num of hit, hit ratio)\n";
            
            for (unsigned int i=0; i<NUM_CACHE_LAYERS; i++){
                if (i != 0 && i%1 == 0)
                    ss << "\n";
                ss << "(" << i+1 << ", " << stat->layer_size[i] << ", " <<
                    stat->num_req_per_layer[i].load() << ", " <<
                    stat->num_hit_per_layer[i].load() << ", " <<
                    static_cast<double>(stat->num_hit_per_layer[i].load())/
                    (stat->num_req_per_layer[i].load()==0 ? 1 : stat->num_req_per_layer[i].load()) << ")";
            }
            ss << "\n"; 
            return ss.str();
        }
        
        static std::string build_stat_str_short(cacheServerStat* stat){
            std::stringstream ss;
            ss.precision(8);
            ss << "overall\t" << stat->num_req.load() << "\t" << stat->num_hit.load() << "\t" <<
            static_cast<double>(stat->num_hit.load())/(stat->num_req.load()==0?1:stat->num_req.load());
            
            for (unsigned int i=0; i<NUM_CACHE_LAYERS; i++){
                ss << "\tlayer" << i+1 << "\t" << stat->layer_size[i] << "\t" <<
                stat->num_req_per_layer[i].load() << "\t" <<
                stat->num_hit_per_layer[i].load() << "\t" <<
                static_cast<double>(stat->num_hit_per_layer[i].load())/
                (stat->num_req_per_layer[i].load()==0 ? 1 : stat->num_req_per_layer[i].load()); 
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

