//
//  cacheServer.cpp
//  akamaiSimulator
//
//  Created by Juncheng on 7/11/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//


#include "cacheServer.hpp"
#include <iostream>

namespace akamaiSimulator {
    
    
    cacheServerStat::cacheServerStat(const unsigned long server_id,
                                     const unsigned long cache_size,
                                     const double* const boundaries) {
        
        
        unsigned long i;
        
        this->cache_server_id = server_id;
        this->cache_size = cache_size;
        this->num_req = 0;
        this->num_hit = 0;
        
        for (i=0; i<NUM_CACHE_LAYERS; i++){
            this->layer_size[i] = (unsigned long) (boundaries[i] * cache_size);
            this->num_req_per_layer[i] = 0;
            this->num_hit_per_layer[i] = 0;
        }
    }
    
    
    // copy constructor
    cacheServerStat::cacheServerStat(const cacheServerStat& stat){
        *this = stat;
    }
    
    
    
    cacheServerStat::~cacheServerStat(){
        ;
    }
    
    
    cacheServerStat& cacheServerStat::operator= (const cacheServerStat& stat){
        if (this == &stat)
            return *this;
        this->cache_server_id = stat.cache_server_id;
        this->cache_size = stat.cache_size;
        this->num_req = stat.num_req;
        this->num_hit = stat.num_hit;
        
        for (unsigned int i=0; i<NUM_CACHE_LAYERS; i++){
            this->layer_size[i] = stat.layer_size[i];
            this->num_req_per_layer[i] = stat.num_req_per_layer[i];
            this->num_hit_per_layer[i] = stat.num_hit_per_layer[i];
        }
        
        return *this;
    }
    
    void cacheServerStat::set_new_boundaries(double *boundaries){
        for (unsigned int i=0; i<NUM_CACHE_LAYERS; i++){
            this->layer_size[i] = (unsigned long) (boundaries[i] * cache_size);
        }
    }

    
    
    
    
    
    
    /******************* constructor *******************/
    
    cacheServer::cacheServer(const unsigned long server_id,
                             akamaiStat* const akamai_stat,
                             const gint64 size,
                             const double* const boundaries,
                             const cache_type cache_alg,
                             const char data_type,
                             const int block_size,
                             void *params,
                             const std::string server_name){
        
        
        int i;

        this->cache_server_id = server_id;
        this->cache_server_name = server_name;
        this->cache_size = size;
        this->akamai_stat = akamai_stat;
        this->server_stat = new cacheServerStat(server_id, size, boundaries);
        
        for (i=0; i<NUM_CACHE_LAYERS; i++){
            this->boundaries[i] = boundaries[i];
        }

        
        
        switch (cache_alg) {
            case e_LRU:
                for (i=0; i<NUM_CACHE_LAYERS; i++)
                    this->caches[i] = LRU_init(this->server_stat->layer_size[i],
                                               data_type, block_size, params);
                
                break;
                
            default:
                error_msg("given algorithm is not supported: %d\n", cache_alg);
                break;
        }
    }
    
    
    /** use an array of caches to initialize the cache server,
     *  the first cache in the array is layer 1 cache, second is layer 2, etc.
     *  ATTENTION: the life of caches is handed over to cache server,
     *  thus resource de-allocation will be done by cacheServer */
    cacheServer::cacheServer(const unsigned long id,
                             const cache_t ** const caches,
                             akamaiStat* const akamai_stat,
                             const std::string server_name){

        
        this->cache_server_id = id;
        this->cache_server_name = server_name;
        this->set_caches(caches);
        this->akamai_stat = akamai_stat;
        
        
        this->server_stat = new cacheServerStat(id, this->cache_size, this->boundaries);
    }
    
    
    
    
    /******************* setter *******************/
    gboolean cacheServer::set_L1cache(const cache_t * const cache){
        this->caches[0] = (cache_t*) cache;
        if ((unsigned long) cache->core->size != this->server_stat->layer_size[0]){
            error_msg("new L1 cache size is different, %lu %ld\n",
                      this->server_stat->layer_size[0], cache->core->size);
            abort();
        }
        return TRUE;
    }
    
    
    gboolean cacheServer::set_L2cache(const cache_t * const cache){
        this->caches[1] = (cache_t*) cache;
        if ((unsigned long) cache->core->size != this->server_stat->layer_size[1]){
            error_msg("new L2 cache size is different, %lu %ld\n",
                      this->server_stat->layer_size[0], cache->core->size);
            abort();
        }

        return TRUE;
    }
    
    gboolean cacheServer::set_Lncache(int n, const cache_t* const cache){
        if (n > NUM_CACHE_LAYERS){
            error_msg("specified %d th cache, but only %d layer of cache\n", n, NUM_CACHE_LAYERS);
            abort();
        }
        this->caches[n-1] = (cache_t*) cache;
        if ((unsigned long) cache->core->size != this->server_stat->layer_size[n-1]){
            error_msg("new Ln cache size is different, %lu %ld\n",
                      this->server_stat->layer_size[0], cache->core->size);
            abort();
        }

        return TRUE;
    }

    
    gboolean cacheServer::set_caches(const cache_t ** const caches){
        this->cache_size = 0;
        for (int i=0; i<NUM_CACHE_LAYERS; i++){
            this->caches[i] = (cache_t*) (caches[i]);
            this->server_stat->layer_size[i] = (unsigned long) (this->caches[i]->core->size);
            this->cache_size += this->server_stat->layer_size[i];
        }
        for (int i=0; i<NUM_CACHE_LAYERS; i++)
            this->boundaries[i] = this->server_stat->layer_size[i] / (double) (this->cache_size);
        return TRUE;
    }
    
    
    gboolean cacheServer::set_boundary(const double* const boundaries){
        for (int i=0; i<NUM_CACHE_LAYERS; i++)
            this->boundaries[i] = boundaries[i];
        this->adjust_caches();
        
        info("reset boundary of cache server %s (id %lu), new size %ld, size of each layer ",
             this->cache_server_name.c_str(), this->cache_server_id, this->cache_size);
        for (int i=0; i<NUM_CACHE_LAYERS; i++)
            printf("%ld, ", this->caches[i]->core->size);
        printf("\n");
        
        return TRUE;
    }
    
    gboolean cacheServer::set_size(const gint64 size){
        /* if shrinking size of cache,
         * the actual shrinking is done when next request comes in */
        
        this->cache_size = size;
        this->adjust_caches();
        
        info("reset size of cache server %s (id %lu), new size %ld, size of each layer ",
             this->cache_server_name.c_str(), this->cache_server_id, this->cache_size);
        for (int i=0; i<NUM_CACHE_LAYERS; i++)
            printf("%ld, ", this->caches[i]->core->size);
        printf("\n");
        
        return TRUE;
    }
    
    
    
    /** this function is used internally for adjust the real cache size of each layer **/ 
    gboolean cacheServer::adjust_caches(){
        int i;
        for (i=0; i<NUM_CACHE_LAYERS; i++){
            this->server_stat->layer_size[i] = (gint64) (this->cache_size * this->boundaries[i]);
            this->caches[i]->core->size = (long) (this->server_stat->layer_size[i]);
        }
        return TRUE;
    }
    
    
    
    
    /** add a request to certain layer 
     *  \param cp          cache request struct
     *  \param layer_id    the index of cache layer which begins from 1
     *  \return            whether it is a hit or miss on this layer 
     */
    gboolean cacheServer::add_request(const cache_line_t *const cp,
                                      const unsigned long layer_id){
        /* log server stat */
        this->server_stat->num_req_per_layer[layer_id-1] ++;
        this->server_stat->num_req ++;
        
        /* also write stat into akamaiStat */
        this->akamai_stat->req[layer_id-1]++;

        gboolean hit = this->caches[0]->core->add_element(this->caches[layer_id-1],
                                                   (cache_line_t*) cp);
        
        if (hit){
            this->server_stat->num_hit ++;
            this->server_stat->num_hit_per_layer[layer_id-1] ++;
            this->akamai_stat->hit[layer_id-1]++; 
        }
        return hit; 
    }
    
    

    
    
    
    /******************* getter *******************/
    double* cacheServer::get_boundary(){
        return this->boundaries;
    }
    
    
    gint64 cacheServer::get_cache_size(){
        return this->cache_size;
    }
    
    unsigned long* cacheServer::get_layer_size(){
        return this->server_stat->layer_size;
    }
    
    
    unsigned long cacheServer::get_server_id(){
        return this->cache_server_id;
    }
        
    
    gboolean cacheServer::verify(){
        error_msg("verify is not implemented"); 
        return FALSE;
    }
    
    
    
    cacheServer::~cacheServer(){
        for (int i=0; i<NUM_CACHE_LAYERS; i++)
            this->caches[i]->core->destroy(this->caches[i]);
        delete this->server_stat;
    }
    
    
}
