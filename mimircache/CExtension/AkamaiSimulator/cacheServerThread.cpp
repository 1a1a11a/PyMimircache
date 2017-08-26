//
//  cacheServerThread.cpp
//  akamaiSimulator
//
//  Created by Juncheng on 7/11/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//


#include "cacheServerThread.hpp"
#include <iostream>
#include <exception>





namespace akamaiSimulator {
    
    
    
    cacheServerThread::cacheServerThread(cacheServer *cache_server,
                                         cacheLayerThread **cache_layer_threads,
                                         reader_t *reader,
                                         bool use_real_time){
        this->cache_server = cache_server;
        this->cache_server_id = cache_server->get_server_id();
        this->last_log_output_time = 0;
        for (int i=0; i<NUM_CACHE_LAYERS-1; i++)
            this->cache_layer_threads[i] = cache_layer_threads[i];
        this->trace_reader = reader;
        this->use_real_time = use_real_time;
    }
    
    
    
    void cacheServerThread::set_trace_reader(reader_t* reader){
        this->trace_reader = reader;
    }
    
    
    
    void cacheServerThread::run(unsigned int log_interval, const std::string log_folder){
        try{
            
            info("server %lu began running\n", this->cache_server->get_server_id());
            if (this->trace_reader == NULL)
                throw std::runtime_error("trace reader is not set\n");
            if (this->trace_reader->base->type == PLAIN)
                this->use_real_time = false;
            
            if (log_interval != 0){
                mkdir(log_folder.c_str(), 0770);
                this->log_file_loc = std::string(log_folder) + std::string("/server")
                + std::to_string(this->cache_server_id);
                this->log_filestream.open(this->log_file_loc);
            }
            
            
            cache_line_t *cp = new_cacheline();
            
            cp->type = this->trace_reader->base->data_type;
            //  if (cp->type != this->cache_server.)  // verify cache dataType is the same
            
            
            // this is not used, block_unit_size goes with cache
            cp->block_unit_size = (size_t) this->trace_reader->base->block_unit_size;
            cp->disk_sector_size = (size_t) this->trace_reader->base->disk_sector_size;
            
            read_one_element(this->trace_reader, cp);
            while (cp->valid) {
                
//                if (cp->ts > 9600)
//                    verbose("server %lu ts %lu real time %ld\n", this->cache_server_id,
//                            cp->ts, cp->real_time);
                
                this->add_original_request(cp);
                
                if (cp->real_time - this->last_log_output_time > log_interval){
                    this->log_server_stat((unsigned long) cp->real_time);
                    this->last_log_output_time = cp->real_time;
                }
                
                
                read_one_element(this->trace_reader, cp);
            }
            
            for (int i=0; i<NUM_CACHE_LAYERS-1; i++)
                this->cache_layer_threads[i]->cache_server_finish(this->cache_server_id);
            info("cache server %lu finished trace, ts %lu, real time %ld\n",
                 this->cache_server_id, cp->ts, cp->real_time);
            destroy_cacheline(cp);
            
            
            if (log_interval != 0)
                this->log_filestream.close();
            
            
        }catch (std::exception &e) {
            std::cerr<<e.what()<<std::endl;
            print_stack_trace();
            abort();
        }
        
    }
    
    
    
    
    /** this is the logic for cache request flow,
     *  it decides how a request go through different layers of cache,
     *  if cache request flow needs to change, then change this part
     *  current flow logic
     *  when a request comes in, it first touches first layer cache,
     *  if not in first layer, then goes to second layer, meanwhile,
     *  it will add the corresponding obj to both first layer and second layer,
     *  when an obj is evicted, it is independently evicted from layer one or layer two,
     *  it won't affect each other.
     *  \param cp the struct containing request
     *  \return the index of cache layer which it is a hit, if it is -1, then
     *  it goes to origin.
     */
    
    void cacheServerThread::add_original_request(cache_line_t* const cp){
        // THIS PART ONLY WORKS on TWO LAYER CACHE, seems cacheLayerThread has solved this problem?
        // THIS PART CURRENTLY DOES NOT CONSIDER OBJ SIZE
        // OBJ SIZE SHOULD BE TURNED OVER TO CACHE TO HANDLE
        
        int layer_id = 1;
        
        /** keep adding to layer x of cache if it is a miss on x-1 layer of cache,
         *  if layer x has the obj, then the obj is updated
         *  and stop propogating to x+1 layer of cache */
        
        gboolean hit_flag = this->cache_server->add_request(cp, layer_id);
        
        layer_id ++;
        if (layer_id <= NUM_CACHE_LAYERS) {
            if (hit_flag)
                this->cache_layer_threads[layer_id-2]->add_request(this->cache_server_id, cp, false);
            else
                this->cache_layer_threads[layer_id-2]->add_request(this->cache_server_id, cp, true);
        }
        
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    cacheServer* cacheServerThread::get_cache_server(){
        return this->cache_server;
    }
    
    unsigned long cacheServerThread::get_cache_server_id(){
        return this->cache_server_id;
    }
    
    
    void cacheServerThread::log_server_stat(unsigned long server_time){
        this->log_filestream << server_time << "\t" <<
        cacheServer::build_stat_str_short(this->cache_server->server_stat);
    }
    
    
    
    cacheServerThread::~cacheServerThread(){
        
        // delete this->cache_server;
        close_reader(this->trace_reader);
    }
    
}
