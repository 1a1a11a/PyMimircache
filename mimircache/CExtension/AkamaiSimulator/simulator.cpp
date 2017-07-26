//
//  simulator.cpp
//  akamaiSimulator
//
//  Created by Juncheng on 7/11/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//




#include "simulator.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <future>


#define L1_LATENCY 10 
#define L2_LATENCY 20 
#define ORIGIN_LATENCY 50



namespace akamaiSimulator {
    
    
    void log_akamai_stat(akamaiStat *akamai_stat, bool *stop_flag){
        std::ofstream ofs(std::string(LOG_FOLDER)+"akamai_stat", std::ios::out);
        
        while (! *stop_flag){
            std::this_thread::sleep_for(std::chrono::seconds(2));

            ofs << akamai_stat->get_avg_latency() << "\t" <<
            akamai_stat->get_avg_latency_L1() << "\t" <<
            akamai_stat->get_avg_latency_L2() << "\t" <<
            akamai_stat->get_traffic_to_origin() << "\t" <<
            akamai_stat->get_traffic_between_first_second_layer() << "\n";
        }
    }
    
    
    
    
    void akamai_run(std::vector<std::string> traces,
                    double *boundaries,
                    unsigned long* cache_sizes,
                    int akamai_data_type){
        
        unsigned int i;
        unsigned long num_servers = traces.size();
        
        reader_t *readers[num_servers];

        csvReader_init_params* reader_init_params;
        if (akamai_data_type == 3)
            reader_init_params = AKAMAI3_CSV_PARAM_INIT;
        else
            reader_init_params = AKAMAI_CSV_PARAM_INIT;

        
        /* threads for cacheServerThread and cacheLayerThread */
        std::thread *t;
        std::vector<std::thread*> threads;
        
        bool thread_stop_flag = false;
        akamaiSimulator::akamaiStat* akamai_stat =
            new akamaiSimulator::akamaiStat(L1_LATENCY, L2_LATENCY, ORIGIN_LATENCY);
        std::thread log_akamai_stat_thread(log_akamai_stat,
                                           akamai_stat,
                                           &thread_stop_flag);
        
        
        /* initialize cacheServers and readers */
        akamaiSimulator::cacheServer* cache_servers[num_servers];
        akamaiSimulator::cacheServerThread *cache_server_threads[num_servers];
        for (i=0; i<num_servers; i++){
            cache_servers[i] = new akamaiSimulator::cacheServer(i, cache_sizes[i],
                                                                boundaries, e_LRU,
                                                                'c', 0, akamai_stat, NULL);
            readers[i] = setup_reader(traces.at(i).c_str(), CSV, 'c', 0, 0,
                                      (void*) reader_init_params);
        }

        /* initialize cacheLayer and cacheLayerThread */
        akamaiSimulator::cacheLayer* cache_layers[NUM_CACHE_LAYERS-1];
        akamaiSimulator::cacheLayerThread *cache_layer_threads[NUM_CACHE_LAYERS-1];
        for (i=0; i<NUM_CACHE_LAYERS-1; i++){
            cache_layers[i] = new akamaiSimulator::cacheLayer(cache_servers, num_servers, 2 + i);
            cache_layer_threads[i] = new akamaiSimulator::cacheLayerThread(cache_layers[i]);
        }
            
        /* build threads for cacheLayerThread */
        for (i=0; i<NUM_CACHE_LAYERS-1; i++){
            t = new std::thread(&akamaiSimulator::cacheLayerThread::run, cache_layer_threads[i], 120);
            threads.push_back(t);
        }
        
        /* build threads for cacheServerThread */
        for (i=0; i<num_servers; i++){
            cache_server_threads[i] = new akamaiSimulator::cacheServerThread(cache_servers[i],
                                                                             cache_layer_threads,
                                                                             readers[i], true);
            t = new std::thread(&akamaiSimulator::cacheServerThread::run, cache_server_threads[i], 120);
            threads.push_back(t);
        }
        
        
        if (/* DISABLES CODE */ (false)){
            akamaiSimulator::cacheLayerStat *stat;
            for (i=0; i<3; i++){
                stat = cache_layer_threads[0]->get_layer_stat();
                akamaiSimulator::cacheLayer::print_stat(stat);
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
        
        /* wait for cache server and cache layer to finish computation */
        for (auto it=threads.begin(); it<threads.end(); it++){
            (**it).join();
            delete *it;
        }
        
        /* stop akamai_stat_log thread */
        thread_stop_flag = true;
        log_akamai_stat_thread.join();
        
        
        /* print stat of cache server and free cache server */
        for (auto cache_server: cache_servers){
            akamaiSimulator::cacheServer::print_stat(cache_server->server_stat);
            delete cache_server;
        }
        std::cout<<"\n"; 
        
        /* print stat of cache layer and free cache layer */
        for (auto cache_layer: cache_layers){
            akamaiSimulator::cacheLayer::print_stat(cache_layer->layer_stat);
            delete cache_layer;
        }
        
        
        /* free cacheServerThread and cacheLayerThread */
        for (auto cache_server_thread: cache_server_threads)
            delete cache_server_thread;
        for (auto cache_layer_thread: cache_layer_threads)
            delete cache_layer_thread;
    }
}
