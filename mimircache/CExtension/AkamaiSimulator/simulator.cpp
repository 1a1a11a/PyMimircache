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
    
    
    void log_akamai_stat(akamaiStat *akamai_stat, bool *stop_flag,
                         const std::string log_folder){
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::ofstream ofs(std::string(log_folder)+"/akamai_stat");
        FILE* ofile = fopen((std::string(log_folder)+"/akamai_stat2").c_str(), "w");
        if (ofile == NULL)
            printf("error %s\n", strerror(errno));
        
        char buf[1024];
        info("akamai_stat_log thread started, output %s\n",
             (std::string(log_folder)+"/akamai_stat").c_str());
        while (! *stop_flag){
            std::this_thread::sleep_for(std::chrono::seconds(2));
            sprintf(buf, "%lf\t%lf\t%lf\t%lf\t%lf\n",
                    akamai_stat->get_avg_latency(),
                    akamai_stat->get_hr_L1(),
                    akamai_stat->get_hr_L2(),
                    akamai_stat->get_traffic_to_origin(),
                    akamai_stat->get_traffic_between_first_second_layer());
            //            std::cout<<"akamai stat: "<<buf;
            fprintf(ofile, "%s", buf);
            ofs << std::string(buf);
            memset(buf, 0, 1024);
        }
        info("akamai_stat_log thread finished\n");
        fclose(ofile);
        ofs.close();
    }
    
    
    
    
    akamaiStat* akamai_run(std::vector<std::string> traces,
                           double *boundaries,
                           unsigned long* cache_sizes,
                           unsigned long akamai_data_type,
                           const std::string log_folder,
                           bool dynamic_boundary_flag){
        
        unsigned int i;
        unsigned long num_servers = traces.size();
        
#ifdef TIME_SYNCHRONIZATION
        info("time synchronization between layers are enabled, "
             "with max time difference %d\nPAY ATTENTION: "
             "if the trace log is not cleaned and have beginning time difference "
             "larger than max time difference, this will cause deadlock\n",
             SYNCHRONIZATION_TIME_DIFF);
        
#endif
        
        reader_t *readers[num_servers];
        
        csvReader_init_params* reader_init_params;
        if (akamai_data_type == 3){
            debug("read data using akamai dataType 3\n");
            reader_init_params = AKAMAI3_CSV_PARAM_INIT;
        }
        else if (akamai_data_type == 0){
            debug("read data using akamai dataType 0\n");
            reader_init_params = AKAMAI0_CSV_PARAM_INIT;
        }
        else if (akamai_data_type == 1 || akamai_data_type == 2){
            reader_init_params = AKAMAI1_CSV_PARAM_INIT;
            debug("read data using akamai dataType 1\n");
        }
        else {
            error_msg("unknown traceType %lu\n", akamai_data_type);
            abort();
        }
        
        /* threads for cacheServerThread and cacheLayerThread */
        std::thread *t;
        std::vector<std::thread*> threads;
        
        bool thread_stop_flag = false;
        akamaiSimulator::akamaiStat* akamai_stat =
        new akamaiSimulator::akamaiStat(L1_LATENCY, L2_LATENCY, ORIGIN_LATENCY);
        //        std::thread log_akamai_stat_thread(log_akamai_stat,
        //                                           akamai_stat,
        //                                           &thread_stop_flag,
        //                                           log_folder);
        
        try{
            
            
            /* initialize cacheServers and readers */
            if (dynamic_boundary_flag){
                info("cache server dynamic boundary on\n");
            }
            else
                info("cache server dynamic boundary off\n");
            
            akamaiSimulator::cacheServer* cache_servers[num_servers];
            akamaiSimulator::cacheServerThread *cache_server_threads[num_servers];
            for (i=0; i<num_servers; i++){
                cache_servers[i] = new akamaiSimulator::cacheServer(i, akamai_stat,
                                                                    dynamic_boundary_flag,
                                                                    cache_sizes[i],
                                                                    boundaries, e_LRU,
                                                                    'c', 0, NULL);
                readers[i] = setup_reader(traces.at(i).c_str(), CSV, 'c', 0, 0,
                                          (void*) reader_init_params);
            }
            
            /* initialize cacheLayer and cacheLayerThread */
            akamaiSimulator::cacheLayer* cache_layers[NUM_CACHE_LAYERS-1];
            akamaiSimulator::cacheLayerThread *cache_layer_threads[NUM_CACHE_LAYERS-1];
            for (i=0; i<NUM_CACHE_LAYERS-1; i++){
                cache_layers[i] = new akamaiSimulator::cacheLayer(cache_servers, num_servers, akamai_stat, 2 + i);
                cache_layer_threads[i] = new akamaiSimulator::cacheLayerThread(cache_layers[i]);
            }
            
            
            /* build threads for cacheLayerThread */
            for (i=0; i<NUM_CACHE_LAYERS-1; i++){
                t = new std::thread(&akamaiSimulator::cacheLayerThread::run, cache_layer_threads[i], 60, log_folder);
#ifdef THREAD_AFFINITY
                set_thread_affinity(t->native_handle());
#endif
                threads.push_back(t);
            }
            
            /* build threads for cacheServerThread */
            for (i=0; i<num_servers; i++){
                cache_server_threads[i] = new akamaiSimulator::cacheServerThread(cache_servers[i],
                                                                                 cache_layer_threads,
                                                                                 readers[i], true);
                t = new std::thread(&akamaiSimulator::cacheServerThread::run, cache_server_threads[i], 60, log_folder);
#ifdef THREAD_AFFINITY
                set_thread_affinity(t->native_handle());
#endif
                threads.push_back(t);
            }
            
            
            info("main thread waiting for all cache servers and layers\n");
            /* wait for cache server and cache layer to finish computation */
            for (auto it=threads.begin(); it<threads.end(); it++){
                (**it).join();
                delete *it;
            }
            info("all cache servers and layers joined\n"); 
            
            /* stop akamai_stat_log thread */
            thread_stop_flag = true;
            //        log_akamai_stat_thread.join();
            
            
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
            
            
        }catch (std::exception &e) {
            std::cerr<<e.what()<<std::endl;
            print_stack_trace();
        }
        
        
        return akamai_stat;
    }
}
